from typing import Dict, Tuple
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, repeat

from diffusion_policy.policy.schedulers import DDPMTEDiScheduler
from termcolor import cprint
import copy

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_policy import BasePolicy
from diffusion_policy.model.diffusion.causal_transformer import CausalTransformer
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.model_util import print_params
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

class DP(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMTEDiScheduler,
            obs_encoder: MultiImageObsEncoder,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # arch
            n_layer=8,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            obs_as_cond=True,
            use_point_crop=False,
            condition_type="cross_attention",
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            temporally_constant_weight=0.0,
            temporally_increasing_weight=0.0,
            temporally_random_weights=0.0,
            chunk_wise_weight=1.0,
            buffer_init="zero",
            # parameters passed to step
            with_causal=False,
            causal_condition_noise_weight=15.0,
            training_mode_thres=0.2,
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = obs_feature_dim + action_dim
        cond_dim = 0
        if obs_as_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                cond_dim = obs_feature_dim
            else:
                cond_dim = obs_feature_dim * n_obs_steps
        output_dim = input_dim


        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[TransformerBasedDP3] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[TransformerBasedDP3] pointnet_type: {self.pointnet_type}", "yellow")


        model = CausalTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=with_causal
        )
        
        self.normalizer = LinearNormalizer()
        if (horizon < n_obs_steps - 1 + n_action_steps) or (horizon % 4 != 0):
            raise ValueError(
                "Horizon must be longer than (To-1) + Ta \n Also, the horizon must be divisible by 4 for the UNet to accept it."
                % (horizon - n_obs_steps, n_action_steps)
            )
            
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # TEDi action buffer
        self.action_buffer = None
        self.buffer_diff_steps = None

        # KV cache
        self.k_cache, self.v_cache = None, None
        self.obs_k_cache, self.obs_v_cache = None, None
        self.cache_start_idx = 0

        self.temporally_constant_weight = temporally_constant_weight
        self.temporally_increasing_weight = temporally_increasing_weight
        self.temporally_random_weights = temporally_random_weights
        self.chunk_wise_weight = chunk_wise_weight

        # Check that the weights sum to 1
        assert math.isclose(
            self.temporally_constant_weight
            + self.temporally_increasing_weight
            + self.temporally_random_weights
            + self.chunk_wise_weight,
            1,
            rel_tol=1e-2,
        )

        self.buffer_init = buffer_init
        assert self.buffer_init in ["zero", "constant", "denoise"]
        
        # causal
        self.with_causal = with_causal
        self.causal_condition_noise_weight = causal_condition_noise_weight
        self.training_mode_thres = training_mode_thres

        print_params(self)

    # ========= common  ============
    def reset(self):
        self.reset_buffer()
    
    def reset_buffer(self):
        self.action_buffer = None
        self.buffer_diff_steps = None

        self.k_cache, self.v_cache = None, None
        self.obs_k_cache, self.obs_v_cache = None, None
        self.cache_start_idx = 0

    def push_buffer(self, new_value, new_sigma_indices):
        """
        Adds the new value to the end of the action buffer like a FIFO queue.
        Also add the corresponding values to buffer_sigma_indices.
        Args:
            new_value: (B, N, Da)
        """
        self.action_buffer = torch.cat([self.action_buffer, new_value], dim=1)
        self.buffer_diff_steps = torch.cat([self.buffer_diff_steps, new_sigma_indices], dim=1)
    
    def pop_cache(self):
        if self.k_cache is not None:
            assert self.v_cache is not None
            assert self.obs_k_cache is not None
            assert self.obs_v_cache is not None
            
            self.k_cache = self.k_cache[:, self.n_action_steps:]
            self.v_cache = self.v_cache[:, self.n_action_steps:]
            self.obs_k_cache = self.obs_k_cache[:, self.n_action_steps:]
            self.obs_v_cache = self.obs_v_cache[:, self.n_action_steps:]
            self.cache_start_idx += self.n_action_steps

    # ========= inference  ============
    @torch.no_grad()
    def initialize_buffer(
        self,
        shape,
        condition_data,
        condition_mask,
        cond=None,
        generator=None,
        **kwargs,
    ):
        """
        Args:
            shape: The shape of the action buffer, for obs_as_global_cond, it is (B, T, Da), for inpainting (B, T, Da+Do)
            condition_data: The conditioning data, shape corresponds to shape
            condition_mask: The mask for the conditioning data, shape corresponds to shape
        """
        # if first step, initialize action buffer as pure noise with std. dev.
        # sigma_max, and do T_a number of steps

        B, Tp, _ = shape
        Ta = self.n_action_steps
        To = self.n_obs_steps
        N = self.num_inference_steps
        scheduler = self.noise_scheduler
        if self.action_buffer is None:
            trajectory = torch.randn(
                size=shape, dtype=self.dtype, device=self.device, generator=generator
            )

            if self.with_causal:
                condition_data[:, :To, :self.action_dim] = trajectory[:, :To, :self.action_dim] / self.causal_condition_noise_weight

            # set step values
            scheduler.set_timesteps(self.num_inference_steps)

            # 1. Denoise buffer
            for t in scheduler.timesteps.to(device=self.device, dtype=torch.long):
                # 1. apply conditioning
                trajectory[condition_mask] = condition_data[condition_mask]

                # 2. predict model output
                model_output = self.model(
                    trajectory,
                    t.repeat(B, Tp),
                    cond=cond,
                )

                # 3. compute previous image: x_t -> x_t-1
                trajectory = scheduler.step(
                    model_output,
                    t.repeat(B, Tp),
                    trajectory,
                    generator=generator,
                    **kwargs,
                ).prev_sample

            self.action_buffer = trajectory

            # 2. Find the diffusion steps (B, T) for each element in the buffer

            # Calculate chunk indices i
            num_complete_chunks = math.floor((Tp - To) / Ta)
            incomplete_index = Ta * num_complete_chunks + To
            indices = torch.arange(0, Tp, device=self.device)
            chunk_indices = torch.div((indices - To), Ta, rounding_mode="floor")
            chunk_indices = torch.where(indices < To, 0, chunk_indices)
            chunk_indices = torch.where(
                indices >= incomplete_index, num_complete_chunks - 1, chunk_indices
            )
            chunk_indices = chunk_indices.repeat(B, 1)  # (B,T)

            # Calculate corresponding diffusion levels
            diff_steps = (
                torch.floor((N * (chunk_indices + 1)) / num_complete_chunks) - 1
            ).long()
            self.buffer_diff_steps = diff_steps

            # 3. Noise action buffer corresponding to the sigma_indices
            # Note: we don't noise the observation steps + 1st chunk
            buffer_to_be_noised = self.action_buffer[:, Ta + To :]
            diff_steps = self.buffer_diff_steps[:, Ta + To :]
            noise = torch.randn(
                size=buffer_to_be_noised.shape, dtype=self.dtype, device=self.device
            )
            self.action_buffer[:, Ta + To :] = scheduler.add_noise(
                buffer_to_be_noised, noise, diff_steps
            )

            # Apply conditioning
            if self.with_causal:
                condition_data[:, :To, :self.action_dim] = self.action_buffer[:, :To, :self.action_dim]

            self.action_buffer[condition_mask] = condition_data[condition_mask]

    def initialize_buffer_as_zero(self, shape, condition_data, condition_mask, generator=None):
        N = self.num_inference_steps
        B, Tp, _ = shape
        Ta = self.n_action_steps
        To = self.n_obs_steps

        action_buffer = torch.zeros(shape, dtype=self.dtype, device=self.device)
        scheduler = self.noise_scheduler

        # Find diffusion steps
        # Calculate chunk indices i
        num_complete_chunks = math.floor((Tp - To) / Ta)
        incomplete_index = Ta * num_complete_chunks + To
        indices = torch.arange(0, Tp, device=self.device)
        chunk_indices = torch.div((indices - To), Ta, rounding_mode="floor")
        chunk_indices = torch.where(indices < To, 0, chunk_indices)
        chunk_indices = torch.where(
            indices >= incomplete_index, num_complete_chunks - 1, chunk_indices
        )
        chunk_indices = chunk_indices.repeat(B, 1)  # (B,T)

        diff_steps = (
            torch.floor((N * (chunk_indices + 1)) / num_complete_chunks) - 1
        ).long()
        self.buffer_diff_steps = diff_steps

        # Add noise
        noise = torch.randn(
            size=action_buffer.shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        self.action_buffer = scheduler.add_noise(action_buffer, noise, diff_steps)
        
        if self.with_causal:
            condition_data[:, :To, :self.action_dim] = noise[:, :To, :self.action_dim] / self.causal_condition_noise_weight
        
        self.action_buffer[condition_mask] = condition_data[condition_mask]

    def initialize_buffer_as_constant(self, shape, cond_data, generator=None):
        N = self.num_inference_steps
        B, Tp, _ = shape
        Ta = self.n_action_steps
        To = self.n_obs_steps
        Da = self.action_dim

        first_agent_pos = cond_data[:, 0, :Da]

        constant_action = first_agent_pos.unsqueeze(1).repeat(1, Tp, 1)
        action_buffer = torch.zeros(shape, dtype=self.dtype, device=self.device)
        action_buffer[:, :, :Da] = constant_action
        scheduler = self.noise_scheduler

        # Find diffusion steps
        # Calculate chunk indices i
        num_complete_chunks = math.floor((Tp - To) / Ta)
        incomplete_index = Ta * num_complete_chunks + To
        indices = torch.arange(0, Tp, device=self.device)
        chunk_indices = torch.div((indices - To), Ta, rounding_mode="floor")
        chunk_indices = torch.where(indices < To, 0, chunk_indices)
        chunk_indices = torch.where(
            indices >= incomplete_index, num_complete_chunks - 1, chunk_indices
        )
        chunk_indices = chunk_indices.repeat(B, 1)  # (B,T)

        diff_steps = (
            torch.floor((N * (chunk_indices + 1)) / num_complete_chunks) - 1
        ).long()
        self.buffer_diff_steps = diff_steps

        # Add noise
        noise = torch.randn(
            size=action_buffer.shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        self.action_buffer = scheduler.add_noise(action_buffer, noise, diff_steps)

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        """
        Sample from the diffusion model conditioned on condition_data. Unlike EDM, we just do one
        condition_data: (B, T, C) The conditioning data
        Returns:
            action_pred: (B, T, Da) The predicted action including the observation steps
        """
        Tp = condition_data.shape[1]
        Ta = self.n_action_steps
        To = self.n_obs_steps

        model = self.model
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)
        
        if self.with_causal:
            condition_data[:, :To, :self.action_dim] = self.action_buffer[:, :To, :self.action_dim]
            self.pop_cache()
        
        # We need to denoise the first T_o+T_a steps, i.e. push their sigma to 0
        diff_step_idx = 0
        while torch.max(self.buffer_diff_steps[:, To + Ta - 1]) != -1:
            # 1. apply conditioning
            self.action_buffer[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            diff_steps = self.buffer_diff_steps[:, To:]
            if diff_step_idx == 0:
                if self.cache_start_idx == 0:
                    sample = self.action_buffer
                    tpe_start = 0
                    sample_cond = cond
                else:
                    sample = self.action_buffer[:, To - Ta:]
                    tpe_start = self.cache_start_idx + To - Ta
                    sample_cond = cond[:, To - Ta:]
        
                model_output, self.k_cache, self.v_cache, self.obs_k_cache, self.obs_v_cache = model.forward_with_cache(
                    sample,
                    diff_steps,
                    tpe_start=tpe_start,
                    cond=sample_cond,
                    diff_step_idx=diff_step_idx,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    cond_k_cache=self.obs_k_cache,
                    cond_v_cache=self.obs_v_cache
                )
            else:
                sample = self.action_buffer[:, To:]
                tpe_start = self.cache_start_idx + To
                sample_cond = cond[:, To:]
                model_output = model.forward_with_cache(
                    sample,
                    diff_steps,
                    tpe_start=tpe_start,
                    cond=sample_cond,
                    diff_step_idx=diff_step_idx,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    cond_k_cache=self.obs_k_cache,
                    cond_v_cache=self.obs_v_cache
                )

            # 3. compute previous image: x_t -> x_t-1
            self.action_buffer[:, self.n_obs_steps:] = scheduler.step(
                model_output,
                self.buffer_diff_steps[:, self.n_obs_steps:],
                self.action_buffer[:, self.n_obs_steps:],
                generator=generator,
                **kwargs,
            ).prev_sample

            # 4. update the diffusion step for the buffer
            self.buffer_diff_steps = torch.clamp(self.buffer_diff_steps - 1, min=-1)
            
            # 5. update diff_step_idx
            diff_step_idx += 1

        # Finally, make sure conditioning is enforced
        self.action_buffer[condition_mask] = condition_data[condition_mask]

        # Return whole buffer as prediction, we slice later
        action_pred = self.action_buffer  # (B, T, Da) or (B, T, Da+Do)

        # Remove the first T_a steps from the buffer,
        # as they are not needed for the next iteration
        self.action_buffer = self.action_buffer[:, Ta:]
        self.buffer_diff_steps = self.buffer_diff_steps[:, Ta:]

        # Remove excess steps (not part of complete chunks) that are partly denoised
        if (Tp - To) % Ta != 0:
            self.action_buffer = self.action_buffer[:, : -((Tp - To) % Ta)]
            self.buffer_diff_steps = self.buffer_diff_steps[:, : -((Tp - To) % Ta)]

        # Add the actions with sigma_max to noise to the end
        B = condition_data.shape[0]
        num_new_actions = Ta + ((Tp - To) % Ta)
        new_noise = torch.randn(
            size=(B, num_new_actions, self.action_buffer.shape[-1]),
            dtype=self.dtype,
            device=self.device,
        )
        new_sigma_indices = torch.ones(
            size=(B, num_new_actions), dtype=torch.long, device=self.device
        ) * (self.num_inference_steps - 1)
        self.push_buffer(new_noise, new_sigma_indices)

        # Noise the first T_o steps
        noise = torch.randn(
            size=self.action_buffer[:, :To].shape,
            dtype=self.dtype,
            device=self.device,
        )
        if self.with_causal:
            # self.action_buffer[:, To-Ta:To] = self.action_buffer[:, To-Ta:To] + noise[:, To-Ta:To] / self.causal_condition_noise_weight
            pass
        else:
            next_chunk_diff_step = self.buffer_diff_steps[:, To]  # (B, 1)
            
            next_chunk_diff_step = next_chunk_diff_step.unsqueeze(1).repeat(1, To)
            self.buffer_diff_steps[:, :To] = next_chunk_diff_step

            self.noise_scheduler.add_noise(
                self.action_buffer[:, : To], noise, next_chunk_diff_step
            )

        return action_pred

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']

        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            shape = (B, T, Da)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            if self.with_causal:
                cond_mask[:,:To] = True
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            if self.with_causal:
                cond_mask[:,:To] = True
            else:
                cond_mask[:,:To,Da:] = True

        ## SAMPLING
        # If the buffer is reset, initialize it
        if self.action_buffer is None:
            if self.buffer_init == "zero":
                self.initialize_buffer_as_zero(shape, cond_data, cond_mask, generator=None)
            elif self.buffer_init == "constant":
                self.initialize_buffer_as_constant(shape, cond_data, generator=None)
            elif self.buffer_init == "denoise":
                self.initialize_buffer(
                    shape,
                    cond_data,
                    cond_mask,
                    cond=cond,
                    generator=None,
                    **self.kwargs,
                )
            else:
                raise ValueError(
                    f"Unsupported buffer initialization {self.buffer_init}"
                )

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def get_diff_steps_training(self, B, Tp, To, Ta):
        """
        Args:
            B: batch size
            T: number of timesteps
            To: number of observation steps
            Ta: number of action steps
        Returns:
            diff_steps: (T,) or (B, T) The indices of the noise injection regime for each timestep
        """
        N = self.noise_scheduler.config.num_train_timesteps

        # Sample the noise injection regime for this batch from a categoriecal distribution
        # over the different noise injection regimes.
        probabilities = torch.tensor(
            [
                self.temporally_constant_weight,
                self.temporally_increasing_weight,
                self.temporally_random_weights,
                self.chunk_wise_weight,
            ]
        )

        # Sample from the categorical distribution based on the defined probabilities
        noise_regime = torch.multinomial(
            probabilities, num_samples=1, replacement=True
        ).item()

        # Select the noise injection regime based on the sampled value
        if noise_regime == 0:  # Constant scheme
            diff_steps = (
                torch.randint(0, N, (B, 1), device=self.device).long().repeat(1, Tp)
            )
        elif noise_regime == 1:  # Linearly increasing scheme
            diff_steps = (
                torch.linspace(0, N - 1, Tp, device=self.device, dtype=self.dtype)
                .long()
                .repeat(B, 1)
            )
        elif noise_regime == 2:  # Random scheme
            diff_steps = timesteps = torch.randint(
                0, N, (B, Tp), device=self.device
            ).long()
        elif noise_regime == 3:  # Chunk-wise scheme

            num_complete_chunks = math.floor((Tp - To) / Ta)
            incomplete_index = Ta * num_complete_chunks + To

            # Calculate chunk indices i
            indices = torch.arange(0, Tp, device=self.device)
            chunk_indices = torch.div((indices - To), Ta, rounding_mode="floor")
            chunk_indices = torch.where(indices < To, 0, chunk_indices)
            chunk_indices = torch.where(
                indices >= incomplete_index, num_complete_chunks - 1, chunk_indices
            )
            chunk_indices = chunk_indices.repeat(B, 1)  # (B,T)

            # 1. Sample j from [0, upper_limit]
            upper_limit = math.floor(N / num_complete_chunks)
            j = torch.randint(
                0, upper_limit, (B, 1), device=self.device
            )  # Don't include N/h, as that corresponds to a clean first chunk.
            j = j.repeat(1, Tp)  # (B,T)

            # 2. Evaluate k(i,j)=round((N-1) * ((i+1)/h)) - j
            diff_steps = (
                torch.floor((N * (chunk_indices + 1)) / num_complete_chunks) - 1 - j
            ).long()

        return diff_steps

    def compute_loss(self, batch):
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()
        
        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        B = trajectory.shape[0]
        T = trajectory.shape[1]
        To = self.n_obs_steps
        Ta = self.n_action_steps

        # Get noise on shape (B, T, D)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        # Get sigmas for each temporal step
        diff_steps = self.get_diff_steps_training(B, T, To, Ta)
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, diff_steps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        # start_idx = 0
        # noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # if self.with_causal:
        #     if torch.rand(1) > self.training_mode_thres:
        #         noisy_trajectory[:, :To, :self.action_dim] += noise[:, :To, :self.action_dim] / self.causal_condition_noise_weight
        #     else:
        #         start_idx = random.randint(1, self.n_obs_steps // self.n_action_steps)
        #         noisy_trajectory[:, :start_idx*self.n_action_steps, :self.action_dim] = noise[:, :start_idx*self.n_action_steps, :self.action_dim] / self.causal_condition_noise_weight
        #         cond[:, :start_idx*self.n_action_steps, :] = cond[:, start_idx*self.n_action_steps-1].unsqueeze(1).expand(-1, start_idx*self.n_action_steps, -1)
        #         noisy_trajectory[:, start_idx*self.n_action_steps:To, :self.action_dim] += noise[:, start_idx*self.n_action_steps:To, :self.action_dim] / self.causal_condition_noise_weight
        
        # if start_idx == 0:
        #     cache_start_idx = random.randint(self.n_obs_steps // self.n_action_steps, self.horizon // self.n_action_steps)*self.n_action_steps
        # else:
        #     cache_start_idx = (self.n_obs_steps // self.n_action_steps - start_idx)*self.n_action_steps

        start_idx = batch['sample_start_idx']
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        mask_temp = torch.arange(self.horizon, device=start_idx.device).unsqueeze(0).expand(B, -1)
        mask_act_obs = mask_temp < To
        mask_act_obs = mask_act_obs.unsqueeze(-1).expand(-1, -1, self.action_dim)
        mask = mask_temp < start_idx.unsqueeze(1)  # 形状为 (batch_size, To // self.n_action_steps)
        mask = mask.unsqueeze(-1).expand(-1, -1, self.action_dim)  # 扩展到 (batch_size, To // self.n_action_steps, action_dim)
        # 使用掩码操作
        noisy_trajectory[mask] = 0.0
        noisy_trajectory[mask_act_obs] += noise[mask_act_obs] / self.causal_condition_noise_weight
        
        cache_start_idx = torch.where(start_idx == 0, batch['buffer_start_idx'], To-start_idx)

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            diff_steps[:, To:],
            tpe_start=cache_start_idx,
            cond=cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        
        loss_dict = {
            'bc_loss': loss.item(),
        }

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")
        
        return loss, loss_dict
