from typing import Union, Optional, Tuple
import logging
import torch
import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.cached_attention import TemporalCachedAttention
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

def modulate(x, shift, scale):
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // scale.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def gate(x, g):
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x

class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        n_obs_steps,
        n_action_steps,
        dropout: float = 0.1,
        mlp_ratio=4.0
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        self.t_norm1 = nn.LayerNorm(hidden_size, eps=1e-5, bias=True)
        self.t_attn1 = TemporalCachedAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads
        )
        self.t_dropout1 = nn.Dropout(dropout)
        
        self.t_norm2 = nn.LayerNorm(hidden_size, eps=1e-5, bias=True)
        self.t_attn2 = TemporalCachedAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads
        )
        self.t_dropout2 = nn.Dropout(dropout)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.t_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size)
            )
        self.t_norm3 = nn.LayerNorm(hidden_size, eps=1e-5, bias=True)
    
    def forward(self, x, cond, causal_mask):
        # self attn
        x_sttn, _, _ = self.t_attn1(self.t_norm1(x), self.t_norm1(x[:, :self.n_obs_steps]), self.t_norm1(x[:, self.n_obs_steps:]), causal_mask=causal_mask)
        x = x + self.t_dropout1(x_sttn)
        
        # cross attn
        x_cttn, _, _ = self.t_attn2(self.t_norm2(x), None, cond)
        x = x + self.t_dropout2(x_cttn)
        
        # feedback
        x = x + self.t_mlp(self.t_norm3(x))
        return x

    def forward_with_cache(self, x, cond, k_cache, v_cache, cond_k_cache, cond_v_cache, causal_mask, diff_step_idx):
        # self attn
        x_past = None
        x_cur = x
        if diff_step_idx  == 0:
            if k_cache is None:
                x_past = x[:, :self.n_obs_steps]
                x_cur = x[:, self.n_obs_steps:]
            else:
                x_past = x[:, :self.n_action_steps]
                x_cur = x[:, self.n_action_steps:]

            x_past = self.t_norm1(x_past)
            
        x_sttn, k, v = self.t_attn1(self.t_norm1(x), x_past, self.t_norm1(x_cur), k_cache=k_cache, v_cache=v_cache, causal_mask=causal_mask)
        x = x + self.t_dropout1(x_sttn)
        
        # cross attn
        x_cttn, cond_k, cond_v = self.t_attn2(self.t_norm2(x), None, cond, k_cache=cond_k_cache, v_cache=cond_v_cache)
        x = x + self.t_dropout2(x_cttn)
        
        # feedback
        x = x + self.t_mlp(self.t_norm3(x))
        return x, k[:, :self.n_obs_steps], v[:, :self.n_obs_steps], cond_k[:, :-1], cond_v[:, :-1]

class CausalTransformer(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int,
            n_action_steps: int,
            cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
        ) -> None:
        super().__init__()
        
        self.n_emb = n_emb
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        
        # input embedding
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.drop = nn.Dropout(p_drop_emb)
        
        # position embedding
        self.register_buffer("pos_emb", self.get_temporal_pos_embed())

        # condition embedding
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.time_emb = nn.Sequential(
            Rearrange('b t -> (b t)'),  # reshape to (B*T,)
            SinusoidalPosEmb(n_emb),
            Rearrange('(b t) d -> b (t d)', d=n_emb, t = horizon-n_obs_steps),    # reshape to (B, T*dsed)
            nn.Linear((horizon-n_obs_steps) * n_emb, n_emb * 4),
            nn.Mish(),
            nn.Linear(n_emb * 4, n_emb),
        )
        
        # causal mask
        self.chunked_causal_mask = self.create_chunked_causal_mask()
        
        # attn
        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(
                n_emb,
                n_head,
                n_obs_steps,
                n_action_steps,
                dropout=p_drop_attn,
            )
            for _ in range(n_layer)
        ])
        self.n_layer = n_layer
        
        # output
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    
    def create_chunked_causal_mask(self):
        num_full_chunks = self.horizon // self.n_action_steps  # 完整块的数量
        last_chunk_size = self.horizon % self.n_action_steps  # 最后一个块的大小
        
        num_obs_chunks = self.n_obs_steps // self.n_action_steps

        # 创建一个全零矩阵
        mask = torch.zeros(self.horizon, self.horizon)

        # 填充块内部为 0（可见），块外部为负无穷（不可见）
        for i in range(num_full_chunks):
            start_i = i * self.n_action_steps
            end_i = start_i + self.n_action_steps

            for j in range(num_full_chunks):
                start_j = j * self.n_action_steps
                end_j = start_j + self.n_action_steps

                if i < num_obs_chunks:
                    if i == j:
                        mask[start_i:end_i, start_j:end_j] = 0.0
                    else:
                        mask[start_i:end_i, start_j:end_j] = float('-inf')
                else:
                    if i >= j:  # 当前块及之前的块可见
                        mask[start_i:end_i, start_j:end_j] = 0.0
                    else:  # 当前块之后的块不可见
                        mask[start_i:end_i, start_j:end_j] = float('-inf')

        # 处理最后一个块
        if last_chunk_size > 0:
            start_i = num_full_chunks * self.n_action_steps
            end_i = start_i + last_chunk_size

            for j in range(num_full_chunks + 1):
                start_j = j * self.n_action_steps
                end_j = start_j + (self.n_action_steps if j < num_full_chunks else last_chunk_size)

                if start_i >= start_j:  # 当前块及之前的块可见
                    mask[start_i:end_i, start_j:end_j] = 0.0
                else:  # 当前块之后的块不可见
                    mask[start_i:end_i, start_j:end_j] = float('-inf')

        return mask
  
    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(self.n_emb, self.horizon, scale=1.0)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed
            
    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb,
            TemporalTransformerBlock,
            nn.ModuleList,
            nn.Mish,
            nn.SiLU,
            nn.GELU,
            TemporalCachedAttention,
            CausalTransformer,
            nn.Sequential,
            nn.Identity,
            Rearrange)
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
        

    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, TemporalCachedAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Identity)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("_dummy_variable")
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward_with_cache(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int],
        tpe_start: int,
        cond: torch.Tensor,
        diff_step_idx: int,
        k_cache,
        v_cache,
        cond_k_cache,
        cond_v_cache,
        **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B, T)
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """

        # 0. position embeding idx   
        T = sample.shape[1]
        tpe_ids = [i % self.horizon for i in range(tpe_start, tpe_start+T)]
        
        T_cond = cond.shape[1]
        tpe_cond_ids = [i % self.horizon for i in range(tpe_start, tpe_start+T_cond)]
        
        # 1. input embeding
        token_embeddings = self.input_emb(sample)
        position_embeddings = self.pos_emb[:, tpe_ids, :]
        x = self.drop(token_embeddings + position_embeddings)
    
        # 2. condition embeding
        cond_embeddings = self.cond_obs_emb(cond)
        position_embeddings = self.pos_emb[:, tpe_cond_ids, :]
        cond_embeddings = self.drop(cond_embeddings + position_embeddings)
        time_emb = self.time_emb(timestep).unsqueeze(1)
        cond_embeddings = torch.cat([cond_embeddings, time_emb], dim=1)

        # 3. Transformer
        self.chunked_causal_mask = self.chunked_causal_mask.to(x.device)
        # self attn
        if diff_step_idx == 0:
            B, _, D =cond_embeddings.shape
            new_k_cache = torch.zeros((B, self.n_obs_steps, self.n_layer, D), dtype=cond_embeddings.dtype, device=cond_embeddings.device)
            new_v_cache = torch.zeros_like(new_k_cache)
            new_cond_k_cache = torch.zeros_like(new_k_cache)
            new_cond_v_cache = torch.zeros_like(new_k_cache)
            
            if k_cache is None:
                for i, block in enumerate(self.blocks):
                    x, new_k_cache[:, :, i, :], new_v_cache[:, :, i, :], new_cond_k_cache[:, :, i, :], new_cond_v_cache[:, :, i, :] = \
                        block.forward_with_cache(x, cond_embeddings, None, None, None, None, causal_mask=self.chunked_causal_mask, diff_step_idx=0)    
                x = self.head(self.ln_f(x[:, self.n_obs_steps:]))
            else:
                assert k_cache is not None
                assert v_cache is not None
                assert cond_k_cache is not None
                assert cond_v_cache is not None
                
                for i, block in enumerate(self.blocks):
                    x, new_k_cache[:, :, i, :], new_v_cache[:, :, i, :], new_cond_k_cache[:, :, i, :], new_cond_v_cache[:, :, i, :] = \
                        block.forward_with_cache(x, cond_embeddings, k_cache[:, :, i, :], v_cache[:, :, i, :], cond_k_cache[:, :, i, :], cond_v_cache[:, :, i, :], causal_mask=self.chunked_causal_mask[-x.shape[1]:, :], diff_step_idx=0) 
                x = self.head(self.ln_f(x[:, self.n_action_steps:]))
            
            return x, new_k_cache, new_v_cache, new_cond_k_cache, new_cond_v_cache
                
        else:
            for i, block in enumerate(self.blocks):
                x, _, _, _, _ = block.forward_with_cache(x, cond_embeddings, k_cache[:, :, i, :], v_cache[:, :, i, :], cond_k_cache[:, :, i, :], cond_v_cache[:, :, i, :], causal_mask=self.chunked_causal_mask[-x.shape[1]:, :], diff_step_idx=diff_step_idx) 
            
            x = self.head(self.ln_f(x))
            return x

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int],
        tpe_start: torch.Tensor,
        cond: torch.Tensor,
        **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B, T)
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """

        # 0. position embeding idx
        # tpe_ids = [i % self.horizon for i in range(tpe_start, tpe_start+self.horizon)]
        # tpe_cond_ids = [i % self.horizon for i in range(tpe_start, tpe_start+self.n_obs_steps)]
        
        offsets = torch.arange(self.horizon, device=tpe_start.device)  # 形状为 (self.horizon,)
        cond_offsets = torch.arange(self.n_obs_steps, device=tpe_start.device)  # 形状为 (self.n_obs_steps,)
        tpe_start_expanded = tpe_start.unsqueeze(-1)  # 形状为 (batch_size, 1)
        tpe_ids = ((tpe_start_expanded + offsets) % self.horizon).unsqueeze(-1).expand(-1, -1, self.n_emb)  # 形状为 (batch_size, self.horizon)
        tpe_cond_ids = ((tpe_start_expanded + cond_offsets) % self.horizon).unsqueeze(-1).expand(-1, -1, self.n_emb) 

        # 1. input embeding
        token_embeddings = self.input_emb(sample)
        # position_embeddings = self.pos_emb[:, tpe_ids, :]
        pos_emb = self.pos_emb.repeat(tpe_ids.shape[0], 1, 1)
        position_embeddings = torch.gather(pos_emb, 1, tpe_ids)
        x = self.drop(token_embeddings + position_embeddings)
    
        # 2. condition embeding
        cond_embeddings = self.cond_obs_emb(cond)
        # position_embeddings = self.pos_emb[:, tpe_cond_ids, :]
        position_embeddings = torch.gather(pos_emb, 1, tpe_cond_ids)
        cond_embeddings = self.drop(cond_embeddings + position_embeddings)
        time_emb = self.time_emb(timestep).unsqueeze(1)
        cond_embeddings = torch.cat([cond_embeddings, time_emb], dim=1)

        # 3. Transformer
        self.chunked_causal_mask = self.chunked_causal_mask.to(x.device)
        for block in self.blocks:
            x = block(x, cond_embeddings, self.chunked_causal_mask)
            
        # 4. output
        x = self.head(self.ln_f(x))
        return x


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb