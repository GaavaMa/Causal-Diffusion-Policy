if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
from hydra import compose, initialize
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.env_runner.base_runner import BaseRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg1: OmegaConf, cfg2: OmegaConf, output_dir=None):
        self.cfg1 = cfg1
        self.cfg2 = cfg2
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg1.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if cfg1.name == "train_dp3":
            from diffusion_policy.policy.dp3 import DP3
        elif cfg1.name == "train_dp3_transformer":
            from diffusion_policy.policy.dp3_transformer import DP3
        else:
            from diffusion_policy.policy.dp3_transformer_causal import DP3

        # configure model
        self.model1: DP3 = hydra.utils.instantiate(cfg1.policy)

        self.ema_model1: DP3 = None
        if cfg1.training.use_ema:
            try:
                self.ema_model1 = copy.deepcopy(self.model1)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model1 = hydra.utils.instantiate(cfg1.policy)



        if cfg2.name == "train_dp3":
            from diffusion_policy.policy.dp3 import DP3
        elif cfg2.name == "train_dp3_transformer":
            from diffusion_policy.policy.dp3_transformer import DP3
        else:
            from diffusion_policy.policy.dp3_transformer_causal import DP3

        # configure model
        self.model2: DP3 = hydra.utils.instantiate(cfg2.policy)

        self.ema_model2: DP3 = None
        if cfg2.training.use_ema:
            try:
                self.ema_model2 = copy.deepcopy(self.model2)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model2 = hydra.utils.instantiate(cfg2.policy)


    def load_model_only(self, path=None, tag='latest', model_key='model1'):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        
        # 只加载指定模型的状态字典
        model_state_dict = payload['state_dicts'][model_key]
        getattr(self, model_key).load_state_dict(model_state_dict)
        
    def eval(self):
        # load the latest checkpoint
        lastest_ckpt_path1 = "/home/jiahuama/data/outputs/adroit_hammer-dp3_transformer-0112_seed0/checkpoints/latest.ckpt"
        cprint(f"Resuming from checkpoint {lastest_ckpt_path1}", 'magenta')
        with open(lastest_ckpt_path1, 'rb') as f:
            payload1 = torch.load(f, pickle_module=dill, map_location='cpu')
            self.ema_model1.load_state_dict(payload1['state_dicts']['ema_model'])
            # 提取 normalizer 的状态
            normalizer_state_dict = {}
            for key, value in payload1['state_dicts']['ema_model'].items():
                if key.startswith('normalizer.'):
                    normalizer_key = key[len('normalizer.'):]
                    normalizer_state_dict[normalizer_key] = value

            # 加载 normalizer 的状态
            self.ema_model1.normalizer.load_state_dict(normalizer_state_dict)
                            
        
        lastest_ckpt_path2 = "/home/jiahuama/data/outputs/adroit_hammer-dp3_transformer_causal-0112_seed0/checkpoints/latest.ckpt"
        cprint(f"Resuming from checkpoint {lastest_ckpt_path2}", 'magenta')
        with open(lastest_ckpt_path2, 'rb') as f:
            payload2 = torch.load(f, pickle_module=dill, map_location='cpu')
            self.ema_model2.load_state_dict(payload2['state_dicts']['ema_model'])
             # 提取 normalizer 的状态
            normalizer_state_dict = {}
            for key, value in payload2['state_dicts']['ema_model'].items():
                if key.startswith('normalizer.'):
                    normalizer_key = key[len('normalizer.'):]
                    normalizer_state_dict[normalizer_key] = value

            # 加载 normalizer 的状态
            self.ema_model2.normalizer.load_state_dict(normalizer_state_dict)

        # configure env
        cfg = copy.deepcopy(self.cfg1)
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir="/home/jiahuama/data/outputs/adroit_hammer-dp3_transformer-0112_seed0")
        assert isinstance(env_runner, BaseRunner)
        policy1 = self.model1
        if cfg1.training.use_ema:
            policy1 = self.ema_model1
        policy1.eval()
        policy1.cuda()

        policy2 = self.model2
        if cfg2.training.use_ema:
            policy2 = self.ema_model2
        policy2.eval()
        policy2.cuda()

        runner_log = env_runner.run2(policy1, policy2)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')    


def main(cfg1, cfg2):
    workspace = TrainDP3Workspace(cfg1, cfg2)
    workspace.eval()

if __name__ == "__main__":
    initialize(
        version_base=None,
        config_path="diffusion_policy/config"
    )

    cfg1 = compose(config_name="dp3_transformer")
    cfg2 = compose(config_name="dp3_transformer_causal")

    main(cfg1, cfg2)
