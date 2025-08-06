# Installation

Please follow the instructions for installing the conda environments and dependencies, referring to [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) repository.

---
1. Git clone this repo and `cd` into it.

```bash
    git clone https://github.com/GaavaMa/Causal-Diffusion-Policy.git
```

---
1. Create conda environment.
```bash
    conda create -n cdp python=3.8
    conda activate cdp
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---
3. Install CDP.
```bash
    cd Causal-Diffusion-Policy && pip install -e . && cd ..
```

---
4. Install mujoco.
```bash
    cd ~/.mujoco
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate

    tar -xvzf mujoco210.tar.gz
```
add the following lines to your shell startup script (typically `~/.bashrc`):
```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    export MUJOCO_GL=egl
```
run `source ~/.bashrc` and then install mujoco-py (in the folder of `$YOUR_PATH_TO_THIRD_PARTY`):
```bash
    cd YOUR_PATH_TO_THIRD_PARTY
    cd mujoco-py-2.1.2.14
    pip install -e .
    cd ../..
```

----
5. Install simulation environments.
```bash
    pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

    cd YOUR_PATH_TO_THIRD_PARTY
    cd dexart-release && pip install -e . && cd ..
    cd gym-0.21.0 && pip install -e . && cd ..
    cd Metaworld && pip install -e . && cd ..
    cd rrl-dependencies && pip install -e mj_envs/. && pip install -e mjrl/. && cd ..
```
download assets from [Google Drive](https://drive.google.com/file/d/1DxRfB4087PeM3Aejd6cR-RQVgOKdNrL4/view?usp=sharing), unzip it, and place it in `$YOUR_PATH_TO_THIRD_PARTY/dexart-release/assets`. 

download Adroit RL experts from [OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS), unzip it, and place the `ckpts` folder under `$YOUR_PATH_TO_THIRD_PARTY/VRL3/`.

---
6. Install pytorch3d.
```bash
    cd third_party/pytorch3d_simplified && pip install -e . && cd ..
```

---
7. Install other necessary packages.
```bash
    pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor
```

---
8. Install our visualizer for pointclouds.
```bash
    pip install kaleido plotly
    cd visualizer && pip install -e . && cd ..
```