---
applyTo: '**'
---

# DexScrew Quick Context (for future code agents)

## 1) Repo identity
- Type: Isaac Gym based secondary development project (custom tasks + custom PPO/ProprioAdapt).
- Core stack in code: `isaacgym`, `torch`, `hydra`, `omegaconf`, `wandb` (optional), `tensorboardX`.
- Not found as runtime dependency in code: `rl_games`, `isaacgymenvs`, `legged_gym`, `rsl_rl`, `mujoco`.

## 2) Main entry points
- Training entry: `train.py`
- Student export/eval helper: `student_eval.py`
- Task mapping: `dexscrew/tasks/__init__.py` (`isaacgym_task_map`)
- Core envs: `dexscrew/tasks/xhand_hora.py`, `dexscrew/tasks/xhand_pasini.py`
- Core algos: `dexscrew/algo/ppo/ppo.py`, `dexscrew/algo/ppo/padapt.py`

## 3) Most used run commands
- Teacher train: `scripts/screwdriver_teacher.sh 0 42 <exp_name>`
- Student train: `scripts/screwdriver_student_padapt.sh 0 42 <exp_name>`
- Teacher vis: `scripts/vis_screwdriver_teacher.sh 0 42 <exp_name>`
- Student vis: `scripts/vis_screwdriver_student_padapt.sh 0 42 <exp_name>`
- JIT export: `scripts/convert_student_jit.sh 0 42 <exp_name> screwdriver.pt`

## 4) Runtime behavior facts
- Hydra config root: `configs/config.yaml`
- Headless supported: yes (`headless=True` widely used in training scripts).
- GUI needed only for `vis_*` scripts (`headless=False`).
- Frequent env args: `sim_device`, `rl_device`, `graphics_device_id`, `physics_engine=physx`.
- Env var used by scripts: `CUDA_VISIBLE_DEVICES` (required in practice for GPU selection).
- `PYTHONPATH` / `LD_LIBRARY_PATH` hard requirements in this repo: not found.

## 5) Minimal reproducible env (from repo docs + code)
- Python: `3.8` (explicit in `docs/install.md`)
- Isaac Gym: **Preview 4.0** (explicit in `docs/install.md`)
- PyTorch: install with CUDA (`pytorch-cuda=12.1` shown in docs)
- Required python packages (root): from `requirements.txt`
  - `hydra-core>=1.1`
  - `termcolor`
  - `omegaconf`
  - `gym`
  - `tensorboard`, `tensorboardx`
  - `gdown`
  - `trimesh`
  - `numpy==1.22.4`
  - `wandb`
- Isaac Gym python binding install mode: `pip install -e` under `isaacgym/python` (docs requirement).

## 6) Migration target note (Host Ubuntu 22.04 + Docker Ubuntu 20.04)
- Feasible approach: container uses Ubuntu 20.04 + Python 3.8 + CUDA-compatible torch + Isaac Gym Preview 4.
- Keep training headless first; add GUI/X11 path only if visualization is needed.
- Keep `CUDA_VISIBLE_DEVICES` pass-through in run scripts.

## 7) Suggested minimal porting plan
1. Build base image (Ubuntu 20.04 + Python 3.8 + pip/conda).
2. Install Python deps from `requirements.txt`.
3. Install Isaac Gym Preview 4.0 and run `pip install -e isaacgym/python`.
4. Run smoke test headless:
   - `python train.py task=XHandHoraScrewDriver headless=True train.algo=PPO test=True checkpoint=<ckpt>`
5. Then run script-level test:
   - `scripts/screwdriver_teacher.sh 0 42 smoke`

## 8) Still-missing hard facts (must verify during port)
- Exact NVIDIA driver minimum version.
- Exact torch/torchvision/torchaudio pinned versions compatible with Isaac Gym Preview 4 in this environment.
- Full system package list for rendering/EGL/Vulkan in container when GUI visualization is required.
