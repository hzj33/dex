We provide scripts for baselines with different sensing capabilities and privileged info configurations.

## Current student baseline positioning (important)
In this repo, the default student baseline is **current ProprioAdapt-style student**, not pure BC.

Methodologically, this current student is a combined imitation-distillation pipeline:
- action behavior cloning (student action supervised by teacher action)
- latent distillation (student latent supervised by teacher latent)
- adapter-only adaptation (main policy backbone frozen; adaptation module trained)
- privileged-teacher to proprio-history-student transfer

So later diffusion comparison is **not** “replace a weak BC baseline”.
The key question is whether diffusion can add value beyond this already-strong current student:
- multimodal action modeling
- temporal consistency in contact-rich phases
- recovery behavior generation
- lower failure reset rate / better robustness under perturbation

### Teacher policy: toggle privileged info
You can enable/disable different inputs to the Teacher for ablations, similar to the penspin repo.

- Without tactile information:
```bash
scripts/screwdriver_teacher.sh train.ppo.enable_tactile=False
```

- Without point cloud information:
```bash
scripts/screwdriver_teacher.sh task.env.hora.point_cloud_sampled_dim=0 train.ppo.use_point_cloud_info=False
```

- Without privileged information to the policy (no priv vector fed to the actor/critic):
```bash
scripts/screwdriver_teacher.sh train.ppo.priv_info=False
```

- With few point cloud points (example: 100 points):
```bash
scripts/screwdriver_teacher.sh task.env.hora.point_cloud_sampled_dim=100 train.ppo.use_point_cloud_info=True
```

- With all signals on (privileged info + point cloud):
```bash
scripts/screwdriver_teacher.sh train.ppo.priv_info=True train.ppo.use_point_cloud_info=True
```

### Important: Student-Teacher consistency (dimension matching)
- The Student policy (padapt) and the Teacher policy must use the SAME privileged info and input settings, otherwise model input dimensions will not match.
- Concretely, keep the following flags consistent between Teacher training and Student training:
  - `train.ppo.priv_info`
  - `train.ppo.use_point_cloud_info`
  - `task.env.hora.point_cloud_sampled_dim` (if using point cloud)

Example pair (Teacher then Student) with matching settings:
```bash
# Teacher
scripts/screwdriver_teacher.sh \
  train.ppo.priv_info=True \
  train.ppo.use_point_cloud_info=True \
  task.env.hora.point_cloud_sampled_dim=100

# Student (use the checkpoint produced by the Teacher above)
scripts/screwdriver_student_padapt.sh \
  train.ppo.priv_info=True \
  train.ppo.use_point_cloud_info=True \
  task.env.hora.point_cloud_sampled_dim=100
```

### 15-minute acceptance for current student (docker)
Use this canonical script to run a fixed-time acceptance window for the current student baseline:
```bash
scripts/screwdriver_student_padapt_15min_docker.sh 0 42 run_a 900
```
Arguments:
- `0`: GPU id
- `42`: seed
- `run_a`: teacher cache name under `outputs/XHandHoraScrewDriver_teacher/`
- `900`: acceptance window in seconds (15 min)

Output:
- `outputs/XHandHoraScrewDriver_student_padapt/<student_cache>/stage2_nn/model_best.ckpt`
- `outputs/XHandHoraScrewDriver_student_padapt/<student_cache>/train_<window>s.log`

### Pure BC baseline (separate algorithm file, no padapt overwrite)
Pure BC is implemented in its own trainer file and launch script:
- trainer: `dexscrew/algo/ppo/pure_bc.py` (`PureBC`)
- train script: `scripts/screwdriver_student_purebc.sh`

Run training:
```bash
scripts/screwdriver_student_purebc.sh 0 42 run_a
```

Run 15-minute acceptance (docker):
```bash
scripts/screwdriver_student_purebc_15min_docker.sh 0 42 run_a 900
```

Pure BC outputs are isolated under:
- `outputs/XHandHoraScrewDriver_student_purebc/<cache>/stage2_bc_nn/`
- `outputs/XHandHoraScrewDriver_student_purebc/<cache>/stage2_bc_tb/`
