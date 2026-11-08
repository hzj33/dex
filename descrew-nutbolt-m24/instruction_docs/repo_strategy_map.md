# repo_strategy_map

## 1. Project mission in one paragraph
This repository is organized around a sim-to-real dexterous manipulation program for rotating and driving screw-like objects with a robotic hand under imperfect simulation. The README frames the full method as four stages: learn an oracle policy with privileged information and point clouds in simulation, distill that oracle into a sensorimotor student using proprioceptive adaptation, deploy the rotational policy on hardware, then fuse additional real-world signals through a later behavior-cloning stage that lives in another repository. In practice, this codebase mainly implements the first two stages, plus a narrow export/deployment bridge. From an experiment perspective, the repo is about learning a strong simulation teacher, then compressing its privileged knowledge into a deployable student that can act from less privileged observations.

## 2. High-level training architecture
The end-to-end experimental flow is best understood as a staged pipeline rather than a single trainer:

1. Environment and task instantiation
- `train.py` is the central entrypoint.
- Hydra root config in `configs/config.yaml` selects a task config from `configs/task/` and a matched training config from `configs/train/`.
- `dexscrew/tasks/__init__.py` maps task names into one of two active environment classes: `dexscrew/tasks/xhand_hora.py` or `dexscrew/tasks/xhand_pasini.py`.

2. Teacher training
- Teacher experiments run with `train.algo=PPO` through `dexscrew/algo/ppo/ppo.py`.
- The policy consumes the normal observation plus a latent derived from privileged information, and optionally point cloud information, through `dexscrew/algo/models/models.py`.
- Rollouts, PPO updates, checkpointing, and TensorBoard logging are all handled inside the PPO trainer.

3. Student distillation / adaptation
- Student experiments run with `train.algo=ProprioAdapt` through `dexscrew/algo/ppo/padapt.py`.
- Stage 2 loads a trained teacher checkpoint, freezes most of the teacher policy, and trains an adaptation module that predicts the teacher's latent from proprioceptive history.
- The student is supervised online in the environment through latent matching and action-level behavior cloning against the teacher.

4. Evaluation and export
- Teacher and student visualization generally reuse `train.py` with `test=True` via shell scripts in `scripts/`.
- `student_eval.py` is a special-purpose student evaluation/export path that performs a short rollout and exports a TorchScript policy for deployment.

5. Downstream deployment and real-world continuation
- `xhand-deploy/` appears to be the handoff path for deploying an exported student rotational policy.
- The README says the later real-world fine-tuning / multimodal fusion stage is handled in an external repository, so this repo does not contain the full final policy-learning stack.

The clearest currently connected experimental path is:
`XHandHoraScrewDriver` task -> teacher PPO training -> best stage1 checkpoint -> `ProprioAdapt` student distillation -> stage2 checkpoint -> TorchScript export.

## 3. Teacher path
The teacher is an oracle policy trained with PPO under privileged simulation access.

What the teacher is:
- A policy/value network built in `dexscrew/algo/models/models.py`.
- Its actor input is the regular observation `obs` plus a latent embedding produced from `priv_info`, and optionally fused with a point-cloud embedding.
- This makes the teacher the privileged policy in the experiment: it has access to information that the final deployable student is not supposed to use at inference.

Where the teacher training logic is organized:
- Entrypoint: `train.py`
- Trainer: `dexscrew/algo/ppo/ppo.py`
- Storage / rollout minibatching: `dexscrew/algo/ppo/experience.py`
- Model definition: `dexscrew/algo/models/models.py`
- Default task/train pairing for the canonical path: `configs/task/XHandHoraScrewDriver.yaml` and `configs/train/XHandHoraScrewDriver.yaml`
- Typical launcher: `scripts/screwdriver_teacher.sh`

What inputs and outputs matter experimentally:
- Inputs:
  - `obs`: the non-privileged policy observation.
  - `priv_info`: the privileged simulator state summary.
  - `point_cloud_info`: sampled object point cloud if enabled.
- Outputs:
  - action mean for the hand policy.
  - value prediction for PPO.
  - privileged latent representation that later becomes the student supervision target.

What likely controls teacher performance most strongly:
- Task config in `configs/task/*.yaml`, especially reward scales, reset thresholds, domain randomization, and action/control parameters.
- Whether the actor has privileged information and point cloud input.
- PPO collection/update settings in `configs/train/*.yaml`, especially `numEnvs`, `horizon_length`, `minibatch_size`, `learning_rate`, `mini_epochs`, and KL threshold.
- The privileged information schema defined in the task class, since it determines what the oracle can exploit.
- The quality and difficulty of the task randomization schedule, since the teacher is being trained to survive imperfect simulation.

Experimentally, the teacher is not just a baseline policy. It is the representation source for the whole stage-2 adaptation pipeline.

## 4. Student path
The student is the deployable sensorimotor policy stage built by distilling the teacher into a less privileged policy.

What the student is:
- A policy that replaces direct access to `priv_info` with a latent predicted from `proprio_hist`.
- Architecturally, it reuses the same actor backbone but swaps the privileged latent source for an adaptation latent predicted by `adapt_tconv` in `dexscrew/algo/models/block.py` and wired in `dexscrew/algo/models/models.py`.

How the current student pipeline works at a high level:
- Launch via `train.py` with `train.algo=ProprioAdapt`.
- Load a teacher checkpoint from stage 1.
- Freeze almost all model parameters.
- Train only the temporal adaptation module inside `dexscrew/algo/ppo/padapt.py`.
- For each online environment step:
  - build student input from `obs`, `proprio_hist`, and optionally point cloud information;
  - build teacher supervision target from `priv_info`;
  - regress student latent toward teacher latent;
  - clone teacher action behavior;
  - step the environment with the student action.

Where PAdapt or related adaptation/distillation logic sits in the overall flow:
- Core trainer: `dexscrew/algo/ppo/padapt.py`
- Shared model logic: `dexscrew/algo/models/models.py`
- Main shell path: `scripts/screwdriver_student_padapt.sh`
- Student visualization: `scripts/vis_screwdriver_student_padapt.sh`
- Student export: `scripts/convert_student_jit.sh` and `student_eval.py`

What assumptions the current student stage appears to make:
- Stage 2 assumes a trained teacher checkpoint exists and is structurally compatible.
- Stage 2 assumes `priv_info` is still available during training as a supervision target, even though it disappears from the final student inference path.
- Teacher and student must match key input settings such as privileged-info usage and point-cloud usage, as documented in `docs/baseline.md`.
- The current implementation appears primarily aligned with the Hora pipeline. A notable warning sign is that the adaptation module input is hardcoded around the Hora-sized proprio stream, while the Pasini task uses a different action/proprio dimensionality.
- The student stage is online and environment-coupled, not a separate offline dataset distillation pipeline.

Practically, the student stage is better understood as privileged-latent distillation plus behavior cloning, not as a second RL phase.

## 5. Environment and task logic
Only two environment classes appear active, but they are parameterized into multiple tasks:
- `dexscrew/tasks/xhand_hora.py`
- `dexscrew/tasks/xhand_pasini.py`

Task definition:
- Task registration happens in `dexscrew/tasks/__init__.py`.
- The repository exposes four named tasks through Hydra:
  - `XHandHoraScrewDriver`
  - `XHandHoraNutBolt`
  - `XHandPasiniScrewDriver`
  - `XHandPasiniBulb`
- Experiment semantics are driven mostly by YAML config rather than separate task classes.

Observation groups:
- `obs`: the main policy observation, implemented as a lagged window over joint positions and target commands. It is the non-privileged stream used by both teacher and student.
- `proprio_hist`: a longer temporal history buffer used by the adaptation module in stage 2.
- `priv_info`: the privileged vector assembled from object state, fingertip state, hand state options, contact-related fields, and other simulator-only quantities depending on config toggles.
- `point_cloud_info`: transformed object point cloud samples when enabled.
- `rot_axis_buf`: present in the observation dict, but it does not appear central to the current policy path.

Action interface:
- Hora uses 12 policy actions; Pasini uses 16.
- Actions are not direct torques from the policy. They are delta targets integrated into controller targets, then executed through low-level torque control by default.
- The task masks some action dimensions depending on setup and pads an extra zero dimension for the object joint.
- This means the learned policy is selecting hand target adjustments, while low-level PD control handles actuation.

Reward structure:
- The main task incentive is rotational progress at the screw/nut joint.
- The reward is shaped by several additive components, including:
  - rotation reward,
  - pose-difference penalty,
  - torque penalty,
  - work penalty,
  - point-cloud vertical spread penalty,
  - overspeed rotation penalty,
  - proximity reward based on fingertip distance to the manipulated object.
- Reward scales are configurable and can follow simple curricula through task config.
- Pasini adds some extra shaping and debug-oriented hooks, but the main reward logic remains close to Hora.

Reset / termination logic:
- Reset conditions include maximum episode length, fingertip-object distance failure, stagnation of the nut/screw joint, loss of contact, and approaching the screw joint upper limit.
- Resets also refresh history buffers and several per-episode randomization quantities.
- This makes termination logic an important part of the experiment definition, because "success" is partly encoded as sustained contact and forward rotational progress without triggering those failure modes.

Domain randomization and privileged information:
- Task configs expose substantial randomization around mass, COM, friction, scale, PD gains, observation noise, action noise, pose noise, and object tilt.
- Some randomization appears to happen at environment creation, while other perturbations are refreshed per reset or per step.
- The privileged vector is highly configurable and is the key difference between what the teacher sees and what the student must infer.

Core experimental path vs auxiliary engineering code:
- Core: `train.py`, `dexscrew/tasks/xhand_hora.py`, `dexscrew/tasks/xhand_pasini.py`, `dexscrew/algo/ppo/ppo.py`, `dexscrew/algo/ppo/padapt.py`, `dexscrew/algo/models/models.py`, and matching configs/scripts.
- Auxiliary: camera/GIF plumbing, deployment packaging in `xhand-deploy/`, debug helpers, pose dumping, and convenience shell wrappers.

## 6. Config and experiment control surface
A researcher would mostly control experiments through Hydra config plus shell-script overrides.

Main config surfaces:
- Root selection: `configs/config.yaml`
- Task semantics: `configs/task/*.yaml`
- PPO/model hyperparameters: `configs/train/*.yaml`
- Launch wrappers: `scripts/*.sh`

What researchers are most likely to change:
- Task choice: `task=XHandHoraScrewDriver`, `task=XHandPasiniScrewDriver`, and so on.
- Algorithm stage: `train.algo=PPO` vs `train.algo=ProprioAdapt`.
- Teacher/student compatibility flags:
  - `train.ppo.priv_info`
  - `train.ppo.use_point_cloud_info`
  - `task.env.hora.point_cloud_sampled_dim`
- Randomization, reward scales, reset thresholds, object settings, and controller settings in `configs/task/*.yaml`.
- PPO batch and optimization settings in `configs/train/*.yaml`.
- Output naming, checkpoint loading, headless mode, and visualization overrides through scripts.

How teacher vs student experiments are separated:
- Teacher path:
  - `train.algo=PPO`
  - checkpoints under `stage1_nn/`
  - logs under `stage1_tb/`
  - canonical script `scripts/screwdriver_teacher.sh`
- Student path:
  - `train.algo=ProprioAdapt`
  - `train.ppo.proprio_adapt=True`
  - teacher checkpoint passed in through `checkpoint` or `train.load_path`
  - checkpoints under `stage2_nn/`
  - logs under `stage2_tb/`
  - canonical script `scripts/screwdriver_student_padapt.sh`

Currently active vs alternative paths:
- Most complete, end-to-end path: Hora screwdriver teacher -> Hora student PAdapt -> student export.
- Alternative but less clearly completed path: Pasini screwdriver teacher. The repository contains a teacher launcher for it, but no equally explicit student/distillation launcher, and the current adaptation architecture appears more naturally aligned with Hora dimensions.
- Legacy/duplicate/non-core path to keep separate: `dexscrew/tasks/your_hand_hora.py` looks like scaffold or abandoned template code rather than an active experiment path.

## 7. Evaluation and metrics
How success is measured:
- There is no single, explicit benchmark harness in the repo. Success is inferred through reward curves, best-checkpoint selection, rollout visualization, and task-specific logged statistics.
- The most relevant signals for rotation experiments are:
  - episode reward,
  - episode length,
  - rotation reward,
  - screw angular velocity,
  - screw angular position,
  - positive velocity ratio,
  - whether the policy avoids failure resets and maintains contact.
- For the student stage, additional optimization signals are:
  - latent loss,
  - behavior cloning loss,
  - total student loss.

Where metrics / logging / checkpoints are handled:
- PPO logging and checkpointing: `dexscrew/algo/ppo/ppo.py`
- Student logging and checkpointing: `dexscrew/algo/ppo/padapt.py`
- Output root: `outputs/<train.ppo.output_name>/`
- Config snapshot and git diff are written by `train.py`
- Visualization is usually done through `scripts/vis_*.sh`
- Student export/evaluation is handled through `student_eval.py`

What signals are likely useful for comparing methods:
- Stage 1 teacher quality:
  - best reward checkpoint,
  - rotational progress metrics,
  - stability across randomized conditions.
- Stage 2 student quality:
  - how closely it matches teacher behavior without privileged inputs,
  - whether latent loss decreases without harming task reward,
  - whether student checkpoints preserve rotational progress and contact stability,
  - whether exported policies remain usable in deployment.
- For research comparison, the most meaningful metric bundle is likely not just total reward, but reward plus screw-specific progress signals plus robustness under noisy/randomized settings.

## 8. Research leverage points
This is the most promising part of the repo for diffusion-related exploration in the student stage.

### Leverage point 1: Replace deterministic adaptation latent prediction with a conditional diffusion latent model
- Part of pipeline it touches:
  - `dexscrew/algo/ppo/padapt.py`
  - `dexscrew/algo/models/models.py`
  - the student's `adapt_tconv` pathway
- Why it matters experimentally:
  - The current student reduces privileged-to-nonprivileged transfer to a deterministic latent regression problem from `proprio_hist` to teacher latent. That is a narrow bottleneck. A diffusion-style latent predictor could model multimodal uncertainty in what privileged state is consistent with the same proprio history, especially under imperfect simulation and noisy contact transitions.
- Risk level:
  - Low to medium.
- Why this is relatively tractable:
  - It stays inside stage 2, preserves the teacher policy backbone, and swaps the student-side latent inference mechanism without immediately forcing changes to environment logic.

### Leverage point 2: Replace action-level behavior cloning with a conditional action diffusion policy in stage 2
- Part of pipeline it touches:
  - `dexscrew/algo/ppo/padapt.py`
  - `dexscrew/algo/models/models.py`
  - export path in `student_eval.py`
- Why it matters experimentally:
  - The current student uses pointwise behavior cloning against teacher actions. That assumes a single correct action for each observation-history pair. Dexterous contact manipulation is often multimodal and path-dependent, so a diffusion policy could represent richer action uncertainty and improve robustness when teacher behavior is not uniquely determined.
- Risk level:
  - Medium to high.
- Why the risk is higher:
  - It changes the deployed policy interface more directly and may complicate real-time inference and TorchScript export.

### Leverage point 3: Move from single-step imitation to short-horizon trajectory diffusion over action chunks
- Part of pipeline it touches:
  - student training loop in `dexscrew/algo/ppo/padapt.py`
  - possibly rollout collection assumptions in `train.py` and `student_eval.py`
- Why it matters experimentally:
  - The environment already encodes temporal structure through `proprio_hist`, control-frequency substeps, and contact-sensitive reward/termination logic. A chunked diffusion model over short action sequences could better capture stable rotational strategies than stepwise cloning.
- Risk level:
  - Medium.
- Why it is attractive:
  - It directly matches the sequential nature of the control problem without rewriting the teacher stage.

### Leverage point 4: Use teacher latent trajectories as a diffusion-supervised intermediate target instead of only current-step latent matching
- Part of pipeline it touches:
  - `dexscrew/algo/ppo/padapt.py`
  - student representation learning around `proprio_hist`
- Why it matters experimentally:
  - The current stage-2 latent loss treats privileged supervision as a one-step regression target. Diffusion over latent trajectories could force the student to model how privileged state evolves across contact-rich manipulation, which may be more valuable than matching only the current latent.
- Risk level:
  - Medium.
- Why it matters for research:
  - It is a cleaner intervention if the goal is to study representation transfer rather than wholesale policy replacement.

### Leverage point 5: Introduce an offline or hybrid teacher-generated dataset interface for diffusion experiments
- Part of pipeline it touches:
  - mostly `dexscrew/algo/ppo/padapt.py` and any future dataset plumbing
- Why it matters experimentally:
  - Current student training is online and tightly coupled to environment stepping. Diffusion experiments often benefit from stable replay-style datasets of trajectories, histories, latents, and actions. Creating a teacher-generated dataset layer would open up broader student-stage method comparisons without changing teacher RL.
- Risk level:
  - Medium to high.
- Why the risk exists:
  - It changes the training workflow and evaluation assumptions, not just the student model.

Overall recommendation for diffusion research:
- Lowest-friction entry: diffusion over the stage-2 latent inference module.
- Highest-upside but harder change: diffusion over student action sequences.
- Most reusable research infrastructure improvement: build a teacher-trajectory dataset layer for student-stage experiments.

## 9. Current project abstraction
If a researcher had to explain this repo in a small set of bullets, the mental model would be:

- This is a staged dexterous manipulation repo, not a single end-to-end learner.
- The core canonical task is screwdriver rotation with a simulated hand under domain randomization.
- Stage 1 trains a privileged teacher with PPO.
- The teacher sees simulator-side privileged state and optionally point-cloud information.
- Stage 2 trains a student to infer the teacher's latent from proprioceptive history.
- The student is trained by latent imitation plus action behavior cloning, not by a second PPO phase.
- Environment semantics live mostly in two task classes plus Hydra YAML, not in many separate task-specific trainers.
- Reward and reset design are central to what the experiment actually learns.
- Hora is the most complete teacher-to-student path in the current repo.
- Pasini looks like an important parallel development branch, but its student-stage support is less clearly finished.
- The repo also contains export and deployment scaffolding, but the later real-world multimodal fine-tuning stage is outside this codebase.
- The best places for new student-stage research are around the adaptation bottleneck and the teacher-to-student supervision interface.

## 10. Open questions / uncertainties
- Is the true intended active path still `XHandHoraScrewDriver`, or is the repo mid-transition toward `XHandPasiniScrewDriver` as the new main branch?
- Is Pasini supposed to have a stage-2 student path, or is it currently teacher-only?
- If Pasini is supposed to support stage 2, should the adaptation module dimensionality be generalized beyond the Hora setting?
- Are evaluation statistics accumulated in the task classes consumed by tooling outside this repo?
- Is there a canonical success-rate or robustness metric used in the associated paper workflow, beyond reward and logged screw-specific signals?
- Are some randomization knobs intentionally dormant for future work, or are they partially wired remnants from parent repos?
- The environment code suggests a few implementation ambiguities around privileged-buffer layout and some alternative branches. Which of those are known technical debt versus intentional design?
- Is `XHandHoraNutBolt` still a first-class experiment path, or is it mainly an inherited/demo variant?
- For future planning, should deployment constraints such as TorchScript export latency be treated as hard requirements for any new student architecture?

## 11. Suggested handoff note for GPT
### For GPT planning
Remember that this repo is primarily a two-stage sim pipeline wrapped inside a larger four-stage research story. The strongest connected path is `XHandHoraScrewDriver`: train a privileged PPO teacher, then distill it with `ProprioAdapt` into a student that predicts the teacher latent from proprio history, then export that student for deployment. The task classes and Hydra YAML define most of the real experiment semantics, especially reward shaping, reset logic, randomization, and privileged-information design. Treat `Pasini` as an important parallel branch, but do not assume its student-stage pipeline is complete. If you generate `AGENTS.md` or `PLANS.md`, prioritize the stage-2 student bottleneck, teacher-student compatibility constraints, and the separation between core experimental path and auxiliary deployment/debug code.
