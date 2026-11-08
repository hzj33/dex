# DexH13 Lightbulb Init-Pose Keyboard Tuner

This note records the local keyboard-control workflow for tuning the
`Dexh13HoraLightbulb` hand root pose and initial joint angles.

## Run Locally

Run this from the repo root on the local machine, not on the cloud machine:

```bash
./docker-run-isaacgym.sh python scripts/tune_dexh13_lightbulb_initpose.py \
  --task Dexh13HoraLightbulb \
  --gpu 0 \
  --seed 42 \
  --out outputs/initpose_tuning/Dexh13HoraLightbulb_current_fixed120.yaml
```

The script prints status to the terminal that launches it. Press `C` in the
Isaac Gym viewer to print the current pose/joint values again.

The script internally forces a clean one-env viewer setup:

- `headless=False`
- `num_envs=1`
- `task.env.numEnvs=1`
- `randomizeMass=False`
- `randomizeCOM=False`
- `randomizeFriction=False`
- `randomizeScale=False`
- `randomizePDGains=False`
- `task.env.object.init_pos_noise=[0.0,0.0,0.0]`
- `task.env.asset.handRootPosNoise=[0.0,0.0,0.0]`
- `task.env.forceScale=0.0`
- `task.env.randomForceProbScalar=0.0`

For the current fixed-scale YAML, the expected startup marker is:

```text
Generated 5000 random initial poses for XHand at scale 1.2
```

## Keyboard Controls

General:

- `M`: switch between hand mode and joint mode.
- `C`: print current pose/joint status to the terminal.
- `O`: save a YAML snippet to the `--out` path and print it.
- `ESC`: save the YAML snippet and quit.
- `-`: reduce the active step size.
- `=`: increase the active step size.

Hand mode:

- `A` / `D`: move hand root `x` negative / positive.
- `S` / `W`: move hand root `y` negative / positive.
- `Q` / `E`: move hand root `z` negative / positive.
- `F` / `R`: roll negative / positive.
- `G` / `T`: pitch negative / positive.
- `H` / `Y`: yaw negative / positive.

Joint mode:

- `Left` / `Right`: select previous / next DOF.
- `[` / `]`: select previous / next DOF.
- `Down` / `Up`: decrease / increase selected DOF.
- `,` / `.`: decrease / increase selected DOF.
- `0` to `9`: select DOF index 0 to 9.
- `Space`: set selected DOF to `0.0`.

Joint values are clamped by the task's configured DOF limits.

## Saved Output

The saved file contains a YAML snippet intended to be pasted under
`env.asset` in the task YAML:

```yaml
handRootPos: [...]
handRootRPY: [...]
handInitPose:
  right_index_joint_0: ...
  ...
```

Default output path used in this repo:

```text
outputs/initpose_tuning/Dexh13HoraLightbulb_current_fixed120.yaml
```

## Notes

- Close other Isaac Gym viewers before running this script, otherwise the
  visible window may not be the tuner window.
- Run it in a local terminal if you want to see the `print()` output directly.
- The tuner edits the live simulated hand root state and hand DOF state every
  frame; it does not train or load any PPO checkpoint.
