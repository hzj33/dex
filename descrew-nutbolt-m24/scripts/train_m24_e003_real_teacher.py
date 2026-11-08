#!/usr/bin/env python3
"""Bounded PPO adaptation on identified execution + passive index coupling.

Original teacher checkpoint, objective, PPO implementation and task gates are
preserved. New weights are isolated candidates, not automatic replacements.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import train_m24_e002_bounded as bounded
from dexscrew.tasks.e003_actuator_randomization import instrument_actuator_randomization
from dexscrew.tasks.e003_passive_coupling import instrument_passive_coupling
from omegaconf import open_dict

entry = bounded.entry


@entry.hydra.main(config_name='config', config_path=str(Path(__file__).resolve().parents[1]/'configs'))
def main(cfg):
    spec = json.loads(Path(os.environ['E003_TRAIN_EXECUTION_SPEC']).read_text())
    fit = json.loads(Path(spec['coupling_fit']).read_text())
    calibration = json.loads(Path(str(cfg.cycle_calibration)).read_text())
    assert calibration['status'] == 'passed' and calibration['pose_index'] == 743
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos = calibration['custom_init_dof_pos']
    cls = entry.isaacgym_task_map[cfg.task_name]
    ar = dict(identity_fraction=0, per_joint=True, real_anchor_fraction=1,
              real_params=spec['real_params'], control_dt=.05)
    cls = instrument_actuator_randomization(cls, ar)
    cls = instrument_passive_coupling(cls, fit['coefficients_descending'], spec['coupling_k'], spec['coupling_d'])
    entry.isaacgym_task_map[cfg.task_name] = cls
    output = Path('outputs')/str(cfg.train.ppo.output_name)
    output.mkdir(parents=True, exist_ok=True)
    (output/'execution_spec.json').write_text(json.dumps(dict(spec=spec,
        initial_checkpoint=str(cfg.checkpoint), initial_sha256=hashlib.sha256(Path(str(cfg.checkpoint)).read_bytes()).hexdigest(),
        reward_changed=False, physical_gates_changed=False, algorithm_changed=False), indent=2)+'\n')
    signal.signal(signal.SIGTERM, bounded.request_stop)
    entry.PPO.train_epoch = bounded.completed_epoch
    return entry.main.__wrapped__(cfg)


if __name__ == '__main__':
    main()
