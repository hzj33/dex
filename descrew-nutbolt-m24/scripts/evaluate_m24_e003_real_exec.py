#!/usr/bin/env python3
"""Cycle evaluation with the identified real-hand execution model, contact
physics intact. All environments run the per-joint actuator parameters
identified from the real deployment logs (proximal first-order ~0.22/step
with one-step delay, distal near-instant, per-joint velocity limits, PD
multiplier 4 so the composite response matches the smooth trapezoid). This is
the strongest pre-hardware proxy for the real deployment loop; it is a
diagnostic and never replaces the original-execution acceptance gates.
"""
import json
import os
from pathlib import Path
from evaluate_m24_e002_cycle import main, isaacgym_task_map, torch, hydra
from dexscrew.tasks.e003_actuator_randomization import instrument_actuator_randomization

REAL_PARAMS = json.loads(os.environ.get(
    'E003_REAL_EXEC_PARAMS',
    '{"i0":{"gain":0.228,"delay":true,"vmax":0.65,"kp_mult":4.0},'
    '"i1":{"gain":0.22,"delay":true,"vmax":0.5,"kp_mult":4.0},'
    '"i2":{"gain":0.939,"delay":true,"vmax":1.2,"kp_mult":4.0},'
    '"i3":{"gain":0.939,"delay":true,"vmax":1.2,"kp_mult":4.0},'
    '"t0":{"gain":0.215,"delay":true,"kp_mult":4.0},'
    '"t1":{"gain":0.222,"delay":true,"kp_mult":4.0},'
    '"t2":{"gain":0.52,"delay":false,"vmax":0.9,"kp_mult":4.0},'
    '"t3":{"gain":0.812,"delay":true,"vmax":1.08,"kp_mult":4.0}}'))

spec = dict(identity_fraction=0.0, per_joint=True, real_anchor_fraction=1.0,
            real_params=REAL_PARAMS, control_dt=0.05)
for name in list(isaacgym_task_map):
    if 'M24' in name:
        isaacgym_task_map[name] = instrument_actuator_randomization(
            isaacgym_task_map[name], spec)

if os.environ.get('E003_COUPLING_FIT'):
    # Optional shared physical coupling for matched teacher/student validation.
    import evaluate_m24_e003_passive_diagnostic

if __name__ == '__main__':
    # The common evaluator exits explicitly. Preserve its status but still
    # annotate the report; the old post-main code was never reached.
    exit_status = 0
    try:
        hydra.main(config_name="config", config_path=str(Path(__file__).resolve().parents[1]/"configs"))(main.__wrapped__)()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
        exit_status = exc.code or 0
    result_path = Path(os.environ['E003_REAL_EXEC_OUT'])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(result_path.read_text())
    result['real_execution_model'] = dict(
        diagnostic_only=True, acceptance_eligible=False, per_joint=True,
        observation_target_source='unfiltered_sent_command',
        real_params=REAL_PARAMS,
        source='identified 2026-09-17 from six real deployment logs, R2>=0.92')
    if os.environ.get('E003_COUPLING_FIT'):
        result['passive_coupling_diagnostic'] = json.loads((
            Path(os.environ['E003_COUPLING_OUT'])/'coupling_diagnostic.json').read_text())
    result_path.write_text(json.dumps(result, indent=2)+'\n')
    raise SystemExit(exit_status)
