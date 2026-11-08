#!/usr/bin/env python3
"""Matched contact evaluation with continuous nut rotation; opt-in entry."""
import json
import os
from pathlib import Path
import evaluate_m24_e003_real_exec as execution
from dexscrew.tasks.e003_continuous_nut import instrument_continuous_nut

for name in list(execution.isaacgym_task_map):
    if 'M24' in name:
        execution.isaacgym_task_map[name] = instrument_continuous_nut(
            execution.isaacgym_task_map[name])
        if os.environ.get('E003_RETURN_ENTRY_CLOCK') == '1':
            from dexscrew.tasks.e003_return_progress import instrument_return_entry_clock
            execution.isaacgym_task_map[name] = instrument_return_entry_clock(
                execution.isaacgym_task_map[name])

if __name__ == '__main__':
    try:
        execution.hydra.main(config_name='config', config_path=str(
            Path(__file__).resolve().parents[1]/'configs'))(execution.main.__wrapped__)()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    path = Path(os.environ['E003_REAL_EXEC_OUT'])
    report = json.loads(path.read_text())
    report['real_execution_model'] = dict(
        real_params=execution.REAL_PARAMS, diagnostic_only=True,
        observation_target_source='unfiltered_sent_command')
    report['continuous_nut_diagnostic'] = dict(
        original_asset_modified=False, physics_joint='continuous',
        angle_accounting='4pi boundary offset; no physics state writes',
        jump_gate_unchanged=True, base_fixed=True)
    if os.environ.get('E003_COUPLING_FIT'):
        report['passive_coupling_diagnostic'] = json.loads((
            Path(os.environ['E003_COUPLING_OUT'])/'coupling_diagnostic.json').read_text())
    path.write_text(json.dumps(report, indent=2)+'\n')
