#!/usr/bin/env python3
"""Diagnostic-only compliant index PIP/DIP constraint; never a calibrated hand model.

Removes the DIP actuator and adds a reciprocal spring/damper along
q_DIP=f(q_PIP). Does not edit policy targets/history, rewards or stage gates.
"""
import json
import os
from pathlib import Path
from evaluate_m24_e002_cycle import main, isaacgym_task_map, torch, hydra
from isaacgym import gymtorch
from dexscrew.tasks.e003_passive_coupling import instrument_passive_coupling

model = json.loads(Path(os.environ['E003_COUPLING_FIT']).read_text())
coefficients = model['coefficients_descending']
stiffness = float(os.environ.get('E003_COUPLING_K', '0'))
damping = float(os.environ.get('E003_COUPLING_D', '0.1'))
root = Path(os.environ['E003_COUPLING_OUT'])
root.mkdir(parents=True, exist_ok=True)
assert stiffness >= 0 and damping >= 0

def instrument(base):
    return instrument_passive_coupling(base, coefficients, stiffness, damping)

for name in list(isaacgym_task_map):
    if 'M24' in name:
        isaacgym_task_map[name] = instrument(isaacgym_task_map[name])
(root/'coupling_diagnostic.json').write_text(json.dumps(dict(
    diagnostic_only=True, hardware_calibrated=False, stiffness=stiffness,
    damping=damping, fit=model, independent_dip_actuation=(stiffness==0),
    note='Index only; local polynomial extrapolation outside measured range unvalidated; no real hardware execution'),indent=2))
if __name__ == '__main__':
    hydra.main(config_name="config", config_path=str(Path(__file__).resolve().parents[1]/"configs"))(main.__wrapped__)()
    result_path = root/'evaluation.json'
    result = json.loads(result_path.read_text())
    result['passive_coupling_diagnostic'] = json.loads((root/'coupling_diagnostic.json').read_text())
    result['acceptance_eligible'] = False
    result_path.write_text(json.dumps(result, indent=2)+'\n')
