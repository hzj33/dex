#!/usr/bin/env python3
"""Native 20 Hz diagnostic video with explicit simulator-assisted triggers."""
import hashlib
import json
import os
from pathlib import Path

import evaluate_m24_e003_phase_intervention as intervention
from record_m24_e002_policy import main

if __name__ == '__main__':
    intervention.execution.hydra.main(config_name='config', config_path=str(
        Path(__file__).resolve().parents[1]/'configs'))(main.__wrapped__)()
    path = Path(os.environ['E003_REAL_EXEC_TRACE'])
    data = json.loads(path.read_text())
    data['checkpoint_sha256'] = hashlib.sha256(Path(data['checkpoint']).read_bytes()).hexdigest()
    data['diagnostic_only'] = True
    data['acceptance_eligible'] = False
    data['phase_command_source'] = 'simulator-assisted release trigger; not autonomous'
    data['phase_intervention'] = dict(mode=intervention.mode,
        pulse_steps=intervention.release_pulse_steps, joint_feedback_modified=False,
        trigger_source='simulator drive-completion event stands in for human key press')
    data['real_execution_model'] = intervention.execution.REAL_PARAMS
    data['phase_intervention_steps'] = intervention.intervention_rows
    path.write_text(json.dumps(data, indent=2)+'\n')
