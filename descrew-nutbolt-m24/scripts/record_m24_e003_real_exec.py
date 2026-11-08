#!/usr/bin/env python3
"""Record the deployable student in the matched execution diagnostic model."""
import hashlib
import json
import os
from pathlib import Path

import evaluate_m24_e003_real_exec as execution
from record_m24_e002_policy import main


if __name__ == '__main__':
    execution.hydra.main(config_name='config', config_path=str(
        Path(__file__).resolve().parents[1] / 'configs'))(main.__wrapped__)()
    path = Path(os.environ['E003_REAL_EXEC_TRACE'])
    data = json.loads(path.read_text())
    data['checkpoint_sha256'] = hashlib.sha256(
        Path(data['checkpoint']).read_bytes()).hexdigest()
    data['diagnostic_only'] = True
    data['acceptance_eligible'] = False
    data['real_execution_model'] = dict(
        diagnostic_only=True, acceptance_eligible=False,
        observation_target_source='unfiltered_sent_command',
        real_params=execution.REAL_PARAMS)
    data['observation_target_source'] = 'unfiltered_sent_command'
    if os.environ.get('E003_COUPLING_FIT'):
        data['passive_coupling_diagnostic'] = json.loads((
            Path(os.environ['E003_COUPLING_OUT']) / 'coupling_diagnostic.json').read_text())
    path.write_text(json.dumps(data, indent=2) + '\n')
