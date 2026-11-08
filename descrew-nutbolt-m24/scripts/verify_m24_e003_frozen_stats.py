"""Bind frozen-input-statistics verification to two concrete PPO checkpoints."""
import hashlib,json,sys
from pathlib import Path
import torch

base_path,candidate_path=map(Path,sys.argv[1:3]);base=torch.load(base_path,map_location='cpu');candidate=torch.load(candidate_path,map_location='cpu')
checks={n:all(torch.equal(base[n][k],candidate[n][k]) for k in base[n]) for n in ['running_mean_std','priv_mean_std','point_cloud_mean_std']}
checks['actor_mean_changed']=any(not torch.equal(base['model'][k],candidate['model'][k]) for k in ['mu.weight','mu.bias'])
checks['value_stats_changed']=any(not torch.equal(base['value_mean_std'][k],candidate['value_mean_std'][k]) for k in base['value_mean_std'])
result=dict(status='passed' if all(checks.values()) else 'failed',base_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),checks=checks)
print(json.dumps(result,indent=2));assert all(checks.values())
