"""Quantify action clipping and exploration reach; never changes a policy."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.evaluation.read_text());digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();assert digest==d['checkpoint_sha256']
    rows=d['first_episode_steps'];active=[0,1,2,3,12,13,14,15]
    raw=np.array([r['policy_action_raw'] for r in rows])[:,active];actual=np.array([r['action'] for r in rows])[:,active];phase=np.array([r['phase_before'] for r in rows]);assert np.array_equal(np.clip(raw,-1,1),actual)
    sigma=torch.load(a.checkpoint,map_location='cpu')['model']['sigma'].exp()[active].double();mu=torch.tensor(raw,dtype=torch.float64)
    normal=torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(mu))
    report={}
    for label,mask in [('all',np.ones(len(rows),dtype=bool))]+[(str(i),phase==i) for i in range(4)]:
        if not mask.any():continue
        entry=dict(samples=int(mask.sum()),raw_abs_max=float(np.abs(raw[mask]).max()),raw_abs_p95=float(np.quantile(np.abs(raw[mask]),.95)),clipped_fraction=float((np.abs(raw[mask])>=1).mean()),bound_loss_1_1_mean=float((np.maximum(np.abs(raw[mask])-1.1,0)**2).sum(1).mean()))
        for scale in [1.,.25]:
            interior=normal.cdf((1-mu)/(sigma*scale))-normal.cdf((-1-mu)/(sigma*scale))
            saturated=(np.abs(raw[mask])>=1)
            entry['scale_'+str(scale)+'_saturated_mean_interior_probability']=float(interior[mask][saturated].mean()) if saturated.any() else None
        report[label]=entry
    out=dict(checkpoint_sha256=digest,clamped_action_parity=True,active_actions=active,report=report,note='Gaussian interior probability is a diagnostic at frozen recorded states; it does not prove a causal learning failure or justify changing physical force limits.')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
