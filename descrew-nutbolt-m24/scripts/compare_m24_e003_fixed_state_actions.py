"""Compare frozen-normalization PPO candidates on identical recorded policy inputs."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from dexscrew.algo.models.models import ActorCritic


def main():
    p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);p.add_argument('checkpoints',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    d=json.loads(a.evaluation.read_text());cfg=yaml.safe_load(a.evaluation.with_suffix('.config.yaml').read_text());z=np.load(a.evaluation.with_suffix('.policy_inputs.npz'));net=cfg['train']['network'];ppo=cfg['train']['ppo']
    model=ActorCritic(dict(actor_units=net['mlp']['units'],priv_mlp_units=net['priv_mlp']['units'],actions_num=16,input_shape=(z['obs'].shape[-1],),priv_info=ppo['priv_info'],proprio_adapt=False,priv_info_dim=z['priv_info'].shape[-1],point_mlp_units=net['point_mlp']['units'],use_point_cloud_info=ppo['use_point_cloud_info'],proprio_len=ppo['proprio_len'])).eval()
    inputs={k:torch.from_numpy(z[k]) for k in ['obs','priv_info','point_cloud_info','proprio_hist']};active=[0,1,2,3,12,13,14,15];phase=z['phase'];ref=None;out=[]
    for path in a.checkpoints:
        c=torch.load(path,map_location='cpu');sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if ref is None:
            assert sha==d['checkpoint_sha256'],'First checkpoint must be the captured reference'
            ref=c
        else:
            for key in ['running_mean_std','priv_mean_std','point_cloud_mean_std']:
                assert all(torch.equal(ref[key][k],c[key][k]) for k in ref[key]),'Recorded normalized inputs require identical input statistics'
        model.load_state_dict(c['model'],strict=True)
        with torch.no_grad():mu=model.act_inference(inputs)[0].numpy()
        if sha==d['checkpoint_sha256']:np.testing.assert_allclose(mu,z['raw_mean'],rtol=1e-5,atol=1e-5)
        rows={}
        for i in range(4):
            m=phase==i;v=mu[m][:,active];original=z['raw_mean'][m][:,active]
            rows[str(i)]=dict(samples=int(m.sum()),bound_loss=float((np.maximum(np.abs(v)-1.1,0)**2).sum(-1).mean()),raw_abs_max=float(np.abs(v).max()),clipped_fraction=float((np.abs(v)>=1).mean()),clipped_action_rms_change=float(np.sqrt(((np.clip(v,-1,1)-np.clip(original,-1,1))**2).mean())))
        out.append(dict(checkpoint=str(path),sha256=sha,phases=rows))
    a.output.write_text(json.dumps(dict(reference_replay_pass=True,inputs_sha256=hashlib.sha256(a.evaluation.with_suffix('.policy_inputs.npz').read_bytes()).hexdigest(),candidates=out,note='Identical recorded states isolate policy-output change, not closed-loop success or force acceptance.'),indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
