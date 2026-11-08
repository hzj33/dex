"""Build isolated policy/normalizer interventions for read-only cloud eval."""
import argparse,hashlib,json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    base=torch.load(a.base,map_location='cpu',weights_only=False);candidate=torch.load(a.candidate,map_location='cpu',weights_only=False)
    names=['running_mean_std','priv_mean_std','point_cloud_mean_std']
    assert base['running_mean_std']['running_mean'].shape==candidate['running_mean_std']['running_mean'].shape==(101,)
    assert base['model'].keys()==candidate['model'].keys()
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(acceptance_eligible=False,note='Diagnostic interventions only; input normalization changes, critic value normalization retained with each model.',base=dict(path=str(a.base),sha256=digest(a.base)),candidate=dict(path=str(a.candidate),sha256=digest(a.candidate)),checkpoints={})
    for label,actor,norm in [('base_actor_candidate_norm',base,candidate),('candidate_actor_base_norm',candidate,base)]:
        mixed=dict(actor)
        for name in names:
            assert actor[name].keys()==norm[name].keys()
            for k in actor[name]:assert actor[name][k].shape==norm[name][k].shape
            mixed[name]=norm[name]
        path=a.output/(label+'.pth');torch.save(mixed,path)
        reread=torch.load(path,map_location='cpu',weights_only=False)
        assert all(torch.equal(reread['model'][k],actor['model'][k]) for k in actor['model'])
        assert all(torch.equal(reread[n][k],norm[n][k]) for n in names for k in norm[n])
        result['checkpoints'][label]=dict(path=str(path),sha256=digest(path))
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
