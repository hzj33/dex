"""Diagnostic-only return-policy substitution; never a single-policy result."""
from pathlib import Path
import hashlib
import torch
from dexscrew.algo.ppo.ppo import PPO

class ReturnReferenceDiagnostic:
    def __init__(self,candidate,env,output,cfg):
        assert cfg.test and env.e002_cycle is not None
        self.candidate,self.env=candidate,env
        path=str(cfg.cycle_eval_return_reference)
        self.reference=PPO(env,str(Path(output)/'reference_return'),full_config=cfg)
        self.reference.restore_test(path);self.reference.set_eval()
        self.metadata=dict(mode='diagnostic_return_reference',single_policy=False,acceptance_eligible=False,reference_checkpoint=path,reference_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),selection='Original teacher only in measured phase2; candidate otherwise')
    def _eval_select_action(self,obs):
        action,extra=self.candidate._eval_select_action(obs)
        other,other_extra=self.reference._eval_select_action(obs)
        mask=(self.env.e002_cycle.phase==2).to(action.device)[:,None]
        assert extra.keys()==other_extra.keys()
        return torch.where(mask,other,action),{k:torch.where(mask,other_extra[k],v) for k,v in extra.items()}
