"""Frozen-policy exploration diagnostic; never an accepted deployment policy."""
import json,os,runpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from calibrate_m24_e002_a0 import torch  # Shared evaluator bootstrap; IsaacGym first.
from dexscrew.algo.ppo.ppo import PPO

scale=float(os.environ['E003_DIAGNOSTIC_NOISE_SCALE']);seed=int(os.environ['E003_DIAGNOSTIC_NOISE_SEED']);assert 0<=scale<=1
original=PPO._eval_select_action

def select(self,obs):
    action,extra=original(self,obs)
    if not hasattr(self,'_diagnostic_noise_generator'):
        self._diagnostic_noise_generator=torch.Generator(device=action.device).manual_seed(seed)
    noise=torch.randn(action.shape,device=action.device,dtype=action.dtype,generator=self._diagnostic_noise_generator)
    return action+scale*self.model.sigma.exp()*noise,extra

PPO._eval_select_action=select
runpy.run_path(str(Path(__file__).with_name('evaluate_m24_e002_cycle.py')),run_name='__main__')
paths=[Path(x.split('=',1)[1]) for x in sys.argv[1:] if x.startswith('+cycle_eval_out=')];assert len(paths)==1
report=json.loads(paths[0].read_text());report['policy_composition']=dict(mode='diagnostic_frozen_gaussian_exploration',acceptance_eligible=False,noise_scale=scale,noise_seed=seed,independent_common_random_number_stream=True,note='Original frozen mean/std, scaled Gaussian exploration before existing action clamp/mask. Not a trained candidate.')
report['summary']['action_mode']='diagnostic_frozen_gaussian_exploration';paths[0].write_text(json.dumps(report,indent=2)+'\n')
