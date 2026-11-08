#!/usr/bin/env python3
"""Export a phase-commanded student, verify actions against live native inference.

This artifact accepts raw 101-D obs and raw (30,32) proprio history. The phase
command is explicit and must be supplied by the eventual real-hand adapter.
"""
import hashlib
import json
from pathlib import Path
from calibrate_m24_e002_a0 import hydra, isaacgym_task_map, omegaconf_to_dict, torch
from dexscrew.algo.ppo.padapt import ProprioAdapt
from omegaconf import open_dict
import numpy as np


class RawStudent(torch.nn.Module):
    def __init__(self,agent):
        super().__init__()
        self.obs_norm=agent.running_mean_std
        self.hist_norm=agent.sa_mean_std
        self.adapt=agent.model.adapt_tconv
        self.actor=agent.model.actor_mlp
        self.mu=agent.model.mu
        mask=torch.ones(16,device=agent.device);mask[4:12]=0
        self.register_buffer('action_mask',mask)

    def forward(self,obs,hist):
        x=self.obs_norm(obs)
        latent=torch.tanh(self.adapt(self.hist_norm(hist)))
        return torch.clamp(self.mu(self.actor(torch.cat([x,latent],-1))),-1,1)*self.action_mask


@hydra.main(config_name='config',config_path='../configs')
def main(cfg):
    out=Path(str(cfg.export_out));out.mkdir(parents=True,exist_ok=False)
    cal=json.loads(Path(str(cfg.cycle_calibration)).read_text())
    assert cal['status']=='passed'
    with open_dict(cfg.task.env):cfg.task.env.customInitDofPos=cal['custom_init_dof_pos']
    torch.manual_seed(int(cfg.seed));np.random.seed(int(cfg.seed))
    env=isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),sim_device=cfg.sim_device,graphics_device_id=0,headless=True)
    agent=ProprioAdapt(env,str(out/'native'),full_config=cfg)
    agent.restore_test(str(cfg.checkpoint));agent.set_eval()
    obs=env.reset();wrapper=RawStudent(agent).eval()
    example=(obs['obs'],obs['proprio_hist'])
    with torch.no_grad():
        jit=torch.jit.trace(wrapper,example)
        jit.save(str(out/'student_phase101.pt'))
        loaded=torch.jit.load(str(out/'student_phase101.pt'),map_location=agent.device).eval()
        maximum=0.;phases=set();cycles=0
        for step in range(int(cfg.get('export_steps',440))):
            phases.update(env.e002_cycle.phase.cpu().tolist())
            expected,_=agent._eval_select_action(obs);expected[:,4:12]=0
            actual=loaded(obs['obs'],obs['proprio_hist'])
            maximum=max(maximum,float((expected-actual).abs().max()))
            assert torch.isfinite(actual).all() and maximum<1e-5,maximum
            obs,_,done,_=env.step(actual)
            cycles=max(cycles,int(env.e002_cycle.cycles.max()))
            if done.any():break
        cpu=torch.jit.load(str(out/'student_phase101.pt'),map_location='cpu').eval()
        cpu_error=float((cpu(example[0].cpu(),example[1].cpu())-loaded(*example).cpu()).abs().max())
        assert cpu_error<1e-4,cpu_error
        cpu.save(str(out/'student_phase101.pt'))
    result=dict(status='passed:native_jit_action_parity',checkpoint=str(cfg.checkpoint),
                checkpoint_sha256=hashlib.sha256(Path(str(cfg.checkpoint)).read_bytes()).hexdigest(),
                artifact_sha256=hashlib.sha256((out/'student_phase101.pt').read_bytes()).hexdigest(),
                steps=step+1,phase_coverage=sorted(phases),max_cycles_observed=cycles,
                max_native_jit_action_error=maximum,cpu_gpu_error=cpu_error,
                inputs={'obs':[1,101],'proprio_hist':[1,30,32]},output=[1,16],
                action_mask_indices=list(range(4,12)),normalization_included=True,
                phase_command_required=True,phase_source='external; simulation state machine for parity run',
                phase_generation_implemented_in_real_adapter=False,hardware_ready=False,
                controller_note='Simulation Kp2.4 is a task property, not encoded in the neural policy.')
    (out/'export_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
