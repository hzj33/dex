#!/usr/bin/env python3
"""CPU contract checks using the accepted teacher; no optimizer or hardware."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from dexscrew.algo.models.e003_deploy_student import DeployStudent, TEACHER_SHA
from dexscrew.algo.e003_replay import sample_with_expert_anchor
from deploy_m24_e003_student import Policy, PolicyLoop
from m24_e003_deploy_runtime import DeployInference


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--action-scale',type=float,choices=(0.02,0.05),default=0.02)
    args=parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(42)
    replay = {key:torch.arange(64).float().reshape(64,1)+i*100
              for i,key in enumerate(('action','obs','hist','command','latent'))}
    anchor = {key:value+1000 for key,value in replay.items()}
    original_anchor = {key:value.clone() for key,value in anchor.items()}
    state = torch.get_rng_state(); ids = torch.randint(64,(32,)); torch.set_rng_state(state)
    sampled, count = sample_with_expert_anchor(replay,64,anchor,64,32,0)
    assert count == 0 and all(torch.equal(sampled[key],value[ids]) for key,value in replay.items())
    sampled, count = sample_with_expert_anchor(replay,64,anchor,64,32,.25)
    assert count == 8 and (sampled['action'][:8] >= 1000).all() and (sampled['action'][8:] < 64).all()
    for i,key in enumerate(replay):
        assert torch.equal(sampled[key]-i*100,sampled['action'])
        assert torch.equal(anchor[key],original_anchor[key])
    _, count = sample_with_expert_anchor(replay,64,{},0,32,.25)
    assert count == 0
    teacher = Path('outputs/cloud_pipeline_m24_e003_compliant_validate_20260911/selected_teacher/e003_compliant_frozen_teacher.pth')
    assert hashlib.sha256(teacher.read_bytes()).hexdigest() == TEACHER_SHA
    ck = torch.load(teacher, map_location='cpu', weights_only=False)
    cal = json.loads(Path('outputs/cloud_pipeline_m24_e003_student_20260914/a0c/manifests/a0c_diagnostic.json').read_text())
    model = DeployStudent(ck, cal['custom_init_dof_pos'], args.action_scale).eval()
    q = model.init_pose.reshape(1,16)
    hist = model.reset_history(q,q)
    # Backprop reaches the history estimator, never the frozen teacher actor.
    action,logits,clock,latent=model.predict(hist[:,-3:].reshape(1,96),hist)
    loss=action.square().mean()+logits.square().mean()+clock.square().mean()+latent.square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.actor.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())
    # Actual teacher actor weights and frozen normalization are preserved exactly.
    for i in (0,2,4):
        for name in ('weight','bias'):
            assert torch.equal(model.actor[i].state_dict()[name], ck['model'][f'actor_mlp.mlp.{i}.{name}'])
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); artifact=root/'fixture_untrained.pt'
        torch.jit.script(model).save(str(artifact))
        scripted=torch.jit.load(str(artifact)); native=model(q,q,hist)
        for a,b in zip(native, scripted(q,q,hist)): torch.testing.assert_close(a,b,rtol=0,atol=0)
        env=SimpleNamespace(device='cpu',rl_device='cpu',num_actions=16,target_smoothing_alpha=0,
                            config={'env':{'controller':{}}})
        sim=DeployInference(env,artifact,record_predictions=True); real=PolicyLoop(Policy(artifact)); previous=q.clone()
        env.rl_device = 'cuda:0'
        cpu_override = DeployInference(env,artifact,inference_device='cpu')
        assert cpu_override.inference_device == 'cpu'
        env.rl_device = 'cpu'
        with torch.no_grad():
            for step in range(60):
                measured=previous+torch.randn_like(previous)*0.001
                frame=torch.cat((measured,previous),-1).reshape(1,1,32)
                # Poison every unavailable input. Only proprio_hist may be read.
                obs={'proprio_hist':frame,'obs':torch.full((1,101),float('nan')),
                     'priv_info':torch.full((1,126),float('nan')),
                     'point_cloud_info':torch.full((1,100,3),float('nan'))}
                env.prev_targets = previous.clone()
                action,_=sim._eval_select_action(obs)
                assert sim.prediction.shape==(1,5) and torch.isfinite(sim.prediction).all()
                assert torch.equal(sim.contract_errors, torch.zeros(1, 2))
                expected=torch.maximum(torch.minimum(previous+args.action_scale*action,scripted.upper),scripted.lower)
                record=real.step(measured.numpy()[0],20)
                torch.testing.assert_close(torch.from_numpy(record['next_target']).reshape(1,16),expected,rtol=0,atol=0)
                assert (action[:,4:12]==0).all()
                assert float((expected-previous).abs().max()) <= args.action_scale+1e-6
                torch.testing.assert_close(sim.history,real.history,rtol=0,atol=0)
                if step==0:
                    torch.testing.assert_close(sim.history[:,:-1],hist[:,1:],rtol=0,atol=0)
                previous=expected
        # Real legacy entry point, but fake controller: no SDK import/actuation.
        revision=subprocess.check_output(['git','rev-parse','sim2real'],text=True).strip()
        legacy=root/'m24_legacy_deployment.py'
        legacy.write_bytes(subprocess.check_output(['git','show',revision+':scripts/deploy_m24_student.py']))
        subprocess.run([sys.executable,'scripts/deploy_m24_e003_student.py','--legacy-script',str(legacy),
                        '--jit-policy',str(artifact),'--dry-run','--max-policy-steps','3',
                        '--log-dir',str(root/'dryrun')],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        rows=[json.loads(x) for x in (root/'dryrun/policy_steps.jsonl').read_text().splitlines()]
        assert len(rows)==3
    print('PASS: action scale',args.action_scale,'frozen teacher actor, TorchScript, poisoned privileged inputs, original reset-history timing, 60-step sim/deploy parity, legacy fake-controller dry run; fixture untrained')


if __name__=='__main__': main()
