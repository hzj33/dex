"""Predeclared checks of inertial contact reference and raw/8 support estimate."""
import argparse,json
from pathlib import Path
import numpy as np


def errors(e):
    force=np.linalg.norm(e[:,:3],axis=1);tilt=np.linalg.norm(e[:,3:5],axis=1)
    return dict(force_peak_N=float(force.max()),force_rms_N=float(np.sqrt((force**2).mean())),tilt_peak_Nm=float(tilt.max()),tilt_rms_Nm=float(np.sqrt((tilt**2).mean())))

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--sensor-divisor',type=float,choices=[1.,8.],default=8.);a=p.parse_args();state=json.loads((a.root/'status.json').read_text());lim=state['validation_limits'];results={};reference_ok=[];scaling_ok=[];friction_excited=[]
    for label in ['mu0_y','mu1_y','mu0_z','mu1_z']:
        path=a.root/(label+'.json');d=json.loads(path.read_text());r=d['rows'];raw=np.array([x['raw_base_solver'] for x in r]);ref=np.array([x['expected_support_at_base_com'] for x in r]);mask=np.array([x['normal_lambda_sum']>0 for x in r]);heldout=mask&(np.arange(len(r))>=500)
        if not mask.any() or not heldout.any():
            results[label]=dict(status='invalid:contact_lost_before_heldout',contact_samples=int(mask.sum()),heldout_contact_samples=int(heldout.sum()));reference_ok.append(False);scaling_ok.append(False);continue
        full=errors((raw/a.sensor_divisor-ref)[mask]);held=errors((raw/a.sensor_divisor-ref)[heldout]);passed=full['force_peak_N']<=lim['force_peak_error_N'] and full['force_rms_N']<=lim['force_rms_error_N'] and full['tilt_peak_Nm']<=lim['tilt_moment_peak_error_Nm'] and full['tilt_rms_Nm']<=lim['tilt_moment_rms_error_Nm'];scaling_ok.append(passed)
        residual=np.array([x['inertial_contact_on_ball'] for x in r])-np.array([x['normal_readback_on_ball'] for x in r]);norm=np.linalg.norm(residual[mask],axis=1);ref_peak=float(norm.max());ref_rms=float(np.sqrt((norm**2).mean()))
        if d['friction']==0:reference_ok.append(ref_peak<=lim['force_peak_error_N'] and ref_rms<=lim['force_rms_error_N'])
        else:friction_excited.append(ref_rms>.05)
        results[label]=dict(all_contact_error=full,heldout_error=held,scaling_pass=passed,inertial_minus_normal_peak_N=ref_peak,inertial_minus_normal_rms_N=ref_rms,nut_angle_span_rad=float(np.ptp([x['nut_q'] for x in r])))
    out=dict(sensor_divisor=a.sensor_divisor,inertial_reference_valid=all(reference_ok),friction_excited=all(reference_ok) and len(friction_excited)==2 and all(friction_excited),known_rig_scaling_pass=all(reference_ok+scaling_ok+friction_excited),task_wrench_acceptance=False,limits=lim,results=results,note='Even a known-rig pass requires task transfer checks. Rejected scaling cannot be used for PPO physical-wrench reward or acceptance. Useful axial moment is not tested as a balance target.')
    (a.root/('calibration_analysis.json' if a.sensor_divisor==8 else 'calibration_unity_analysis.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
