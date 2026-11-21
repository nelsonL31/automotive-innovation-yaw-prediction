"""Post-run numerical audit, using archived trajectories; no new training or simulation."""
import csv, json, os, platform, subprocess
from pathlib import Path
import numpy as np
import torch
import planned as exp
from kit_models import factory

def audit():
    d=Path(__file__).resolve().parent
    with (d/'gpu_audit_registration.json').open('x') as f:
        json.dump(dict(purpose='Recompute the CPU RMSEs independently from saved predictions and compare all 216 CPU results with CUDA full-precision on the identical saved inputs. No replacement of CPU results or outcome-dependent repeat.',
                       utc=exp.DAY(),code_sha=exp.sha(Path(__file__)),cpu_protocol=exp.sha(d/'protocol.json')),f,indent=2)
    training_env=json.loads((exp.ROOT/'publication_vsd_20261118/factor/gpu_reproduction/protocol.json').read_text())['runtime']
    available=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    device=torch.device('cuda',available.index(training_env['gpu']))
    torch.cuda.set_device(device);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    a=np.load(d/'simulation_truth_and_controls.npz');p=np.load(d/'predictions.npz')
    raw,hist,ids,target=a['raw'],a['hist'],a['origins'],p['target']
    start=torch.as_tensor(raw[:,ids,1:3].reshape(-1,2),dtype=torch.float32,device=device)
    history0=torch.as_tensor(np.stack([hist[:,o-50:o+1] for o in ids],1).reshape(-1,51,4),dtype=torch.float32,device=device)
    with (d/'per_fit_trajectory.csv').open() as f:rows=list(csv.DictReader(f))
    cpu_max=0.;peak=np.zeros(2);rmse_diff=0.;scores={};gpu_preds={}
    with torch.inference_mode():
        for name,seed in exp.CASES:
            state=torch.load(exp.R/'runs'/f'{name}_seed{seed}'/'final.pt',map_location='cpu',weights_only=False)['model']
            model=factory(name,{k:state[k].tolist() for k in ('mean','std','scale')});model.load_state_dict(state);model=model.to(device).eval()
            for mode in exp.MODES:
                ref=np.array([float(r['yaw_rmse_rad_s']) for r in rows if r['model']==name and int(r['fit_seed'])==seed and r['mode']==mode])
                key=f'{name}_{seed}_{mode}'; cpu=p[key];from_cpu=np.sqrt(np.mean((cpu[:,:,1]-target[:,:,1])**2,axis=1))
                cpu_max=max(cpu_max,np.max(np.abs(ref-from_cpu)))
                steer=torch.as_tensor(a[mode].reshape(-1,30),dtype=torch.float32,device=device)
                s=start;hist_in=history0
                for j in range(30):
                    u=torch.stack((torch.full_like(steer[:,j],10),steer[:,j]),1)
                    next_s=model(s,u,hist_in)
                    hist_in=torch.cat((hist_in[:,1:],torch.cat((u[:,:1],s,u[:,1:]),1)[:,None]),1)
                    s=next_s
                pred=s.cpu().numpy().reshape(target.shape);gpu_preds[key]=pred
                peak=np.maximum(peak,np.abs(pred-cpu).max(axis=(0,1)))
                rmse=np.sqrt(np.mean((pred[:,:,1]-target[:,:,1])**2,axis=1))
                rmse_diff=max(rmse_diff,np.abs(rmse-ref).max())
                scores[name,seed,mode]=(ref,rmse)
    np.savez_compressed(d/'gpu_audit_predictions.npz',**gpu_preds)
    comparisons={};same=0
    for mode in exp.MODES:
        track={model:np.mean([scores[model,s,mode] for s in exp.SEEDS],axis=0) for model in ('ode','data')}
        gaps=track['ode']-track['data']
        same+=int(np.sum(np.sign(gaps[0])==np.sign(gaps[1])))
        comparisons[mode]={'cpu_gap':float(gaps[0].mean()),'gpu_gap':float(gaps[1].mean()),'max_gap_discrepancy':float(np.abs(gaps[0]-gaps[1]).max())}
    assert cpu_max<1e-12
    report=dict(verification_utc=exp.DAY(),gpu=str(device),name=available[device.index],original_gpu=training_env['gpu'],tf32=False,
                cpu_archive_rmse_maxdiff=cpu_max,max_endpoint_abs_difference_vy_mps_and_r_radps=peak.tolist(),max_trajectory_yaw_rmse_difference_radps=rmse_diff,
                trajectory_mode_rankings_matched=same,total=36,paired_gap_by_mode=comparisons,archived_cuda_prediction_sha256=exp.sha(d/'gpu_audit_predictions.npz'))
    with (d/'gpu_audit_result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))
if __name__=='__main__':audit()

