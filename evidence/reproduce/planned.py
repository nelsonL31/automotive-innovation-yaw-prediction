"""Prospectively registered, frozen KIT planned-steering test; no fit or tuning."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, itertools, json, platform, sys, traceback
import numpy as np
import torch

import os
ROOT = Path(os.environ['KIT_PROJECT_ROOT']).resolve()
R = ROOT/'vsd_new_sources_v2_20261114/execution_20261115'
D = Path(__file__).resolve().parent
sys.path[:0] = [str(R/'export_validation/vendor'), str(R)]
from kit_models import C, factory
SEEDS = (11,22,33)
SCENE_SEEDS = (5271101,5271102,5271103)
MODES = ('A','B','C')
SCENES = tuple(itertools.product((0.9,0.45),('slow','varying'),SCENE_SEEDS))
CASES = tuple(itertools.product(('data','ode'), SEEDS))
DT, SPEED, LAG, RATE = 0.02, 10.0, 0.12, 0.50
N, H, REPLAN = 801, 30, 5
ORIG = np.arange(100,751,50)
Q = C['physics']
DAY = lambda: datetime.now(timezone.utc).isoformat()
def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()
def save(name, obj):
    with (D/name).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
def tocsv(name,rows):
    with (D/name).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
def freeze():
    cohort=json.loads((R/'frozen_cohort.json').read_text())
    files=['kit_models.py','kit_contract.json','official_build_network.py','integration_result.json','frozen_cohort.json']
    for model,s in CASES:
        fn='runs/'+model+'_seed'+str(s)+'/final.pt'
        row,=[x for x in cohort['results'] if x['arm']==model and x['seed']==s]
        assert row['epochs']==1000 and row['final_sha256']==sha(R/fn)
        files.append(fn)
    save('protocol.json',dict(registered=DAY(),script_sha=sha(Path(__file__)),
        original_files={fn:sha(R/fn) for fn in files},python=sys.version,numpy=np.__version__,torch=str(torch.__version__),
        previous_simulator_sha=sha(ROOT/'publication_vsd_20261118/simulation/baseline_interface_repair/independent.py'),
        experiment='Independent synthetic realizations using KIT FIT geometry and effective stiffness. Simulator and predictors share nominal physics; does not validate actual steering plans or a second vehicle.',
        budget='12 scenario trajectories, 14 origins 2..15 s at 1 s spacing, 6 frozen checkpoints, 3 inputs, 216 trajectory-level RMSE values. Paired phase seeds across regimes and surfaces, 3 independent phase blocks, no window/seed pooling or significance testing.',
        primary='0.6 s yaw endpoint RMSE (rad/s). Compute per trajectory per checkpoint, average matched network seeds then equal weight trajectories. Change = ODE minus data; compare B gap minus A gap, C descriptive. Percentages use data-only mean for the same mode. Report all, including non-reversals. Per-model B-A improvements and RMS plan tracking error secondary.',
        seed_blocks=SCENE_SEEDS, fit_seeds=SEEDS, speed_mps=SPEED,dt_s=DT,mu=[0.9,0.45],
        scenario='16 s, constant longitudinal speed. Slow road-wheel feedforward reference 0.020 sin(2pi*0.04t+p0), varying sum of sines with amplitudes (0.035,0.018,0.010) and Hz (0.35,0.80,1.35), independent PCG64 uniform phases within each of three seeds. Shapes selected from prior synthetic design except slow as near-constant check.',
        controller='Synthetic preview steering driver. At each 100 ms issue a 30-step wheel-angle command vector from known maneuver reference plus held feedback 0.06 s times desired minus current yaw, feedback clipped +/-0.025 rad; desired=v*tan(steer_reference)/wheelbase. Initial lateral state and actual wheel angle zero. Log issue index and complete plan. Apply first 5 steps of each plan then replan. No new command version in B after origin.',
        actuator='First-order nominal wheel actuator held commands: alpha=1-exp(-dt/0.12), increment clipped +/-0.50*dt rad. State/input use actual wheel angle at start of interval. B applies exact actuator map to the origin-issued plan; actual C uses same actuator but command versions updated each 0.10 s. Parameters prescribed synthetic settings without physical identification.',
        truth='Separate vectorized float64 saturated KIT single-track plant, FIT axle stiffness, front/rear load caps proportional to mu, static axle load, ZOH actual wheel angle and speed, RK4 4x5 ms steps; repeat 8x2.5 ms, max state discrepancy 1e-5. 51-row history including prior-bin causal means; hold speed 10 m/s for all.',
        inputs='A origin actual angle held; B single origin-issued full plan transformed to nominal wheel sequence; C realized wheel sequence after actuator and replanning. Same initial states/history/trajectory and fixed rate in all. C diagnostic, not deployment or strict bound.',
        stop='One frozen batch, no model or scenario tuning. Retain failed trajectories and tracebacks, no aggregate if nonfinite/missing. Any implementation repair must retain failed attempt and freeze anew before readout. One CPU thread.'))
def reference(times, phases):
    t=times[None,:]
    slow=.020*np.sin(2*np.pi*.04*t+phases[:,0,None])
    fast=np.sum(np.sin(2*np.pi*t[:,:,None]*np.array([.35,.80,1.35])+phases[:,None,:])*np.array([.035,.018,.010]),axis=2)
    return np.where(np.array([k[1]=='slow' for k in SCENES])[:,None],slow,fast)
def actuator(w,u):
    return w+np.clip((-np.expm1(-DT/LAG))*(u-w),-RATE*DT,RATE*DT)
def rhs(s,w,mu):
    m,iz,lf,lr,cf,cr=[Q[k] for k in ('m','Iz','lf','lr','Cf','Cr')]
    den=np.sqrt(SPEED**2+Q['vx_floor']**2);g=SPEED**2/(SPEED**2+Q['speed_gate']**2)
    f=cf*g*(w-np.arctan2(s[:,0]+lf*s[:,1],den))
    r=-cr*g*np.arctan2(s[:,0]-lr*s[:,1],den)
    fcap=mu*m*9.81*lr/(lf+lr);rcap=mu*m*9.81*lf/(lf+lr)
    f=fcap*np.tanh(f/fcap)*np.cos(w);r=rcap*np.tanh(r/rcap)
    return np.column_stack(((f+r)/m-SPEED*s[:,1],(lf*f-lr*r)/iz))
def simulate(nsub):
    K=len(SCENES);mu=np.array([s[0] for s in SCENES])
    phases=np.stack([np.random.default_rng(s[2]).uniform(0,2*np.pi,3) for s in SCENES])
    refs=reference(np.arange(N+H)*DT,phases)
    states=np.zeros((K,N,4));states[:,:,0]=SPEED
    history=states.copy()
    plans=np.empty((K,(N-1)//REPLAN,H));indices=np.arange(0,N-1,REPLAN)
    for j in range(N-1):
        s=states[:,j,1:3].copy();w=states[:,j,3]
        if j%REPLAN==0:
            desired=SPEED*np.tan(refs[:,j])/(Q['lf']+Q['lr'])
            correction=np.clip(.06*(desired-s[:,1]),-.025,.025)
            plans[:,j//REPLAN,:]=refs[:,j:j+H]+correction[:,None]
        actual_command=plans[:,j//REPLAN,j%REPLAN]
        delta_t=DT/nsub;avg=np.zeros_like(s)
        for _ in range(nsub):
            avg+=s/nsub
            a=rhs(s,w,mu);b=rhs(s+delta_t*a/2,w,mu)
            c=rhs(s+delta_t*b/2,w,mu);e=rhs(s+delta_t*c,w,mu)
            s=s+delta_t*(a+2*b+2*c+e)/6
        states[:,j+1,1:3]=s;states[:,j+1,3]=actuator(w,actual_command)
        history[:,j+1,1:3]=avg;history[:,j+1,3]=w
    return states,history,plans,indices
def steering_windows(states,plans,issues):
    assert np.all(issues[ORIG//REPLAN]==ORIG)
    a=np.repeat(states[:,ORIG,3,None],H,axis=2)
    b=a.copy();plan=plans[:,ORIG//REPLAN,:]
    for j in range(1,H):
        b[:,:,j]=actuator(b[:,:,j-1],plan[:,:,j-1])
    c=np.stack([states[:,o:o+H,3] for o in ORIG],axis=1)
    np.testing.assert_allclose(b[:,:,:REPLAN+1],c[:,:,:REPLAN+1],atol=1e-12,rtol=0)
    return {'A':a,'B':b,'C':c}
def forecast(model,state0,history0,angles):
    s=state0;history=history0
    steering=torch.tensor(angles.reshape(-1,H),dtype=torch.float32)
    for j in range(H):
        inp=torch.stack((torch.full_like(steering[:,j],SPEED),steering[:,j]),1)
        nxt=model(s,inp,history)
        row=torch.cat((inp[:,:1],s,inp[:,1:]),1)
        history=torch.cat((history[:,1:],row[:,None]),1);s=nxt
    return s.numpy().reshape(len(SCENES),len(ORIG),2)
def run():
    protocol=json.loads((D/'protocol.json').read_text())
    assert protocol['script_sha']==sha(Path(__file__)) and all(sha(R/n)==v for n,v in protocol['original_files'].items())
    save('started.json',{'utc':DAY(),'protocol_sha':sha(D/'protocol.json'),'platform':platform.platform()})
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert C['dt']==DT and C['history_steps']==51
    raw,hist,plans,issue=simulate(4);fine,*_=simulate(8)
    refinement=np.max(np.abs(fine[:,:,1:3]-raw[:,:,1:3]),axis=(0,1))
    assert max(refinement)<1e-5
    windows=steering_windows(raw,plans,issue)
    np.savez_compressed(D/'simulation_truth_and_controls.npz',raw=raw,hist=hist,plans=plans,issue=issue,origins=ORIG,**windows)
    start=torch.tensor(raw[:,ORIG,1:3].reshape(-1,2),dtype=torch.float32)
    history=torch.tensor(np.stack([hist[:,o-50:o+1] for o in ORIG],1).reshape(-1,51,4),dtype=torch.float32)
    target=raw[:,ORIG+H,1:3];preds={};rows=[]
    # Both causal interfaces can be reproduced from information available at each origin alone.
    for k,o in enumerate(ORIG):
        causal_plan=plans[:,o//REPLAN,:]
        w=raw[:,o,3].copy()
        for j in range(H):
            np.testing.assert_array_equal(w,windows['B'][:,k,j])
            w=actuator(w,causal_plan[:,j])
    with torch.no_grad():
        for name,seed in CASES:
            ck=torch.load(R/'runs'/f'{name}_seed{seed}'/'final.pt',map_location='cpu',weights_only=False)
            saved=ck['model'];norm={k:saved[k].tolist() for k in ['mean','std','scale']}
            model=factory(name,norm);model.load_state_dict(saved,strict=True);model.eval()
            for mode in MODES:
                yp=forecast(model,start,history,windows[mode])
                preds[f'{name}_{seed}_{mode}']=yp
                assert np.isfinite(yp).all()
                rmse=np.sqrt(np.mean((yp-target)**2,axis=1))
                for k,(mu,man,scene) in enumerate(SCENES):
                    rows.append(dict(mu=mu,maneuver=man,scene_seed=scene,model=name,fit_seed=seed,mode=mode,
                        yaw_rmse_rad_s=rmse[k,1],lateral_speed_rmse_m_s=rmse[k,0]))
    assert len(rows)==216
    tocsv('per_fit_trajectory.csv',rows);np.savez_compressed(D/'predictions.npz',target=target,**preds)
    pairs=[]
    for k,(mu,man,scene) in enumerate(SCENES):
        p=dict(mu=mu,maneuver=man,scene_seed=scene)
        for mode in MODES:
            for key in ['data','ode']:
                values=[r['yaw_rmse_rad_s'] for r in rows if (r['mu'],r['maneuver'],r['scene_seed'],r['model'],r['mode'])==(mu,man,scene,key,mode)]
                assert len(values)==3;p[f'{key}_{mode}']=np.mean(values)
            p[f'gap_{mode}']=p[f'ode_{mode}']-p[f'data_{mode}']
        p['gap_B_minus_A']=p['gap_B']-p['gap_A']
        p['wheel_B_C_rms_rad']=np.sqrt(np.mean((windows['B'][k]-windows['C'][k])**2))
        p['issued_command_C_rms_rad']=np.sqrt(np.mean((plans[k,ORIG//REPLAN,:]-windows['C'][k])**2))
        pairs.append(p)
    tocsv('paired_trajectories.csv',pairs)
    def summarize(items):
        x={k:float(np.mean([p[k] for p in items])) for k in items[0] if k not in ('mu','maneuver','scene_seed')}
        for mode in MODES:
            x[f'ode_pct_{mode}']=100*(x['ode_'+mode]/x['data_'+mode]-1)
            x[f'ode_lower_n_{mode}']=int(sum(p['gap_'+mode]<0 for p in items))
        x['A_B_reversal_count']=sum(int(np.sign(p['gap_A'])!=np.sign(p['gap_B'])) for p in items)
        x['n']=len(items)
        return x
    summary=dict(all=summarize(pairs),by_maneuver={m:summarize([p for p in pairs if p['maneuver']==m]) for m in ('slow','varying')},
            completed_utc=DAY(),refinement_max=refinement.tolist(),n_traj=12,n_origins=14,n_rows=len(rows),protocol_sha=sha(D/'protocol.json'))
    save('summary.json',summary)
    save('evidence.json',{f.name:sha(f) for f in D.iterdir() if f.is_file() and f.name!='run.log'})
    print(json.dumps(summary))
if __name__=='__main__':
    if sys.argv[1:] == ['freeze']:freeze()
    elif sys.argv[1:] == ['run']:
        try:run()
        except Exception:
            save('failure.json',dict(utc=DAY(),traceback=traceback.format_exc()));raise
    else:raise SystemExit('Use freeze or run')

