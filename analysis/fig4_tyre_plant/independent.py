"""Registered KIT-conditioned lateral mechanism experiment. Standard library only until freeze."""
from pathlib import Path
import os
import hashlib, json, platform, sys, datetime, itertools, csv
R = Path(os.environ['KIT_FROZEN_ROOT']).expanduser().resolve()
D = Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(name, value):
    with (D/name).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)
def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
methods = ['data','ode']
seeds = [11,22,33]
modes = ['hold','speed','steer','both']
plants = ['linear_slip','saturated']
alphas = [0.7,1.0,1.3]
if '--freeze' in sys.argv:
    manifest = json.loads((R/'frozen_cohort.json').read_text())
    files = ['kit_contract.json','kit_models.py','official_build_network.py','integration_result.json','frozen_cohort.json','evaluation/linear_arx_frozen.npz']
    for a,s in itertools.product(methods,seeds):
        path = 'runs/'+a+'_seed'+str(s)+'/final.pt'
        match, = [x for x in manifest['results'] if x['arm']==a and x['seed']==s]
        assert sha(R/path)==match['final_sha256'] and match['epochs']==1000
        files.append(path)
    dump('protocol.json', dict(registered_utc=now(), analysis_script_sha=sha(Path(__file__)), files={f:sha(R/f) for f in files},
        primary='0.6 s yaw rate RMSE. Test original ODE/data rank reversal: ODE worse than data in hold and better under both replay in matched linear-slip plant. Steering contribution: steer vs hold and both vs speed. Report outcomes in all cells.',
        primary_unit='24 independently drawn multisine phase vectors; origins, training seeds and stiffness/plant variants paired, not independent. Average within trajectory over origins via RMSE, then arithmetic average over fit seeds and trajectories. Paired 95% percentile bootstrap across trajectories (4000 draws, bootstrap seed 9239571); describes phase variability only.',
        generation='PCG64 seeds 691311170 through 691311193; same five phases across all scenario cells per seed: speed 10+2.5sin(2pi*0.08t+p0)+1.5sin(2pi*0.20t+p1) m/s; steering 0.035sin(2pi*.35t+p2)+0.018sin(2pi*.8t+p3)+0.010sin(2pi*1.35t+p4) rad. All five phases iid Uniform(0,2pi).',
        state='Starts at rest laterally. Starts at moving speed. At t=0..16 sample 20ms; continuous state truth integrates speed/steering under 20ms zero-order-hold via f64 RK4, four 5ms substeps. Repeat with eight 2.5ms steps for convergence (both channels max <1e-5). History timestep measured previous ZOH box average using four substep left samples in state truth; future history uses old predicted state as KIT.',
        scenarios='Independent factors for front and rear axle effective stiffness [0.7,1,1.3] times fixed KIT FIT. m,Iz,lf,lr,gate,floor unchanged. Second plant saturates each unprojected axle lateral force with Fmax*tanh(F/Fmax), Fmax=.9*m*g*(opposite wheelbase lever)/wheelbase, g=9.81. Steering cosine unchanged.',
        forecasters='6 fixed checkpoints plus frozen ARX and fixed calibrated nominal predictor. No training. Same 14 origins at t=2..15 seconds inclusive (no future-state inputs); horizon 30 steps; four conditions speed and steering independently held/available. Data only ID is identical architecture and train seeds. Publish every case RMSE and max absolute for vy and r.',
        inference='Single CPU thread, no GPU; finite predictions requirement, all failures recorded and incomplete comparisons suppressed. Secondary full factorial and paired direction counts; report steering vs speed contribution to ODE-data difference by half-sum across complementary conditions. Synthetic truth shares nominal geometry and slip model, cannot establish field generalization or adequacy of saturation model.',
        measures=['yaw RMSE','vy RMSE','yaw max abs','vy max abs'], stop='one batch; no tuning or reruns on outcomes'))
    print(sha(D/'protocol.json'));sys.exit()
protocol=json.loads((D/'protocol.json').read_text())
assert sha(Path(__file__))==protocol['analysis_script_sha']
assert all(sha(R/k)==v for k,v in protocol['files'].items())
dump('started.json',dict(utc=now(),protocol_sha256=sha(D/'protocol.json')))
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'export_validation/vendor'))
import numpy as np, torch
from kit_models import C,factory,step
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
q=C['physics']
controls=[]
for i in range(24):
    phase=np.random.default_rng(691311170+i).uniform(0,2*np.pi,5)
    t = np.arange(801)*.02
    waves=np.sin(2*np.pi*t[:,None]*np.array([.08,.20,.35,.80,1.35])+phase)
    controls.append(np.column_stack((10+waves[:,:2]@[2.5,1.5],waves[:,2:]@[.035,.018,.010])))
cells=list(itertools.product(plants,alphas,alphas,range(24)))
u=np.array([controls[c[3]] for c in cells])
cf=np.array([c[1]*q['Cf'] for c in cells]);cr=np.array([c[2]*q['Cr'] for c in cells])
sat=np.array([c[0]=='saturated' for c in cells])
m,iz,lf,lr=q['m'],q['Iz'],q['lf'],q['lr']
ffcap,frcap=.9*m*9.81*lr/(lf+lr),.9*m*9.81*lf/(lf+lr)
def truth_rhs(s,v,d):
    den=np.sqrt(v*v+q['vx_floor']**2);g=v*v/(v*v+q['speed_gate']**2)
    front=cf*g*(d-np.arctan2(s[:,0]+lf*s[:,1],den))
    rear=-cr*g*np.arctan2(s[:,0]-lr*s[:,1],den)
    front=np.where(sat,ffcap*np.tanh(front/ffcap),front)*np.cos(d)
    rear=np.where(sat,frcap*np.tanh(rear/frcap),rear)
    return np.column_stack(((front+rear)/m-v*s[:,1],(lf*front-lr*rear)/iz))
def generate(n):
    raw=np.zeros((len(cells),801,4))
    raw[:,:,0]=u[:,:,0];raw[:,:,3]=u[:,:,1]
    hist=raw.copy();h=.02/n
    for j in range(800):
        s=raw[:,j,1:3].copy()
        mean=np.zeros_like(s);v,d=u[:,j].T
        for _ in range(n):
            mean+=s/n
            a=truth_rhs(s,v,d);b=truth_rhs(s+h*a/2,v,d)
            c=truth_rhs(s+h*b/2,v,d);e=truth_rhs(s+h*c,v,d)
            s=s+h*(a+2*b+2*c+e)/6
        raw[:,j+1,1:3]=s
        hist[:,j+1,0]=v;hist[:,j+1,3]=d;hist[:,j+1,1:3]=mean
    return raw,hist
raw,hist=generate(4);fine,_=generate(8)
convergence=np.max(np.abs(fine[:,:,1:3]-raw[:,:,1:3]),axis=(0,1))
assert np.max(convergence)<1e-5,convergence
ids=np.arange(100,751,50);assert len(ids)==14
truth=raw[:,ids+30,1:3]
np.savez_compressed(D/'synthetic_truth.npz',raw=raw,history=hist,origin=ids)
objects={}
norm=None
for arm,seed in itertools.product(methods,seeds):
    saved=torch.load(R/'runs'/f'{arm}_seed{seed}'/'final.pt',map_location='cpu',weights_only=False)['model']
    norm={k:v.tolist() for k,v in saved.items() if k in ['mean','std','scale']}
    net=factory(arm,norm);net.load_state_dict(saved);objects[f'{arm}{seed}']=net.eval()
coef=torch.tensor(np.load(R/'evaluation/linear_arx_frozen.npz')['coef'],dtype=torch.float32)
class ARX:
    def __call__(self,s,inp,h):
        base=objects['data11']
        x=base.inputs(s,inp,h)
        return s+torch.cat((x,torch.ones_like(x[:,:1])),1)@coef*base.scale
objects.update(nominal=lambda s,u,h:step(s,u),arx=ARX())
history0=torch.from_numpy(np.stack([hist[:,k-50:k+1] for k in ids],axis=1).reshape(-1,51,4).astype('float32'))
state0=torch.tensor(raw[:,ids,1:3].reshape(-1,2),dtype=torch.float32)
future_u=torch.from_numpy(np.stack([u[:,k:k+30] for k in ids],axis=1).reshape(-1,30,2).astype('float32'))
def make_pred(model,mode):
    predicts=[]
    for start in range(0,len(state0),512):
        sel=slice(start,start+512)
        hist0=history0[sel];s=state0[sel];us=future_u[sel]
        for j in range(30):
            v=us[:,j if mode in ['both','speed'] else 0,:1]
            d=us[:,j if mode in ['both','steer'] else 0,1:]
            inp=torch.cat((v,d),1);new=model(s,inp,hist0)
            hist0=torch.cat((hist0[:,1:],torch.cat((v,s,d),1)[:,None]),1)
            s=new
        predicts.append(s.numpy())
    return np.concatenate(predicts).reshape(truth.shape)
preds={};rows=[];scores={}
with torch.no_grad():
    for name,model in objects.items():
        for mode in modes:
            key=(name,mode);yp=make_pred(model,mode)
            preds['_'.join(key)]=yp;er=np.abs(yp-truth)
            scores[key]=np.sqrt(np.mean(er**2,axis=1))
            for i,(plant,front,rear,traj) in enumerate(cells):
                vy,r=scores[key][i];mx=er[i].max(0).tolist()
                rows.append(dict(plant=plant,front=front,rear=rear,trajectory=traj,case=name,mode=mode,vy_rmse=vy,yaw_rmse=r,vy_max=mx[0],yaw_max=mx[1],finite=int(np.isfinite(er[i]).all())))
        print('done',name,flush=True)
np.savez_compressed(D/'all_predictions.npz',truth=truth,**preds)
with (D/'all_metrics.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
assert all(r['finite'] for r in rows),'nonfinite; see per-case metrics'
boot=np.random.default_rng(9239571).integers(0,24,(4000,24))
summary=[]
for plant,front,rear in itertools.product(plants,alphas,alphas):
    mask=np.array([k[:3]==(plant,front,rear) for k in cells]);assert sum(mask)==24
    def eval_method(a,j):
        names=[a+str(s) for s in seeds] if a in methods else [a]
        return np.mean([scores[(n,j)][mask] for n in names],axis=0)
    gaps={};res={}
    def summarize(v):
        return dict(mean=float(v.mean()),ci95=np.percentile(v[boot].mean(axis=1),[2.5,97.5]).tolist(),negative_count=int((v<0).sum()),positive_count=int((v>0).sum()))
    for j in modes:
        out={a:eval_method(a,j).mean(axis=0).tolist() for a in ['data','ode','arx','nominal']}
        dg=eval_method('ode',j)-eval_method('data',j);gaps[j]=dg
        out['ode_vs_data_pct']=(100*(np.array(out['ode'])/out['data']-1)).tolist()
        out['yaw_ode_minus_data']=summarize(dg[:,1]);res[j]=out
    dsteer=(gaps['steer']-gaps['hold']+gaps['both']-gaps['speed'])*.5
    dspeed=(gaps['speed']-gaps['hold']+gaps['both']-gaps['steer'])*.5
    assert np.allclose(dsteer+dspeed,gaps['both']-gaps['hold'])
    summary.append(dict(plant=plant,front=front,rear=rear,modes=res,steering_gap_change=summarize(dsteer[:,1]),speed_gap_change=summarize(dspeed[:,1]),reversal_scenarios=int(((gaps['hold'][:,1]>0)&(gaps['both'][:,1]<0)).sum())))
dump('cell_summary.json',summary)
completion=dict(utc=now(),python=sys.version,numpy=np.__version__,torch=torch.__version__,processor=platform.machine(),axes=['vy_mps','yaw_radps'],trajectories=len(cells),unique_control_seeds=24,origin_count_per_record=len(ids),rows=len(rows),nonfinite_rows=sum(1-r['finite'] for r in rows),max_refinement_change=convergence.tolist(),protocol_sha=sha(D/'protocol.json'),evidence_sha={f:sha(D/f) for f in ['synthetic_truth.npz','all_predictions.npz','all_metrics.csv','cell_summary.json']})
dump('completion.json',completion)
for z in summary:
    if z['front']==z['rear']==1:
        print(z['plant'],json.dumps(z),flush=True)
print(json.dumps(completion))

