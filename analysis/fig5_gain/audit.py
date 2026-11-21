"""Frozen PGNN trajectory sensitivity, full state/history feedback. CPU only."""
from pathlib import Path
import sys, os, json, hashlib, traceback, platform
R=Path(os.environ['KIT_FROZEN_ROOT']).expanduser().resolve()
O=Path(__file__).resolve().parent
sys.path[:0]=[str(R/'export_validation/vendor'),str(R)]
import numpy as np,torch
from scipy.stats import spearmanr
from evaluate_kit import read_channels,adapt,STEPS
from kit_models import C,PGNN,step,rhs
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
def save(name,value):
    (O/name).write_text(json.dumps(value,indent=2,allow_nan=False))
def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def arr(x):return torch.from_numpy(np.ascontiguousarray(x)).double()
def advance(h,s,u):
    row=torch.cat((u[:,:1],s,u[:,1:]),1)
    return torch.cat((h[:,1:],row[:,None,:]),1)
def trajectory(model,s,h,u):
    path_s=[];path_h=[]
    with torch.no_grad():
        for _ in range(30):
            path_s.append(s);path_h.append(h)
            new=model(s,u,h)
            h=advance(h,s,u);s=new
    return s,torch.stack(path_s),torch.stack(path_h)
def gradients(out,inputs):
    js=[]
    for c in range(2):
        deriv=torch.autograd.grad(out[:,c].sum(),inputs,retain_graph=c==0)
        js.append(torch.cat([v.reshape(len(out),-1) for v in deriv],1))
    return torch.stack(js,1).detach().numpy()
def differentials(model,ss,hh,us):
    full=[];instant=[];local=[];stiff=[]
    for i in range(0,len(ss),18):
        s=ss[i:i+18].detach().requires_grad_();h=hh[i:i+18].detach().requires_grad_();u=us[i:i+18]
        p=model.parameters_for(s,u,h)
        full.append(gradients(step(s,u,p),(s,h)))
        instant.append(gradients(step(s,u,p.detach()),(s,)))
        local.append(gradients(rhs(s,u,p.detach()),(s,)))
        stiff.append(p.detach().numpy())
    return [np.concatenate(a) for a in (full,instant,local,stiff)]
def compose(steps,ns):
    # Back-propagate final state covectors through history shift and append of old s.
    J=np.broadcast_to(np.eye(2,206),(ns,2,206)).copy()
    for block in reversed(steps):
        cur=J[:,:,:2]
        prev=np.zeros_like(J)
        prev[:,:,:block.shape[-1]]=cur@block
        prev[:,:,:2]+=J[:,:,-3:-1]
        prev[:,:,6:]+=J[:,:,2:-4]
        J=prev
    return J
def run(model,pack,archive,record,arm,seed):
    a=pack['ids'];sel=np.unique(np.rint(np.linspace(0,len(a)-1,min(9,len(a)))).astype(int));ids=a[sel];n=len(ids)
    raw=pack['raw'];hs=pack['history'];u=arr(raw[ids][:,[0,3]]);s=arr(raw[ids][:,[1,2]])
    h=arr(np.stack([hs[j-50:j+1] for j in ids]))
    scale=model.std.numpy();scl=scale[1:3]
    inscale=np.r_[scl,np.tile(scale,51)]
    final,ss,hh=trajectory(model,s,h,u)
    F,G,L,P=differentials(model,ss.flatten(0,1),hh.flatten(0,1),u.repeat(30,1))
    J=compose(F.reshape(30,n,2,206),n)*inscale/scl[None,:,None]
    J0=compose(G.reshape(30,n,2,2),n)*inscale/scl[None,:,None]
    margin=np.linalg.eigvals(L.reshape(30,n,2,2)).real.max(axis=-1)
    p=P.reshape(30,n,2)
    direct=np.sin(np.arange(1,207,dtype=float));direct/=np.linalg.norm(direct)
    ds=arr((direct[:2]*scl)[None,:]);dh=arr((direct[2:]*inscale[2:]).reshape(1,51,4));eps=1e-4
    plus=trajectory(model,s[:1]+eps*ds,h[:1]+eps*dh,u[:1])[0]
    minus=trajectory(model,s[:1]-eps*ds,h[:1]-eps*dh,u[:1])[0]
    fd=((plus-minus).numpy()/2/eps/scl)[0];prediction=J[0]@direct
    np.testing.assert_allclose(fd,prediction,rtol=.003,atol=1e-5)
    pack2=np.load(archive)
    np.testing.assert_array_equal(pack2['origin_indices'][sel],ids);at=int(np.where(pack2['horizon_steps']==30)[0][0])
    obs=raw[ids+30][:,[1,2]];np.testing.assert_array_equal(obs,pack2['target'][sel,at])
    dev=np.abs(final.numpy()-pack2['prediction'][sel,at])
    errors=np.linalg.norm((final.numpy()-obs)/scl,axis=1)
    gain=np.linalg.svd(J,compute_uv=False)[:,0];gain0=np.linalg.svd(J0,compute_uv=False)[:,0];gain2=np.linalg.svd(J[:,:,:2],compute_uv=False)[:,0]
    np.savez_compressed(O/(archive.stem+'.npz'),origin_indices=ids,CF_CR=p,finite_horizon_full=J,fixed_schedule_J=J0,local_spectral_abscissa_fixed_coeff=margin,prediction=final.numpy(),target=obs,scale=scl,history_scale=scale,sigma=gain,sigma_fixed_schedule=gain0,sigma_initial_state=gain2,error=errors,fd=fd,jvp=prediction)
    lo,hi=C['physics']['bound_Cf_Cr']
    counts={}
    for j,key in enumerate(['Cf','Cr']):
        v=p[:,:,j];counts[key+'_range']=[float(v.min()),float(v.max())];counts[key+'_near_low']=int(np.sum(v<lo+.01*(hi-lo)));counts[key+'_near_high']=int(np.sum(v>hi-.01*(hi-lo)))
    assoc=spearmanr(errors,gain).statistic
    return dict(record=record,arm=arm,seed=seed,n=n,fd_max=float(abs(fd-prediction).max()),archive_max_abs=dev.max(axis=0).tolist(),archive_max_std=float((dev/scl).max()),rho=float(assoc) if np.isfinite(assoc) else None,gain=gain.tolist(),gain0=gain0.tolist(),gain2=gain2.tolist(),error=errors.tolist(),positive_eigs=int((margin>0).sum()),total=int(margin.size),oversteer_sign=int((p[:,:,0]*C['physics']['lf']>p[:,:,1]*C['physics']['lr']).sum()),**counts)
def main():
    protocol=load(O/'protocol.json');record_summary=R/'evaluation/summary.json'
    assert sha(record_summary)==load(R/'evaluation/completion.json')['summary_sha256']
    cohort=load(R/'frozen_cohort.json')
    cases=[x for x in load(record_summary)['raw_results'] if x['group'] in ('sealed_id','sealed_surface') and x['arm'] in protocol['arms'] and x['mode']=='hold']
    sources={};models={};bindings={};rows=[];fail=[]
    norm=load(R/'integration_result.json')['normalization']
    for case in cases:
        rec,arm,seed=case['record'],case['arm'],case['seed']
        try:
            if rec not in sources:
                sources[rec]=adapt(read_channels(R/'evaluation/source_records'/rec))[0]
            if (arm,seed) not in models:
                path=R/'runs'/case['case']/'final.pt'
                snap=next(x for x in cohort['results'] if x['arm']==arm and x['seed']==seed)
                assert snap['epochs']==1000 and sha(path)==snap['final_sha256']
                saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['binding']==cohort['binding']
                model=PGNN(norm,arm=='pgnn_ref')
                model.load_state_dict(saved['model'])
                model.requires_grad_(False);model=model.eval().double()
                models[arm,seed]=model;bindings[case['case']]=snap['final_sha256']
            archive=R/'evaluation/predictions'/case['prediction_file']
            assert sha(archive)==case['prediction_sha256']
            result=run(models[arm,seed],sources[rec],archive,rec,arm,seed)
            result['group']=case['group'];rows.append(result)
        except Exception:
            fail.append({'case':case['case'],'record':rec,'failure':traceback.format_exc()})
        save('progress.json',dict(ok=len(rows),failed=len(fail),total=len(cases),last_failure=fail[-1:]));print(len(rows),len(fail),flush=True)
    save('results.json',dict(protocol_sha=sha(O/'protocol.json'),code_sha=sha(Path(__file__)),eval_code_sha=sha(R/'evaluate_kit.py'),models_code_sha=sha(R/'kit_models.py'),checkpoint_sha=bindings,summary_sha=sha(record_summary),environment=[sys.version,platform.platform(),torch.__version__,np.__version__],rows=rows,failures=fail))
if __name__=='__main__':main()
