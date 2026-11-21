"""Post-analysis robustness check on the new coordinate Jacobians; no data access."""
from pathlib import Path
import numpy as np,json,hashlib
o=Path(__file__).parent
res=json.loads((o/'results.json').read_text())
proj=np.zeros((206,2))
proj[:2,:]=np.eye(2)
for i in range(51):
    proj[2+4*i+1,0]=1;proj[2+4*i+2,1]=1
assert np.array_equal(proj.T@proj,52*np.eye(2))
rows=[]
for case in res['rows']:
    rec=Path(case['record']).stem
    file=o/(case['group']+'__'+rec+'__'+case['arm']+'_seed'+str(case['seed'])+'__hold.npz')
    x=np.load(file)
    A=x['finite_horizon_full']@proj
    B=x['fixed_schedule_J']@proj
    rows.append(dict(record=case['record'],group=case['group'],arm=case['arm'],seed=case['seed'],gain_per_input_SD=np.linalg.svd(A,compute_uv=False)[:,0].tolist(),gain_per_unit_stacked_norm=(np.linalg.svd(A,compute_uv=False)[:,0]/np.sqrt(52)).tolist(),detached_gain_per_input_SD=np.linalg.svd(B,compute_uv=False)[:,0].tolist(),columns=A.tolist()))
out=[]
for g in ('sealed_id','sealed_surface'):
    for arm in ('pgnn_ref','pgnn_matched'):
        rr=[i for i in rows if i['group']==g and i['arm']==arm]
        G=np.concatenate([i['gain_per_input_SD'] for i in rr])
        GD=np.concatenate([i['detached_gain_per_input_SD'] for i in rr])
        out.append(dict(group=g,arm=arm,N=len(G),median_per_bias_standard_deviation=float(np.median(G)),median_unit_stacked_history=float(np.median(G)/np.sqrt(52)),median_detached_per_bias_sd=float(np.median(GD)),n_per_bias_sd_gain_above_one=int((G>1).sum()),n_unit_stacked_gain_above_one=int((G>np.sqrt(52)).sum())))
(o/'bias_projection.json').write_text(json.dumps(dict(aggregate=out,origins=rows),indent=2))
print(json.dumps(out))
