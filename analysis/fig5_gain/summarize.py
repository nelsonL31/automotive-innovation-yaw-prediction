from pathlib import Path
import hashlib,json,sys,itertools
import numpy as np
from statistics import median
o=Path(__file__).resolve().parent
res=json.loads((o/'results.json').read_text());rows=res['rows']
out={'failures':res['failures'],'record_seed_cases':len(rows),'max_abs_fd_error':max(x['fd_max'] for x in rows),'max_abs_archive':np.max([x['archive_max_abs'] for x in rows],axis=0).tolist(),'max_difference_in_FIT_std':max(x['archive_max_std'] for x in rows),'aggregate':[]}
for group,arm in itertools.product(('sealed_id','sealed_surface'),('pgnn_ref','pgnn_matched')):
    rr=[x for x in rows if x['group']==group and x['arm']==arm]
    def macro_median(name):
        return median([median([median(np.atleast_1d(x[name])) for x in rr if x['record']==rec]) for rec in sorted(set(x['record'] for x in rr))])
    g=np.concatenate([x['gain'] for x in rr]);g0=np.concatenate([x['gain0'] for x in rr]);g2=np.concatenate([x['gain2'] for x in rr]);tp=sum(x['total'] for x in rr)
    z=dict(group=group,arm=arm,records=len(set(x['record'] for x in rr)),origins_seeds=int(len(g)),origin_steps=tp,positive_eigs=sum(x['positive_eigs'] for x in rr),sign_oversteer=sum(x['oversteer_sign'] for x in rr),gain_median_pooled=float(np.median(g)),gain_minmax=[float(g.min()),float(g.max())],gain_above_one=int(sum(g>1)),gain_stateonly_above_one=int(sum(g2>1)),gain_detached_above_one=int(sum(g0>1)),gain_detached_median_pooled=float(np.median(g0)),gain_stateonly_median_pooled=float(np.median(g2)),gain_ratio_median=float(np.median(g/g0)),gain_higher_than_detached=int(sum(g>g0)),rho_median=macro_median('rho'),rho_range=[min(x['rho'] for x in rr),max(x['rho'] for x in rr)])
    for key in ('Cf','Cr'):
        z[key+'_range']=[min(x[key+'_range'][0] for x in rr),max(x[key+'_range'][1] for x in rr)]
        for bound in ('near_high','near_low'):z[key+'_'+bound]=sum(x[key+'_'+bound] for x in rr)
    out['aggregate'].append(z)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(o/'aggregate.json').write_text(json.dumps(out,indent=2))
(o/'sha256_manifest.json').write_text(json.dumps({p.name:sha(p) for p in sorted(o.iterdir()) if p.suffix in ('.py','.json','.npz') and p.name not in ('progress.json','sha256_manifest.json')},indent=2))
print(json.dumps(out))
