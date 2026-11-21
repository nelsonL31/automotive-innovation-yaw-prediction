import sys,json,hashlib
from pathlib import Path
import os
R=Path(os.environ['KIT_FROZEN_ROOT']).expanduser().resolve();sys.path.insert(0,str(R));sys.path.insert(0,str(R/'export_validation/vendor'))
import numpy as np
from evaluate_kit import adapt,read_channels
E=R/'evaluation';D=Path(__file__).parent
s=json.loads((E/'summary.json').read_text());cache={};rows=[]
for a in s['raw_results']:
 name=a['record']
 if name not in cache:
  data,_=adapt(read_channels(E/'source_records'/name));cache[name]=data
 data=cache[name];p=E/'predictions'/a['prediction_file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==a['prediction_sha256']
 z=np.load(p);ids=z['origin_indices'];raw=data['raw'];e=z['prediction'].astype(float)-z['target'];j=list(z['horizon_steps']).index(30)
 features={'speed':(abs(raw[ids,0]),[0,2,5,10,1000]),'yaw':(abs(raw[ids,2]),[0,.1,.3,1000]),'future_delta_change':(np.max(abs(np.stack([raw[ids+k,3]-raw[ids,3] for k in range(30)])),axis=0),[0,.02,.1,1000])}
 for feature,(value,bins) in features.items():
  for lo,hi in zip(bins[:-1],bins[1:]):
   mask=(value>=lo)&(value<hi);n=int(mask.sum());rmse=np.sqrt(np.mean(e[mask,j]**2,0)).tolist() if n else None
   rows.append({'group':a['group'],'record':name,'method':a['arm'],'seed':a['seed'],'mode':a['mode'],'feature':feature,'range':[lo,hi],'n':n,'rmse_vy_yaw':rmse})
out={'protocol':json.loads((D/'protocol.json').read_text()),'summary_sha256':hashlib.sha256((E/'summary.json').read_bytes()).hexdigest(),'rows':rows}
(D/'regime_results.json').write_text(json.dumps(out,indent=2))
# Aggregate record first within seed, then seed; include effective support.
groups={}
for r in rows:
 if r['n']:
  key=(r['group'],r['method'],r['mode'],r['feature'],tuple(r['range']));groups.setdefault(key,[]).append(r)
agg=[]
for key,rr in groups.items():
 seeds=sorted(set(str(x['seed']) for x in rr));values=[np.mean([x['rmse_vy_yaw'] for x in rr if str(x['seed'])==seed],0) for seed in seeds]
 agg.append(dict(zip(['group','method','mode','feature','range'],key),rmse_vy_yaw=np.mean(values,0).tolist(),records=len(set(x['record'] for x in rr)),seed_count=len(seeds),origins_per_seed=sum(x['n'] for x in rr)//len(seeds)))
(D/'regime_aggregate.json').write_text(json.dumps(agg,indent=2));print('DONE',len(rows),len(agg))
