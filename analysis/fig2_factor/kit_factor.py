"""Frozen KIT input factorial. All outcomes are post-hoc, descriptive."""
from pathlib import Path
import os
import sys
import datetime
import hashlib
import json
import platform
import traceback
from collections import defaultdict

D = Path(__file__).resolve().parent
R = Path(os.environ['KIT_FROZEN_ROOT']).expanduser().resolve()
E = R / 'evaluation'
sys.path[:0] = [str(R / 'export_validation' / 'vendor'), str(R)]
import evaluate_kit as ev
import numpy as np
import torch

GROUPS = ['sealed_id', 'sealed_surface']
SEEDS = [11, 22, 33]
ARMS = ['data', 'ode', 'linear_ARX']
MODES = {'hold': (False, False), 'speed_only': (True, False),
         'steer_only': (False, True), 'replay': (True, True)}
BASE_SHA = '5febe2a355d7b7551170181aae39e3304ad40dac2a0d171c6e1e606eabf754b2'
ATOL, RTOL = 0.0005, 0.00005

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def load(path):
    return json.loads(path.read_text())

def write(name, obj):
    with (D / name).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def predict(model, raw, hist, origins, flags):
    speed, steering = flags
    state = torch.from_numpy(raw[origins][:, [1, 2]]).cuda()
    history = torch.from_numpy(np.stack([hist[i-50:i+1] for i in origins])).cuda()
    outputs = []
    for j in range(max(ev.STEPS)):
        inputs = torch.from_numpy(np.stack([
            raw[origins + (j if speed else 0), 0],
            raw[origins + (j if steering else 0), 3]], axis=1)).cuda()
        new = model(state, inputs, history)
        if j + 1 in ev.STEPS:
            outputs.append(new.cpu().numpy())
        row = torch.cat((inputs[:, :1], state, inputs[:, 1:]), dim=1)
        history = torch.cat((history[:, 1:], row[:, None, :]), dim=1)
        state = new
    return np.stack(outputs, axis=1)

def contrasts(v):
    a, b, c, d = [v[k] for k in MODES]
    return {'speed_at_steer_hold': b-a, 'speed_at_steer_replay': d-c,
            'steer_at_speed_hold': c-a, 'steer_at_speed_replay': d-b,
            'speed_marginal': ((b-a)+(d-c))/2,
            'steer_marginal': ((c-a)+(d-b))/2,
            'interaction': d-c-b+a, 'joint_replay_minus_hold': d-a}

def run():
    ev.resources()
    assert sha(E/'summary.json') == BASE_SHA
    summary = load(E/'summary.json')
    frozen = load(R/'frozen_cohort.json')
    cohort = load(E/'cohort_before_test.json')
    assert frozen == cohort['training'] and frozen['binding'] == summary['training_binding']
    assert sha(R/'evaluate_kit.py') == cohort['code_sha256']
    logs = [json.loads(s) for s in (E/'access_log.jsonl').read_text().splitlines()]
    records = sorted(set((x['group'], x['record']) for x in summary['raw_results'] if x['group'] in GROUPS))
    assert [sum(g==group for g, _ in records) for group in GROUPS] == [5, 4]
    assert len(summary['missing_records']) == 16
    orig = {(x['group'], x['record'], x['case'], x['mode']):x for x in summary['raw_results']}
    weights = {f"{x['arm']}_seed{x['seed']}":x for x in frozen['results'] if x['arm'] in ('ode','data')}
    cases = [(arm, s, f'{arm}_seed{s}') for arm in ['data','ode'] for s in SEEDS]
    cases.append(('linear_ARX',None,'linear_ARX'))
    assert len(weights)==6
    for name, row in weights.items():
        assert row['epochs']==1000 and sha(R/'runs'/name/'final.pt') == row['final_sha256']
    arx_path = E/'linear_arx_frozen.npz'
    assert sha(arx_path) == cohort['arx_sha256']
    src = {v['record']:v['raw_sha256'] for v in logs}
    for _, record in records:
        assert sha(E/'source_records'/record) == src[record]
    protocol = {
        'utc_before_new_predictions':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'objective':'Separate sensitivity to longitudinal speed and steering preview in previously reported replay advantage. This conditional diagnostic of existing models is not a fresh test or a causal tyre-effect estimate.',
        'amendment':'CPU run aborted at preregistered parity tolerance on third record. Full original attempt preserved in parent directory. Change numerical backend to original GPU instead of relaxing threshold. No mixed-cell effects inspected before amendment.',
        'budget':'original GPU 0, one CPU thread, stop if other GPU processes are present; one forward evaluation per record/case/cell; no weight updates, hyperparameter selection, validation or additional test records',
        'groups':records, 'omissions':summary['missing_records'], 'arms_and_cases':cases,
        'seeds':SEEDS, 'modes_speed_steer_flags':MODES, 'primary':'0.6 s yaw RMS on four surface-shift runs; repeat on five within-surface runs; lateral RMS and full original horizons/metrics reported',
        'records_seeds':'record equal weighted, followed by seed mean; no pooled origins, no seed-as-independent inference or p-values',
        'contrasts':'RMSE in native units; speed marginal=(speed_only-hold+replay-steer_only)/2; steer marginal=(steer_only-hold+replay-speed_only)/2; interaction=replay-speed_only-steer_only+hold. Negative marginal lowers RMSE. Paired ODE-minus-data contrasts match seed and record.',
        'inclusion':'all and only previously evaluable dynamic runs in the two previously opened groups. Original 16 missing-signal records stay in manifest. Single-pass diagnostics do not authorize model selection.',
        'guard':'Verify previous data, code and complete 1000-epoch weights then verify CPU hold/replay against every published frozen prediction and the original target/origins. Any failure invalidates the aggregate. Dedicated toy test perturbing all future lateral and yaw measurements.',
        'gpu_cpu_parity_atol':ATOL,'gpu_cpu_parity_rtol':RTOL,
        'prior_summary_sha256':BASE_SHA, 'training_binding':frozen['binding'],
        'cohort_sha256':sha(E/'cohort_before_test.json'), 'weights':weights,'arx_sha256':sha(arx_path),
        'source_sha256':{record:src[record] for _,record in records},
        'code_sha256':{str(p):sha(p) for p in (Path(__file__),R/'evaluate_kit.py',R/'kit_models.py',R/'kit_data.py',R/'export_validation'/'kit_mat_reader.py')},
        'runtime':{'gpu':torch.cuda.get_device_name(0),'python':sys.version,'numpy':np.__version__,'torch':str(torch.__version__),'platform':platform.platform(),'threads':torch.get_num_threads()}}
    write('protocol.json',protocol)
    with np.load(arx_path,allow_pickle=False) as archive:
        arx_coef=archive['coef'].copy()
    train = ev.Dataset()
    models={}
    for arm, seed, case in cases:
        if arm=='linear_ARX':
            model = ev.Baseline(arm,train.norm,arx_coef)
        else:
            ck=torch.load(R/'runs'/case/'final.pt',map_location='cpu',weights_only=False)
            assert ck['binding']==frozen['binding']
            model=ev.factory(arm,train.norm)
            model.load_state_dict(ck['model'],strict=True)
        models[case]=model.cuda().eval()
    rs=np.random.default_rng(20261118)
    toy=rs.normal(size=(112,4)).astype('f')
    changed=toy.copy()
    changed[51:,1:3]+=50
    for flag in MODES.values():
        y=predict(models['ode_seed11'],toy,toy,np.array([50]),flag)
        np.testing.assert_array_equal(y,predict(models['ode_seed11'],changed,changed,np.array([50]),flag))
    pred_dir=D/'predictions'
    pred_dir.mkdir(exist_ok=False)
    rows=[]
    parity=defaultdict(float)
    for group, file in records:
        ev.resources()
        rec, missing=ev.adapt(ev.read_channels(E/'source_records'/file))
        assert not missing
        ids,raw,hist=rec['ids'],rec['raw'],rec['history']
        target=np.stack([raw[ids+k][:,[1,2]] for k in ev.STEPS],axis=1)
        for arm,seed,case in cases:
            ev.resources()
            pred_payload={}
            for mode, flags in MODES.items():
                forecasts=np.concatenate([predict(models[case],raw,hist,ids[i:i+256],flags) for i in range(0,len(ids),256)])
                assert forecasts.shape==target.shape and np.isfinite(forecasts).all()
                pred_payload[mode]=forecasts
                if mode in ('hold','replay'):
                    prior=orig[group,file,case,mode]
                    previous=E/'predictions'/prior['prediction_file']
                    assert sha(previous)==prior['prediction_sha256'] and prior['status']=='ok'
                    with np.load(previous,allow_pickle=False) as z:
                        np.testing.assert_array_equal(ids,z['origin_indices'])
                        np.testing.assert_array_equal(rec['time'][ids],z['origin_time'])
                        np.testing.assert_array_equal(ev.STEPS,z['horizon_steps'])
                        np.testing.assert_array_equal(target,z['target'])
                        err=np.max(np.abs(forecasts-z['prediction']),axis=(0,1))
                        for label,diff in zip(('vy','yaw'),err):
                            parity[label]=max(parity[label],float(diff))
                        np.testing.assert_allclose(forecasts,z['prediction'],atol=ATOL,rtol=RTOL)
                metric=ev.metrics(forecasts,target)
                assert metric['status']=='ok'
                rows.append(dict(group=group,record=file,arm=arm,case=case,seed=seed,mode=mode,**metric))
            for key in MODES:
                np.testing.assert_array_equal(pred_payload[key][:,0],pred_payload['hold'][:,0])
            np.savez_compressed(pred_dir/(group+'__'+Path(file).stem+'__'+case+'.npz'),origin_indices=ids,origin_time=rec['time'][ids],horizon_steps=np.array(ev.STEPS),target=target,**pred_payload)
        print('completed',group,file,len(ids),flush=True)
    write('record_results.json',rows)
    metrics=ev.P['metrics']
    agg=[]
    paired=[]
    for group in GROUPS:
        names=[n for g,n in records if g==group]
        for kidx,k in enumerate(ev.STEPS):
            by_arm={}
            for arm in ARMS:
                sub=[x for x in rows if x['group']==group and x['arm']==arm]
                arm_seeds=[None] if arm=='linear_ARX' else SEEDS
                lookup={(x['record'],x['seed'],x['mode']):x['by_horizon'][kidx] for x in sub}
                for metric in metrics:
                    record_cells={n:{mode:float(np.mean([lookup[n,s,mode][metric] for s in arm_seeds])) for mode in MODES} for n in names}
                    cells={mode:float(np.mean([record_cells[n][mode] for n in names])) for mode in MODES}
                    effects=contrasts(cells)
                    record_effects={n:contrasts(record_cells[n]) for n in names}
                    agg.append({'group':group,'method':arm,'steps':k,'metric':metric,'cells':cells,'effects':effects,'record_cells':record_cells,'record_effects':record_effects})
                    by_arm[arm,metric]=record_cells
            for metric in metrics:
                diffs={n:{m:by_arm['ode',metric][n][m]-by_arm['data',metric][n][m] for m in MODES} for n in names}
                dmean={m:float(np.mean([r[m] for r in diffs.values()])) for m in MODES}
                denom={m:float(np.mean([r[m] for r in by_arm['data',metric].values()])) for m in MODES}
                pct={m:100*dmean[m]/denom[m] for m in MODES}
                paired.append({'group':group,'steps':k,'metric':metric,'ode_minus_data':dmean,'pct_vs_data':pct,'effects_on_ode_minus_data':contrasts(dmean),'record_deltas':diffs})
    write('aggregate.json',{'qualification':'Post-hoc original KIT same-campaign runs, noncausal preview diagnostic; absence of future response verified by synthetic invariance. Results descriptive.','parity_max_abs':dict(parity),'total_record_case_cells':len(rows),'new_mixed_cells':sum(x['mode'] in ('speed_only','steer_only') for x in rows),'aggregate':agg,'paired':paired})
    manifests={p.name:sha(p) for p in sorted(pred_dir.iterdir())}
    write('completion.json',{'status':'complete','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'aggregate_sha256':sha(D/'aggregate.json'),'protocol_sha256':sha(D/'protocol.json'),'record_results_sha256':sha(D/'record_results.json'),'prediction_sha256':manifests,'files':len(manifests),'parity_max_abs':dict(parity),'non_finite':0})
    print('complete',len(rows),dict(parity),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    try:
        with torch.inference_mode():
            run()
    except Exception:
        trace=traceback.format_exc()
        with (D/'failure.log').open('a') as f:
            f.write(trace)
        raise

