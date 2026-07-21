#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,random,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
import pyarrow.dataset as ds

ROOT=Path(__file__).resolve().parents[1]
PRIMARY="tddft_coulomb_attraction_eV_eps3p5_proxy"
FORBIDDEN=("data_registry/gate2e1_validation_unlock_v1.json","logs/gate2e1_validation_metrics.json")
sys.path.insert(0,str(ROOT/'scripts'))
from gate2e1_common import MultiTaskNet,arrays,feature_stats,group_macro_mae,masked_loss,set_seed,task_weights,weighted_stats

def resolve(p):
 p=Path(p);return p if p.is_absolute() else ROOT/p
def sha(p):
 h=hashlib.sha256()
 with resolve(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read_json(p):return json.loads(resolve(p).read_text())
def write_json(p,v):
 p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def config():return read_json('configs/gate2e2a_multitask_crossfit_recovery_v1.json')
def target_graph():return read_json('data_registry/gate2e0_target_graph_v2.json')
def tasks(arm):return [PRIMARY] if arm=='S0' else [PRIMARY,*target_graph()['secondary_optimization']]
def stable_hash(*parts):return hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()
def descriptor_columns(f):return [x for x in f.columns if x.startswith('pair_') and not x.startswith('pair_morgan_') and not x.startswith('pm6_')]
def load_train(protocol):
 c=config();spec=c['protocols'][protocol]
 for item in [spec,c['features'],c['target_graph'],c['primary_sources']['base']]:
  if sha(item['manifest'] if 'manifest' in item else item['path'])!=item.get('manifest_sha256',item.get('sha256')):raise RuntimeError('source hash mismatch')
 m=pd.read_csv(resolve(spec['manifest'])); rows=m[m.partition.eq('train')].copy()
 if len(rows)==0 or set(rows.partition)!={'train'}:raise RuntimeError('official-train firewall')
 forbidden=set(m.loc[~m.partition.eq('train'),'molecule_id']);ids=sorted(rows.molecule_id.astype(str))
 f=pd.read_parquet(resolve(c['features']['path']));desc=descriptor_columns(f);bits=[f'pair_morgan_{i}' for i in range(512)]
 if len(desc)!=20 or any(x not in f for x in bits):raise RuntimeError('C0 schema mismatch')
 rows=rows.merge(f[['molecule_id',*desc,*bits]],on='molecule_id',validate='one_to_one')
 base=c['primary_sources']['base'];tab=ds.dataset(resolve(base['path']),format='parquet').to_table(columns=['molecule_id',base['source_column']],filter=ds.field('molecule_id').isin(ids)).to_pandas().rename(columns={base['source_column']:PRIMARY})
 if protocol=='acceptor_cold':
  fb=c['primary_sources']['protocol_local_fallback']
  if sha(fb['path'])!=fb['sha256']:raise RuntimeError('fallback hash mismatch')
  missing=sorted(set(ids)-set(tab.molecule_id.astype(str)))
  fallback=ds.dataset(resolve(fb['path']),format='parquet').to_table(filter=ds.field('molecule_id').isin(missing)).to_pandas()
  fallback=fallback[['molecule_id',PRIMARY]]
  tab=pd.concat([tab,fallback],ignore_index=True)
  s=c['primary_sources']['acceptor_cold_supplement']
  if sha(s['path'])!=s['sha256']:raise RuntimeError('supplement hash mismatch')
  missing=sorted(set(ids)-set(tab.molecule_id.astype(str)))
  sup=ds.dataset(resolve(s['path']),format='parquet').to_table(filter=ds.field('molecule_id').isin(missing)).to_pandas();tab=pd.concat([tab,sup],ignore_index=True)
 if set(tab.molecule_id.astype(str))!=set(ids) or tab.molecule_id.duplicated().any():raise RuntimeError('primary train coverage')
 if set(tab.molecule_id.astype(str))&forbidden:raise RuntimeError('forbidden primary access')
 rows=rows.merge(tab,on='molecule_id',validate='one_to_one')
 reg=read_json(c['auxiliary_registry']);item=reg['protocols'][protocol]['train']
 if sha(item['artifact_path'])!=item['artifact_sha256']:raise RuntimeError('aux hash mismatch')
 aux=pd.read_parquet(resolve(item['artifact_path']))
 if set(aux.molecule_id.astype(str))!=set(ids):raise RuntimeError('aux train binding')
 rows=rows.merge(aux,on='molecule_id',validate='one_to_one')
 if len(rows)!=len(ids) or not np.isfinite(rows[desc+bits+[PRIMARY]].to_numpy(float)).all():raise RuntimeError('train join/nonfinite')
 return rows.sort_values('molecule_id').reset_index(drop=True),desc,bits

def weighted_unit_targets(frame,unit):
 q=frame.groupby(unit,sort=True).apply(lambda z:np.sum(z[PRIMARY]*z.group_weight)/np.sum(z.group_weight),include_groups=False)
 return {str(k):float(v) for k,v in q.items()}
def reference_weighted_unit_targets(frame,unit):
 out={}
 for key in sorted(frame[unit].astype(str).unique()):
  z=frame[frame[unit].astype(str).eq(key)];num=sum(float(a)*float(b) for a,b in zip(z[PRIMARY],z.group_weight));den=sum(map(float,z.group_weight));out[key]=num/den
 return out
def old_wrong_unit_targets(frame,unit):return frame.groupby(unit,sort=True)[PRIMARY].mean().astype(float).rename(index=str).to_dict()
def assign_units(unit_targets,folds,seed,forced=()):
 forced=set(map(str,forced)); free=sorted(set(unit_targets)-forced); vals=np.array([unit_targets[x] for x in free]);
 bins=pd.qcut(pd.Series(vals),q=min(10,len(np.unique(vals))),labels=False,duplicates='drop').to_numpy() if len(free) else np.array([],int)
 out={x:-1 for x in forced}
 for b in sorted(set(bins.tolist())):
  us=[free[i] for i in range(len(free)) if bins[i]==b];us.sort(key=lambda x:(stable_hash(seed,x),x))
  for j,u in enumerate(us):out[u]=j%folds
 return out
def inner_assignment(frame,unit,outer_fold):
 forced=set(frame.loc[frame.historical_status.eq('HISTORICAL_TRAIN_OVERLAP'),unit].astype(str));targets=weighted_unit_targets(frame,unit);base=assign_units(targets,20,20260721+outer_fold,forced)
 return {u:('inner_fit' if f<0 or f>=3 else 'inner_checkpoint') for u,f in base.items()}

def bootstrap_diff(frame,a,b,unit,reps=10000,seed=20260721):
 d=frame.assign(diff=(frame[a]-frame.y).abs()-(frame[b]-frame.y).abs()).groupby(unit,sort=True)['diff'].mean().to_numpy(float);point=float(d.mean());rng=np.random.default_rng(seed);n=len(d);boot=np.empty(reps)
 for i in range(reps):boot[i]=d[rng.integers(0,n,n)].mean()
 return {'point':point,'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'clusters':n}
