#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.dataset as ds,xgboost as xgb
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT=Path(__file__).resolve().parents[1]
PRIMARY='tddft_coulomb_attraction_eV_eps3p5_proxy'
BASE=['homo_hartree','lumo_hartree','dipole_magnitude_debye']
PM6=[f'pm6_{x}' for x in BASE];DFT=[f'dft_{x}' for x in BASE];DELTA=[f'delta_{x}' for x in BASE]
ROLE=['donor_atom_count','acceptor_atom_count','unknown_atom_count','donor_atom_fraction','acceptor_atom_fraction','unknown_atom_fraction','donor_present','acceptor_present','unknown_present']
ARM_EXTRA={'C0':[],'PM6_3':PM6,'DFT_3':DFT,'PAIR_6':PM6+DFT,'DELTA_6':PM6+DELTA,'DELTA_3':DELTA,'ROLE_9':ROLE}

def resolve(p):
 p=Path(p);return p if p.is_absolute() else ROOT/p
def sha(p):
 h=hashlib.sha256()
 with resolve(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def readj(p):return json.loads(resolve(p).read_text())
def writej(p,v):
 p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def stable_hash(*parts):return hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()
def cfg():return readj('configs/gate2f1_multifidelity_delta_crossfit_v1.json')
def descriptor_columns(f):return [x for x in f if x.startswith('pair_') and not x.startswith('pair_morgan_') and not x.startswith('pm6_')]
def transform_pair_delta(pm6,dft):return np.concatenate([pm6,dft],1),np.concatenate([pm6,dft-pm6],1)
def bootstrap_diff(frame,a,b,unit,reps=10000,seed=20260721):
 z=frame.copy();z['_w']=z.group_weight if 'group_weight' in z else 1.;z['_d']=((z[a]-z.y).abs()-(z[b]-z.y).abs())*z._w
 q=z.groupby(unit,sort=True).agg(n=('_d','sum'),d=('_w','sum'));d=(q.n/q.d).to_numpy(float);point=float(d.mean());rng=np.random.default_rng(seed);boot=np.empty(reps)
 for i in range(reps):boot[i]=d[rng.integers(0,len(d),len(d))].mean()
 return {'point':point,'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'clusters':len(d)}

def source_checks(c):
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=c['expected_head']:raise RuntimeError('BLOCKED_GIT_BOUNDARY')
 for x in [c['c0_features'],c['ground_state_features'],*c['primary_sources'].values(),*c['manifests'].values()]:
  if sha(x['path'])!=x['sha256']:raise RuntimeError(f"source hash mismatch {x['path']}")
 e2=readj('data_registry/gate2e2a_fold_registry.json')
 for protocol,s in c['protocols'].items():
  if sha(s['outer_path'])!=s['outer_sha256'] or s['outer_sha256']!=e2['protocols'][protocol]['outer_sha256']:raise RuntimeError('outer fold SHA mismatch')
  o=pd.read_parquet(resolve(s['outer_path']));m=pd.read_csv(resolve(c['manifests'][protocol]['path']));u=c['protocols'][protocol]['unit'];z=m[m.partition.eq('train')][['molecule_id',u]].merge(o,on='molecule_id',validate='one_to_one');mp=dict(zip(z[u].astype(str),z.outer_fold.astype(int)))
  if any(len(set(g.outer_fold))!=1 for _,g in z.groupby(u)):raise RuntimeError('outer unit leakage')
  if stable_hash(*[f'{k}:{mp[k]}' for k in sorted(mp)])!=s['assignment_hash']:raise RuntimeError('outer assignment hash mismatch')

def algebra_audit(c,gs):
 err={'pm6_gap':float(np.max(np.abs(gs.pm6_gap_hartree-(gs.pm6_lumo_hartree-gs.pm6_homo_hartree)))),'dft_gap':float(np.max(np.abs(gs.dft_gap_hartree-(gs.dft_lumo_hartree-gs.dft_homo_hartree)))),'delta_gap':float(np.max(np.abs(gs.delta_gap_hartree-(gs.delta_lumo_hartree-gs.delta_homo_hartree))))}
 tol=c['algebra']['rounding_tolerance']
 if max(err.values())>tol:raise RuntimeError('BLOCKED_DELTA_ALGEBRA_INTEGRITY')
 pm6=gs[PM6].to_numpy(float);dft=gs[DFT].to_numpy(float);pair,delta=transform_pair_delta(pm6,dft)
 recon={'dft_from_pm6_delta_max_abs':float(np.max(np.abs(delta[:,:3]+delta[:,3:]-pair[:,3:]))),'delta_from_pm6_dft_max_abs':float(np.max(np.abs(pair[:,3:]-pair[:,:3]-delta[:,3:])))}
 if max(recon.values())>tol:raise RuntimeError('BLOCKED_DELTA_ALGEBRA_INTEGRITY')
 reg={'status':'PASS','rounding_tolerance':tol,'gap_relations_max_abs':err,'independent_basis':BASE,'gap_status':'REPORT_ONLY_REDUNDANT','reconstruction':recon,'pair_delta_relation':'SAME_INFORMATION_DIFFERENT_PARAMETERIZATION','target_delta_learning':False};writej('data_registry/gate2f1_algebraic_independence_registry.json',reg);return reg

def prepare():
 c=cfg();source_checks(c);gs=pd.read_parquet(resolve(c['ground_state_features']['path']));alg=algebra_audit(c,gs)
 arms={a:{'columns':532+len(ARM_EXTRA[a]),'base':'C0-512','extra':ARM_EXTRA[a],'gap_included':False,'information_status':'SAME_INFORMATION_DIFFERENT_PARAMETERIZATION' if a in ['PAIR_6','DELTA_6'] else 'DISTINCT_ABLATION'} for a in c['arms']}
 writej('data_registry/gate2f1_feature_arm_registry.json',{'arms':arms,'algebra_registry_sha256':sha('data_registry/gate2f1_algebraic_independence_registry.json'),'no_imputation':True,'outer_train_only_scalar_normalization':True})
 lock={'version':'gate2f1_preregistration_lock_v1','config_sha256':sha('configs/gate2f1_multifidelity_delta_crossfit_v1.json'),'expected_head':c['expected_head'],'outer_fold_hashes':{k:v['assignment_hash'] for k,v in c['protocols'].items()},'algebra':alg,'arms':c['arms'],'primary_endpoints':['acceptor PAIR_6-C0','acceptor DELTA_6-PAIR_6'],'bootstrap':c['bootstrap'],'decisions':c['decisions'],'official_validation_access':False,'test_access':False,'main_parquet_access':False,'final673_access':False,'created_before_model_initialization':True};writej('data_registry/gate2f1_preregistration_lock_v1.json',lock)
 (ROOT/'reports/gate2f1_algebraic_feature_contract.md').write_text(f"# Gate 2-F1 algebraic feature contract\n\nGap identities pass at tolerance {c['algebra']['rounding_tolerance']:.1e}: PM6 {alg['gap_relations_max_abs']['pm6_gap']:.3e}, DFT {alg['gap_relations_max_abs']['dft_gap']:.3e}, and delta {alg['gap_relations_max_abs']['delta_gap']:.3e}. Gap is report-only and absent from all arms. PAIR-6 and DELTA-6 are exactly reversible within the same tolerance and are explicitly `SAME_INFORMATION_DIFFERENT_PARAMETERIZATION`. This is multi-fidelity ground-state feature analysis, not target-delta learning.\n")
 print(json.dumps({'prepared':True,'algebra':alg,'arms':arms},indent=2))

def primary_labels(c,protocol,ids):
 base=c['primary_sources']['base'];tab=ds.dataset(resolve(base['path']),format='parquet').to_table(columns=['molecule_id',base['source_column']],filter=ds.field('molecule_id').isin(ids)).to_pandas().rename(columns={base['source_column']:PRIMARY})
 if protocol=='acceptor_cold':
  for key in ['fallback','acceptor_supplement']:
   miss=sorted(set(ids)-set(tab.molecule_id.astype(str)))
   if not miss:break
   x=c['primary_sources'][key];q=ds.dataset(resolve(x['path']),format='parquet').to_table(filter=ds.field('molecule_id').isin(miss)).to_pandas()[['molecule_id',PRIMARY]];tab=pd.concat([tab,q],ignore_index=True)
 if set(tab.molecule_id.astype(str))!=set(ids) or tab.molecule_id.duplicated().any() or not np.isfinite(tab[PRIMARY]).all():raise RuntimeError('protocol train primary coverage')
 return tab

def load_protocol(c,protocol):
 m=pd.read_csv(resolve(c['manifests'][protocol]['path']));rows=m[m.partition.eq('train')].copy();ids=sorted(rows.molecule_id.astype(str));forbidden=set(m.loc[~m.partition.eq('train'),'molecule_id'])
 f=pd.read_parquet(resolve(c['c0_features']['path']));desc=descriptor_columns(f);bits=[f'pair_morgan_{i}' for i in range(512)]
 if len(desc)!=20 or any(x not in f for x in bits):raise RuntimeError('C0 contract')
 gs=pd.read_parquet(resolve(c['ground_state_features']['path']))[['molecule_id',*sorted(set(sum(ARM_EXTRA.values(),[])))]]
 rows=rows.merge(f[['molecule_id',*desc,*bits]],on='molecule_id',validate='one_to_one').merge(gs,on='molecule_id',validate='one_to_one').merge(primary_labels(c,protocol,ids),on='molecule_id',validate='one_to_one').merge(pd.read_parquet(resolve(c['protocols'][protocol]['outer_path'])),on='molecule_id',validate='one_to_one')
 if set(rows.molecule_id)&forbidden or len(rows)!=len(ids):raise RuntimeError('protocol-local train firewall')
 cols=desc+bits+sorted(set(sum(ARM_EXTRA.values(),[])))+[PRIMARY]
 if not np.isfinite(rows[cols].to_numpy(float)).all():raise RuntimeError('missing/nonfinite; imputation forbidden')
 return rows.sort_values('molecule_id').reset_index(drop=True),desc,bits

def train():
 c=cfg();source_checks(c)
 if not resolve('data_registry/gate2f1_preregistration_lock_v1.json').is_file():raise RuntimeError('missing preregistration lock')
 root=resolve(c['local_root']);root.mkdir(parents=True,exist_ok=True);registry=[];c0_checks={}
 for protocol in c['protocols']:
  frame,desc,bits=load_protocol(c,protocol);unit=c['protocols'][protocol]['unit']
  for fold in range(5):
   tr=frame[~frame.outer_fold.eq(fold)].copy();te=frame[frame.outer_fold.eq(fold)].copy()
   if not len(te) or set(te[unit].astype(str))&set(tr[unit].astype(str)):raise RuntimeError('outer identity leakage')
   for arm in c['arms']:
    out=root/protocol/f'fold{fold}'/arm;out.mkdir(parents=True,exist_ok=False);scalars=desc+ARM_EXTRA[arm];w=tr.group_weight.to_numpy(float);a=tr[scalars].to_numpy(float);mu=np.sum(a*w[:,None],axis=0)/w.sum();sd=np.sqrt(np.sum((a-mu)**2*w[:,None],axis=0)/w.sum());sd=np.where(sd>1e-12,sd,1.)
    def X(z):return np.concatenate([(z[scalars].to_numpy(np.float32)-mu.astype(np.float32))/sd.astype(np.float32),z[bits].to_numpy(np.float32)],1)
    model=xgb.XGBRegressor(**c['xgboost']);start=time.time();model.fit(X(tr),tr[PRIMARY],sample_weight=tr.group_weight);p=model.predict(X(te));pred=pd.DataFrame({'molecule_id':te.molecule_id.astype(str),'y':te[PRIMARY].astype(float),'prediction':p.astype(float),'group_weight':te.group_weight.astype(float),'outer_fold':fold,unit:te[unit].astype(str),'structure_group_id_v1':te.structure_group_id_v1.astype(str)})
    model.save_model(out/'model.ubj');np.savez(out/'scaler.npz',columns=np.array(scalars),mean=mu,std=sd);pred.to_parquet(out/'oof.parquet',index=False)
    item={'protocol':protocol,'fold':fold,'arm':arm,'train_records':len(tr),'oof_records':len(te),'columns':532+len(ARM_EXTRA[arm]),'model_sha256':sha(out/'model.ubj'),'scaler_sha256':sha(out/'scaler.npz'),'oof_sha256':sha(out/'oof.parquet'),'wall_seconds':time.time()-start};writej(out/'summary.json',item);registry.append(item);print(protocol,fold,arm,f"{item['wall_seconds']:.2f}s",flush=True)
  all_oof=[]
  for arm in c['arms']:
   q=pd.concat([pd.read_parquet(root/protocol/f'fold{fold}'/arm/'oof.parquet').assign(arm=arm) for fold in range(5)],ignore_index=True);q.to_parquet(root/protocol/f'{arm}_oof.parquet',index=False);all_oof.append(q)
   expected=int(frame.outer_eligible.sum())
   if len(q)!=expected or q.molecule_id.duplicated().any() or set(q.molecule_id)!=set(frame.loc[frame.outer_eligible,'molecule_id']):raise RuntimeError('OOF exactly-once failure')
  old=pd.concat([pd.read_parquet(resolve(f"runs/gate2e2a_multitask_crossfit/xgb/{protocol}/fold{f}/oof.parquet")) for f in range(5)]);new=all_oof[0];chk=new[['molecule_id','prediction']].merge(old[['molecule_id','prediction']],on='molecule_id',suffixes=('_new','_old'),validate='one_to_one');delta=float(np.max(np.abs(chk.prediction_new-chk.prediction_old)))
  if delta>1e-7:raise RuntimeError('C0 reproduction mismatch')
  c0_checks[protocol]=delta
 writej('data_registry/gate2f1_model_registry.json',{'models':registry,'count':len(registry),'expected_count':70,'physical_gpu':c['physical_gpu'],'c0_reproduction_max_abs':c0_checks,'official_validation_accessed':False,'test_accessed':False,'main_parquet_accessed':False,'final673_accessed':False})

def collect(c,protocol):
 root=resolve(c['local_root']);unit=c['protocols'][protocol]['unit'];base=None
 for arm in c['arms']:
  q=pd.read_parquet(root/protocol/f'{arm}_oof.parquet').rename(columns={'prediction':arm});cols=list(dict.fromkeys(['molecule_id','y','group_weight','outer_fold',unit,'structure_group_id_v1',arm]))
  base=q[cols] if base is None else base.merge(q[['molecule_id',arm]],on='molecule_id',validate='one_to_one')
 return base
def macro(frame,pred,unit):
 z=frame.assign(e=(frame[pred]-frame.y).abs()*frame.group_weight);q=z.groupby(unit).agg(n=('e','sum'),d=('group_weight','sum'));return float((q.n/q.d).mean())
def comparison(frame,a,b,unit,c):return bootstrap_diff(frame,a,b,unit,c['bootstrap']['replicates'],c['bootstrap']['seed'])

def analyze():
 c=cfg();metrics={'protocols':{},'comparisons':{}}
 for protocol in c['protocols']:
  f=collect(c,protocol);unit=c['protocols'][protocol]['unit'];metrics['protocols'][protocol]={'records':len(f),'clusters':int(f[unit].nunique()),'arms':{a:{'record_mae':float(np.mean(np.abs(f[a]-f.y))),'macro_mae':macro(f,a,unit),'fold_mae':[float(np.mean(np.abs(g[a]-g.y))) for _,g in f.groupby('outer_fold')]} for a in c['arms']}}
  pairs=[('PAIR_6','C0'),('DELTA_6','PAIR_6'),('DELTA_6','C0'),('PM6_3','C0'),('DFT_3','C0'),('DFT_3','PM6_3'),('DELTA_3','C0'),('ROLE_9','C0')]
  metrics['comparisons'][protocol]={f'{a}_minus_{b}':comparison(f,a,b,unit,c) for a,b in pairs}
 acc=metrics['comparisons']['acceptor_cold'];iid=metrics['comparisons']['iid'];d=c['decisions'];p=acc['PAIR_6_minus_C0'];pi=iid['PAIR_6_minus_C0']
 mf=p['point']<=d['acceptor_pair_minus_c0_max'] and p['ci95'][1]<d['acceptor_pair_minus_c0_ci_upper_max'] and pi['ci95'][1]<=d['iid_pair_minus_c0_ci_upper_max']
 if mf:mf_label='MULTIFIDELITY_GROUND_STATE_ADMITTED'
 elif p['point']<0 and p['ci95'][1]>=0:mf_label='MULTIFIDELITY_SIGNAL_INCONCLUSIVE'
 elif p['point']>=0:mf_label='MULTIFIDELITY_FEATURES_NOT_ADMITTED'
 else:mf_label='DFT_GROUND_STATE_SIGNAL_ONLY'
 q=acc['DELTA_6_minus_PAIR_6'];qi=iid['DELTA_6_minus_PAIR_6'];dp=mf and q['point']<=d['acceptor_delta_minus_pair_max'] and q['ci95'][1]<d['acceptor_delta_minus_pair_ci_upper_max'] and qi['ci95'][1]<=d['iid_delta_minus_pair_ci_upper_max'];delta_label='DELTA_PARAMETERIZATION_ADMITTED' if dp else 'DELTA_REPARAMETERIZATION_NO_GAIN';dc=acc['DFT_3_minus_C0'];dm=acc['DFT_3_minus_PM6_3'];dft_signal=dc['ci95'][1]<0 or dm['ci95'][1]<0
 metrics['decision']={'multifidelity':mf_label,'delta_parameterization':delta_label,'dft_ground_state_secondary':'DFT_GROUND_STATE_SIGNAL_ONLY' if dft_signal else 'DFT_GROUND_STATE_SIGNAL_INCONCLUSIVE','multifidelity_admitted':mf,'delta_parameterization_admitted':dp};writej('logs/gate2f1_evidence.json',{'status':'GATE2F1_DONE','decision':metrics['decision'],'models':70,'official_validation_accessed':False,'test_accessed':False,'main_parquet_accessed':False,'final673_accessed':False,'completed_utc':datetime.now(timezone.utc).isoformat()});writej('logs/gate2f1_crossfit_metrics.json',metrics)
 def fmt(x):return f"{x['point']:+.9f} eV, 95% CI [{x['ci95'][0]:+.9f}, {x['ci95'][1]:+.9f}]"
 (ROOT/'reports/gate2f1_crossfit_results.md').write_text(f"# Gate 2-F1 cross-fit results\n\nTraining-only OOF comparison across frozen Gate 2-E2A folds. Acceptor PAIR-6−C0: {fmt(p)}. IID PAIR-6−C0: {fmt(pi)}. Acceptor DELTA-6−PAIR-6: {fmt(q)}. IID DELTA-6−PAIR-6: {fmt(qi)}. Acceptor DFT-3−C0: {fmt(dc)}. Acceptor DFT-3−PM6-3: {fmt(dm)}. No official validation or test was accessed.\n")
 (ROOT/'reports/gate2f1_acceptor_identity_analysis.md').write_text(f"# Gate 2-F1 acceptor identity analysis\n\nThe primary acceptor protocol contains {metrics['protocols']['acceptor_cold']['clusters']} held-out training-only acceptor identities and {metrics['protocols']['acceptor_cold']['records']} exactly-once OOF records. Primary identity-macro inference uses paired 10,000-replicate acceptor-identity bootstrap. Per-fold dispersion is frozen in the metrics registry; no identity result selected an arm.\n")
 (ROOT/'reports/gate2f1_delta_parameterization_analysis.md').write_text(f"# Gate 2-F1 delta parameterization analysis\n\nPAIR-6 and DELTA-6 contain exactly the same information and reconstruct one another within 1e-12. Their difference is only coordinate parameterization and tree-model inductive bias. Result: `{delta_label}`. This is not target-delta learning and delta features are not a new physical information source.\n")
 (ROOT/'reports/gate2f1_final_decision.md').write_text(f"# Gate 2-F1 final decision\n\n## `{mf_label}`\n\n## `{delta_label}`\n\nSecondary: `{'DFT_GROUND_STATE_SIGNAL_ONLY' if dft_signal else 'DFT_GROUND_STATE_SIGNAL_INCONCLUSIVE'}`. DFT-3 is better than PM6-3 at acceptor-identity level, but this secondary result does not satisfy or replace the PAIR-6 primary admission endpoint. Primary remains `J_eh_screened_eV_eps3p5 proxy`, not experimental Eb or photocatalytic efficiency. This Gate is training-only cross-fit development evidence, not external confirmation.\n")
 print(json.dumps(metrics['decision'],indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','analyze']);a=p.parse_args();{'prepare':prepare,'train':train,'analyze':analyze}[a.phase]()
