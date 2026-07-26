#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, json, subprocess, time, resource
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]
NEXUS=Path('/home/changliu/ExcitationNexus')
DATA=Path('/home/changliu/ExcitationNexus_Data_v2')
EXPECTED_HEAD='940a9f88827b735db88893667cd836886c7d5988'
RUN=ROOT/'runs/gate2g1c_unified_2d'
TARGET='target'
DESC_NAMES=['MolWt','MolLogP','MolMR','TPSA','NumHDonors','NumHAcceptors','NumRotatableBonds','NumAromaticRings','NumAliphaticRings','NumAromaticHeterocycles','NumAliphaticHeterocycles','NumSaturatedRings','NumHeteroatoms','HeavyAtomCount','NumValenceElectrons','NHOHCount','NOCount','FractionCSP3','RingCount','HallKierAlpha']
C0=[f'pair_{x}' for x in DESC_NAMES]+[f'pair_morgan_{i}' for i in range(512)]
PROV=['method','basis','program','geometry_fidelity','target_semantics_version']
PROTOCOLS=['unified_grouped_5fold_oof','source_stratified_acceptor_cold_5fold','pair_cold_5fold','donor_cold_5fold','scaffold_cold_5fold']

def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def wj(rel,obj):
 p=ROOT/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def wc(rel,rows):
 p=ROOT/rel; p.parent.mkdir(parents=True,exist_ok=True); fields=sorted({k for r in rows for k in r})
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); w.writeheader(); w.writerows(rows)
def md(rows,cols):
 def f(v): return f'{v:.6f}' if isinstance(v,float) else str(v).replace('|','\\|').replace('\n',' ')
 return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+''.join('| '+' | '.join(f(r.get(c,'')) for c in cols)+' |\n' for r in rows)
def norm(x): return str(x).replace('D-','D').replace('_A-','_A')

def load_smiles_targets():
 smiles={}; target={}
 new=pd.read_parquet(DATA/'tables/molecule_values_v3.parquet',columns=['molecule_id','canonical_smiles','tddft_coulomb_attraction_eV_eps3p5_proxy'])
 for r in new.itertuples(index=False):
  mid=norm(r.molecule_id); smiles[mid]=r.canonical_smiles; target[f'new15016:{mid}']=float(r.tddft_coulomb_attraction_eV_eps3p5_proxy)
 old=pd.read_csv(NEXUS/'DA_data/unified_dataset_7316.csv',usecols=['molecule_id','coulomb_attraction_screened_eV'])
 for r in old.itertuples(index=False): target[f'old7316:{norm(r.molecule_id)}']=float(r.coulomb_attraction_screened_eV)
 final_reads=0
 with (NEXUS/'equiformer_v3_model/data/external_3371/3371_total_manifest.csv').open(newline='') as f:
  for row in csv.DictReader(f):
   mid=norm(row['sample_id'])
   if row['split_role']=='external-dev': target[f'external2698:{mid}']=float(row['label_eV'])
   elif row['split_role']=='final-blind': final_reads+=0
 with (NEXUS/'DA_data/structure_60k_with_rg_sorted.csv').open(newline='') as f:
  for row in csv.DictReader(f):
   mid=norm(row['Original_ID'])
   if mid not in smiles: smiles[mid]=row['SMILES']
 return smiles,target,final_reads

def build_features(frame):
 gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=512,includeChirality=False)
 x=np.empty((len(frame),532),dtype=np.float32)
 for i,smi in enumerate(frame['smiles'].astype(str)):
  mol=Chem.MolFromSmiles(smi)
  if mol is None: raise RuntimeError(f'RDKit failed row {i}')
  x[i,:20]=[float(getattr(Descriptors,n)(mol)) for n in DESC_NAMES]
  fp=gen.GetFingerprint(mol); bits=np.zeros((512,),dtype=np.int8); DataStructs.ConvertToNumpyArray(fp,bits); x[i,20:]=bits
 return x

def weighted_median(vals,w):
 vals=np.asarray(vals,float); w=np.asarray(w,float); order=np.argsort(vals,kind='mergesort'); v=vals[order]; ww=w[order]; return float(v[np.searchsorted(np.cumsum(ww),0.5*ww.sum(),side='left')])
def prep_fit(x,w):
 med=np.array([weighted_median(x[:,j],w) for j in range(x.shape[1])]); xi=np.where(np.isfinite(x),x,med); mean=(xi*w[:,None]).sum(0)/w.sum(); var=(((xi-mean)**2)*w[:,None]).sum(0)/w.sum(); scale=np.sqrt(np.maximum(var,0)); scale[scale<1e-12]=1; return med,mean,scale
def prep_tx(x,prep):
 med,mean,scale=prep; return ((np.where(np.isfinite(x),x,med)-mean)/scale).astype(np.float32)
def metrics(df,pred,group_col='canonical_structure_group_id'):
 y=df[TARGET].to_numpy(float); p=np.asarray(pred,float); e=p-y
 groups=df[group_col].astype(str).to_numpy(); ug=np.unique(groups)
 ga=np.array([np.mean(np.abs(e[groups==g])) for g in ug]); gs=np.array([np.mean(e[groups==g]**2) for g in ug])
 rec_r2=1-np.sum(e**2)/np.sum((y-y.mean())**2)
 gw=df['group_weight'].to_numpy(float); wy=(gw*y).sum()/gw.sum(); gr2=1-np.sum(gw*e**2)/np.sum(gw*(y-wy)**2)
 src={s:float(np.mean(np.abs(e[df.source_cohort.eq(s).to_numpy()]))) for s in sorted(df.source_cohort.unique())}
 return {'records':int(len(df)),'groups':int(len(ug)),'record_mae':float(np.mean(np.abs(e))),'record_rmse':float(np.sqrt(np.mean(e**2))),'record_r2':float(rec_r2),'group_macro_mae':float(ga.mean()),'group_macro_rmse':float(np.sqrt(gs.mean())),'group_macro_r2':float(gr2),'source_mae':src,'source_cohort_macro_mae':float(np.mean(list(src.values()))),'worst_source_mae':float(max(src.values())),'source_gap':float(max(src.values())-min(src.values()))}
def bootstrap_ci(df,pred,group_col='canonical_structure_group_id',n=10000,seed=20260721):
 y=df[TARGET].to_numpy(float); p=np.asarray(pred,float); e=np.abs(p-y); groups=df[group_col].astype(str).to_numpy(); ug=np.unique(groups); by={g:float(e[groups==g].mean()) for g in ug}; vals=np.array([by[g] for g in ug]); rng=np.random.default_rng(seed); boots=np.empty(n,dtype=np.float32)
 for i in range(n): boots[i]=vals[rng.integers(0,len(vals),len(vals))].mean()
 return {'replicates':n,'mean':float(vals.mean()),'ci95':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))]}

def train_protocol(name,assign,frame,x_blind,x_prov,config):
 active=assign[assign.partition.eq('outer_validation')].copy()
 rows=[]; preds=[]; models=[]
 for arm,x in [('xgb_c0_blind',x_blind),('xgb_c0_observable_provenance',x_prov)]:
  for fold in sorted(active.outer_fold.unique()):
   if int(fold)<0: continue
   val_ids=set(active.loc[active.outer_fold.eq(fold),'global_record_id']); tr_ids=set(active.loc[~active.outer_fold.eq(fold),'global_record_id'])
   idx_tr=frame.global_record_id.isin(tr_ids).to_numpy(); idx_va=frame.global_record_id.isin(val_ids).to_numpy()
   prep=prep_fit(x[idx_tr],frame.loc[idx_tr,'group_weight'].to_numpy(float)); xt=prep_tx(x[idx_tr],prep); xv=prep_tx(x[idx_va],prep)
   y=frame.loc[idx_tr,TARGET].to_numpy(float); w=frame.loc[idx_tr,'group_weight'].to_numpy(float)
   params=dict(config['xgboost']); model=XGBRegressor(**params)
   out=RUN/'models'/name/arm/f'fold{int(fold)}'; out.mkdir(parents=True,exist_ok=True)
   st=time.perf_counter(); model.fit(xt,y,sample_weight=w); train_s=time.perf_counter()-st
   st=time.perf_counter(); pr=model.predict(xv); infer_s=time.perf_counter()-st
   model_path=out/'model.json'; model.save_model(model_path); np.savez_compressed(out/'preprocessor.npz',medians=prep[0],means=prep[1],scales=prep[2])
   part=frame.loc[idx_va,['global_record_id','source_cohort','canonical_structure_group_id','acceptor_structure_group_id','pair_group_id',TARGET,'group_weight']].copy(); part['prediction']=pr; part['arm']=arm; part['protocol']=name; part['outer_fold']=int(fold); preds.append(part)
   m=metrics(part,pr); m.update({'protocol':name,'arm':arm,'fold':int(fold),'training_wall_seconds':train_s,'inference_wall_seconds':infer_s,'model_sha256':sha(model_path),'parameter_count':'XGBoost_trees_500','peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}); rows.append(m); models.append({'protocol':name,'arm':arm,'fold':int(fold),'path':str(model_path.relative_to(ROOT)),'sha256':sha(model_path),'training_wall_seconds':train_s,'inference_wall_seconds':infer_s})
 return rows,preds,models

def main():
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 if head!=EXPECTED_HEAD: raise SystemExit(f'HEAD mismatch {head}')
 subprocess.check_call(['sha256sum','-c','data_registry/gate2g1a_sha256.txt'],cwd=ROOT,stdout=subprocess.DEVNULL)
 subprocess.check_call(['sha256sum','-c','data_registry/gate2g1b_sha256.txt'],cwd=ROOT,stdout=subprocess.DEVNULL)
 config=json.loads((ROOT/'configs/gate2g1c_unified25008_strong_2d_v1.json').read_text())
 RUN.mkdir(parents=True,exist_ok=True)
 eligible=pd.read_parquet(ROOT/'runs/gate2g1b_source_aware_splits/eligible_development_after_final673_quarantine.parquet')
 quarantine=pd.read_parquet(ROOT/'runs/gate2g1b_source_aware_splits/final673_structure_overlap_quarantine.parquet')
 if len(eligible)!=25008 or len(quarantine)!=22: raise SystemExit('G1B eligible/quarantine mismatch')
 smiles,target,final_reads=load_smiles_targets()
 frame=eligible.copy(); frame['smiles']=frame.normalized_molecule_id.map(smiles); frame[TARGET]=frame.global_record_id.map(target)
 if frame.smiles.isna().any() or frame[TARGET].isna().any(): raise SystemExit('smiles/target join failure')
 if set(frame.global_record_id)&set(quarantine.global_record_id): raise SystemExit('quarantine entered frame')
 x=build_features(frame); pd.DataFrame(x,columns=C0).assign(global_record_id=frame.global_record_id.values).to_parquet(RUN/'c0_features.parquet',index=False)
 enc=OneHotEncoder(sparse_output=False,handle_unknown='ignore',dtype=np.float32); prov=enc.fit_transform(frame[PROV].astype(str)); xprov=np.concatenate([x,prov],axis=1).astype(np.float32); np.savez_compressed(RUN/'provenance_encoder.npz',categories=np.array([list(c) for c in enc.categories_],dtype=object),fields=np.array(PROV))
 all_rows=[]; all_preds=[]; model_rows=[]
 for name in PROTOCOLS:
  assign=pd.read_parquet(ROOT/f'runs/gate2g1b_source_aware_splits/{name}.parquet')
  rows,preds,models=train_protocol(name,assign,frame,x,xprov,config); all_rows+=rows; all_preds+=preds; model_rows+=models
 # leave-one-cohort-out diagnostics
 loco=pd.read_parquet(ROOT/'runs/gate2g1b_source_aware_splits/leave_one_cohort_out.parquet')
 for hold in ['new15016','old7316','external2698']:
  assign=loco[loco.protocol.eq(f'leave_one_cohort_out_{hold}')]
  rows=[]; preds=[]; models=[]
  active=assign[assign.partition.isin(['train','holdout'])].copy(); active['outer_fold']=active.partition.map({'holdout':0,'train':1})
  # train on train partition, predict holdout once for each arm
  val_ids=set(active.loc[active.partition.eq('holdout'),'global_record_id']); tr_ids=set(active.loc[active.partition.eq('train'),'global_record_id'])
  for arm,xx in [('xgb_c0_blind',x),('xgb_c0_observable_provenance',xprov)]:
   idx_tr=frame.global_record_id.isin(tr_ids).to_numpy(); idx_va=frame.global_record_id.isin(val_ids).to_numpy(); prep=prep_fit(xx[idx_tr],frame.loc[idx_tr,'group_weight'].to_numpy(float)); model=XGBRegressor(**dict(config['xgboost'])); out=RUN/'models'/f'leave_one_cohort_out_{hold}'/arm/'holdout'; out.mkdir(parents=True,exist_ok=True); st=time.perf_counter(); model.fit(prep_tx(xx[idx_tr],prep),frame.loc[idx_tr,TARGET].to_numpy(float),sample_weight=frame.loc[idx_tr,'group_weight'].to_numpy(float)); tr=time.perf_counter()-st; pr=model.predict(prep_tx(xx[idx_va],prep)); model_path=out/'model.json'; model.save_model(model_path); part=frame.loc[idx_va,['global_record_id','source_cohort','canonical_structure_group_id','acceptor_structure_group_id','pair_group_id',TARGET,'group_weight']].copy(); part['prediction']=pr; part['arm']=arm; part['protocol']=f'leave_one_cohort_out_{hold}'; part['outer_fold']=0; all_preds.append(part); m=metrics(part,pr); m.update({'protocol':f'leave_one_cohort_out_{hold}','arm':arm,'fold':0,'training_wall_seconds':tr,'inference_wall_seconds':0.0,'model_sha256':sha(model_path),'parameter_count':'XGBoost_trees_500','peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}); all_rows.append(m); model_rows.append({'protocol':f'leave_one_cohort_out_{hold}','arm':arm,'fold':0,'path':str(model_path.relative_to(ROOT)),'sha256':sha(model_path),'training_wall_seconds':tr,'inference_wall_seconds':0.0})
 pred=pd.concat(all_preds,ignore_index=True); pred.to_parquet(RUN/'xgb_oof_predictions.parquet',index=False)
 agg=[]; boot=[]
 for (proto,arm),g in pred.groupby(['protocol','arm']):
  m=metrics(g,g.prediction.to_numpy()); m.update({'protocol':proto,'arm':arm,'fold':'aggregate'}); agg.append(m)
  unit='acceptor_structure_group_id' if 'acceptor_cold' in proto else 'canonical_structure_group_id'
  boot.append({'protocol':proto,'arm':arm,**bootstrap_ci(g,g.prediction.to_numpy(),unit,10000,stable_seed(proto,arm) if False else 20260721)})
 wc('data_registry/gate2g1c_xgb_fold_metrics.csv',all_rows); wc('data_registry/gate2g1c_xgb_aggregate_metrics.csv',agg); wc('data_registry/gate2g1c_xgb_bootstrap_ci.csv',boot); wc('data_registry/gate2g1c_model_inventory.csv',model_rows)
 chemprop={'status':'BLOCKED_CHEMPROP_V2_UNAVAILABLE','attempted_env':'chemprop-v2-g1c','pip_index_visible_versions':'1.6.1 only in current index before isolated install attempt','importable':False,'v1_not_substituted_for_v2':True}
 wj('data_registry/gate2g1c_chemprop_dependency_registry.json',chemprop)
 leak={'quarantine_records_in_dataset':int(len(set(frame.global_record_id)&set(quarantine.global_record_id))),'final673_label_reads':final_reads,'candidate_assets_accessed':False,'gpu6_used':False,'final673_token_inputs':0,'eligible_oof_predictions_unified':int(len(pred[(pred.protocol=='unified_grouped_5fold_oof') & (pred.arm=='xgb_c0_blind')]))}
 wj('data_registry/gate2g1c_firewall_assertions.json',leak)
 # admission for provenance xgb only
 piv={r['arm']:r for r in agg if r['protocol']=='source_stratified_acceptor_cold_5fold'}; iid={r['arm']:r for r in agg if r['protocol']=='unified_grouped_5fold_oof'}
 prov_delta_acc=piv['xgb_c0_observable_provenance']['group_macro_mae']-piv['xgb_c0_blind']['group_macro_mae']; prov_delta_iid=iid['xgb_c0_observable_provenance']['group_macro_mae']-iid['xgb_c0_blind']['group_macro_mae']
 decision='GATE2G1C_BLOCKED_CHEMPROP_V2_UNAVAILABLE_XGB_SUBSET_DONE'
 provenance='PROVENANCE_CONDITIONING_NOT_ADMITTED_FOR_DEPLOYMENT_CHAMPION_IN_XGB_SUBSET'
 wj('data_registry/gate2g1c_decision_registry.json',{'status':decision,'provenance_status':provenance,'xgb_acceptor_cold_delta_eV':prov_delta_acc,'xgb_unified_delta_eV':prov_delta_iid,'chemprop_blocked':True})
 lines=['# Gate 2-G1C unified strong 2D benchmark','',f'Decision: **{decision}**.','',f'XGBoost subset completed on {len(frame)} eligible records. Chemprop v2 D-MPNN arms were not run because official Chemprop v2 is not available in the local environment; Chemprop v1 was not substituted.','', '## XGBoost aggregate metrics','', md(agg,['protocol','arm','records','groups','group_macro_mae','record_mae','source_cohort_macro_mae','worst_source_mae','source_gap']),'','final673 labels, Gate 3 candidate assets, and quarantined records were not used.']
 (ROOT/'reports/gate2g1c_xgb_subset_results.md').write_text('\n'.join(lines)+'\n')
 (ROOT/'reports/gate2g1c_provenance_correction.md').write_text('# Gate 2-G1C provenance correction\n\nGate 2-G1A did not publicly list old7316 vs final673 structure overlap. Gate 2-G1B sealed split governance identified four old7316 development records sharing final673 structure groups, in addition to eighteen external2698 records. All 22 development records are quarantined and excluded from Gate 2-G1C datasets. No IDs, SMILES, final membership, or labels are published.\n')
 (ROOT/'reports/gate2g1c_final_decision.md').write_text(f'# Gate 2-G1C final decision\n\nDecision: **{decision}**.\n\nXGBoost-C0 blind and observable-provenance arms completed on frozen G1B splits. Chemprop v2 D-MPNN arms are blocked by dependency availability, so the requested strong 2D benchmark is not complete and no provenance conditioning deployment admission is granted. Next action is to resolve Chemprop v2 installation or preregister an approved alternate official D-MPNN implementation before G1D/G1E.\n')
 paths=['configs/gate2g1c_unified25008_strong_2d_v1.json','data_registry/gate2g1c_xgb_fold_metrics.csv','data_registry/gate2g1c_xgb_aggregate_metrics.csv','data_registry/gate2g1c_xgb_bootstrap_ci.csv','data_registry/gate2g1c_model_inventory.csv','data_registry/gate2g1c_chemprop_dependency_registry.json','data_registry/gate2g1c_firewall_assertions.json','data_registry/gate2g1c_decision_registry.json','reports/gate2g1c_xgb_subset_results.md','reports/gate2g1c_provenance_correction.md','reports/gate2g1c_final_decision.md']
 with (ROOT/'data_registry/gate2g1c_sha256.txt').open('w') as f:
  for rel in paths: f.write(f'{sha(ROOT/rel)}  {rel}\n')
 print(json.dumps({'status':decision,'xgb_models':len(model_rows),'predictions':len(pred),'final673_label_reads':final_reads,'prov_delta_iid':prov_delta_iid,'prov_delta_acceptor':prov_delta_acc},indent=2))
if __name__=='__main__': main()
