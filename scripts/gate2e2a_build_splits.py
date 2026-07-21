#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess
import pandas as pd
from gate2e2a_common import ROOT,assign_units,config,inner_assignment,load_train,reference_weighted_unit_targets,sha,stable_hash,weighted_unit_targets,write_json

def main():
 c=config();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 if head!=c['expected_head']:raise RuntimeError('BLOCKED_GIT_BOUNDARY')
 out=ROOT/c['local_root']/'folds';out.mkdir(parents=True,exist_ok=True);registry={'version':'gate2e2a_fold_registry_v1','protocols':{},'official_validation_accessed':False,'test_accessed':False,'main_parquet_accessed':False}
 for protocol,spec in c['protocols'].items():
  frame,_,_=load_train(protocol);unit=spec['unit'];prod=weighted_unit_targets(frame,unit);ref=reference_weighted_unit_targets(frame,unit)
  if set(prod)!=set(ref) or not all(abs(prod[u]-ref[u])<=1e-12 for u in prod):raise RuntimeError('production/reference weighted targets differ')
  forced=set(frame.loc[frame.historical_status.eq('HISTORICAL_TRAIN_OVERLAP'),unit].astype(str));outer=assign_units(prod,5,c['outer']['seed'],forced)
  shuffled=frame.sample(frac=1,random_state=91);shuffled_targets=weighted_unit_targets(shuffled,unit)
  if set(shuffled_targets)!=set(prod) or not all(abs(shuffled_targets[u]-prod[u])<=1e-12 for u in prod):raise RuntimeError('weighted target shuffle mismatch')
  shuffled_outer=assign_units(shuffled_targets,5,c['outer']['seed'],forced)
  if shuffled_outer!=outer:raise RuntimeError('assignment shuffle mismatch')
  fold=pd.DataFrame({'molecule_id':frame.molecule_id,'outer_fold':frame[unit].astype(str).map(outer),'outer_eligible':~frame[unit].astype(str).isin(forced)})
  if fold.loc[fold.outer_eligible,'outer_fold'].lt(0).any() or fold.loc[~fold.outer_eligible,'outer_fold'].ne(-1).any():raise RuntimeError('outer coverage')
  inner_rows=[]
  for k in range(5):
   tr=frame[frame[unit].astype(str).map(outer).ne(k)].copy();a=inner_assignment(tr,unit,k)
   inner_rows.extend({'outer_fold':k,'unit_id':u,'inner_partition':p} for u,p in sorted(a.items()))
   if set(tr.loc[tr[unit].astype(str).map(a).eq('inner_checkpoint'),unit])&set(tr.loc[tr[unit].astype(str).map(a).eq('inner_fit'),unit]):raise RuntimeError('inner leakage')
  fold_path=out/f'{protocol}_outer.parquet';inner_path=out/f'{protocol}_inner_units.parquet';fold.to_parquet(fold_path,index=False);pd.DataFrame(inner_rows).to_parquet(inner_path,index=False)
  registry['protocols'][protocol]={'records':len(frame),'units':len(prod),'eligible_records':int(fold.outer_eligible.sum()),'eligible_units':sum(v>=0 for v in outer.values()),'forced_train_records':int((~fold.outer_eligible).sum()),'fold_counts':fold[fold.outer_eligible].outer_fold.value_counts().sort_index().astype(int).to_dict(),'outer_path':str(fold_path.relative_to(ROOT)),'outer_sha256':sha(fold_path),'inner_path':str(inner_path.relative_to(ROOT)),'inner_sha256':sha(inner_path),'assignment_hash':stable_hash(*[f'{u}:{outer[u]}' for u in sorted(outer)]),'production_reference_match':True,'shuffle_invariant':True}
 write_json('data_registry/gate2e2a_fold_registry.json',registry)
 lock={'version':'gate2e2a_preregistration_lock_v1','config_sha256':sha('configs/gate2e2a_multitask_crossfit_recovery_v1.json'),'fold_registry_sha256':sha('data_registry/gate2e2a_fold_registry.json'),'expected_head':c['expected_head'],'decision_thresholds':c['decisions'],'official_validation_access':False,'test_access':False,'final673_access':False,'created_before_training':True}
 write_json('data_registry/gate2e2a_preregistration_lock_v1.json',lock)
 print(json.dumps(registry,indent=2))
if __name__=='__main__':main()
