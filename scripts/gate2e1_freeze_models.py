#!/usr/bin/env python3
from __future__ import annotations
import json
from gate2e1_common import ROOT,config,parameter_contract,sha,write_json
def main():
 c=config(); models={}; norms={}; training=[]
 for protocol in c['protocols']:
  models[protocol]={};norms[protocol]={}
  for arm in c['arms']:
   models[protocol][arm]={};norms[protocol][arm]={}
   for seed in c['training']['seeds']:
    inner=ROOT/c['local_root']/'inner_selection'/protocol/arm/f'seed{seed}'/'summary.json'; full=ROOT/c['local_root']/'full_refit'/protocol/arm/f'seed{seed}'/'summary.json'; model=full.parent/'model.pt'
    if not all(p.is_file() for p in (inner,full,model)):raise RuntimeError('missing formal artifact')
    i=json.loads(inner.read_text());f=json.loads(full.read_text())
    if not i['finite'] or not f['finite'] or i['official_validation_accessed'] or f['official_validation_accessed']:raise RuntimeError('formal run integrity')
    item={'model_path':str(model.relative_to(ROOT)),'model_sha256':sha(model),'inner_summary_sha256':sha(inner),'full_summary_sha256':sha(full),'best_epoch':i['best_epoch'],'inner_metric_eV':i['best_primary_group_macro_mae_eV'],'physical_gpu_inner':i['physical_gpu'],'physical_gpu_refit':f['physical_gpu'],'wall_seconds_inner':i['wall_seconds'],'wall_seconds_refit':f['wall_seconds'],'peak_gpu_bytes':max(i['peak_gpu_bytes'],f['peak_gpu_bytes'])}
    models[protocol][arm][str(seed)]=item;norms[protocol][arm][str(seed)]={'feature':f['normalization']['feature'],'target':f['normalization']['target']};training.append({'protocol':protocol,'arm':arm,'seed':seed,**item})
 reg={'status':'FROZEN_BEFORE_OFFICIAL_VALIDATION','model_count':len(training),'models':models,'parameter_contract':parameter_contract(),'official_validation_accessed':False,'test_accessed':False};write_json('data_registry/gate2e1_model_registry.json',reg);write_json('data_registry/gate2e1_normalization_registry.json',{'models':norms,'protocol_local':True,'full_train_only':True});write_json('logs/gate2e1_training_registry.json',{'runs':training,'count':len(training),'all_finite':True,'official_validation_accessed':False,'test_accessed':False});print(json.dumps({'models':len(training),'parameters':reg['parameter_contract']},indent=2))
if __name__=='__main__':main()
