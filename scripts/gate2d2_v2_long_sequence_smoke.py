#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import torch
from gate2d2_v2_model import ROOT,anon,load_model,pooled,unique_inputs

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--physical-gpu',type=int,required=True); args=ap.parse_args()
 device=torch.device('cuda:0'); c,tok,model=load_model(device)
 correction_path=ROOT/'configs/gate2d2_v2_sequence_length_correction.json'; lock_path=ROOT/'data_registry/gate2d2_v2_sequence_length_correction_lock.json'; correction=json.loads(correction_path.read_text()); lock=json.loads(lock_path.read_text()); correction_sha=hashlib.sha256(correction_path.read_bytes()).hexdigest()
 if correction_sha!=lock['config_sha256'] or lock['molecular_forward_count_before_lock']!=0: raise RuntimeError('sequence-length correction lock mismatch')
 bins=[tuple(x) for x in correction['bins']]; pools=unique_inputs(); selected=[]; summary={}
 for kind,values in pools.items():
  rows=[]
  for text in values:
   ids=tok(text,add_special_tokens=True,truncation=False)['input_ids']; rows.append((len(ids),text))
  summary[kind]={'count':len(rows),'max_length':max(x[0] for x in rows),'bins':{}}
  for lo,hi in bins:
   candidates=sorted((n,s) for n,s in rows if lo<=n<=hi)
   if candidates:
    picks=[candidates[0],candidates[-1]] if len(candidates)>1 else [candidates[0]]
    summary[kind]['bins'][f'{lo}-{hi}']=len(candidates)
    selected.extend((kind,n,s) for n,s in picks)
 expected=correction['frozen_tokenizer_maxima']
 if {k:summary[k]['max_length'] for k in expected}!=expected: raise RuntimeError('audited maximum token length mismatch')
 dedup=[]; seen=set()
 for row in selected:
  key=(row[0],row[2])
  if key not in seen: dedup.append(row); seen.add(key)
 repeat_max=0.0; batch_max=0.0; records=[]; reference=dedup[0][2]; started=time.perf_counter()
 for kind,length,text in dedup:
  one=tok([text],padding=True,truncation=False,return_tensors='pt').to(device)
  a=pooled(model,one); b=pooled(model,one); repeat=float((a-b).abs().max().cpu()); repeat_max=max(repeat_max,repeat)
  other=reference if text!=reference else dedup[-1][2]
  pair=tok([text,other],padding=True,truncation=False,return_tensors='pt').to(device)
  p=pooled(model,pair)[0:1]; delta=float((a-p).abs().max().cpu()); batch_max=max(batch_max,delta)
  if not torch.isfinite(a).all(): raise RuntimeError('non-finite long-sequence output')
  records.append({'kind':kind,'token_length':length,'anonymous_input_sha256':anon(text),'repeat_max_abs':repeat,'single_vs_padded_batch_max_abs':delta,'outside_pretraining_length_support':length>202})
 if repeat_max>c['long_sequence_gate']['repeat_max_abs_tolerance'] or batch_max>c['long_sequence_gate']['single_vs_padded_batch_max_abs_tolerance']: raise RuntimeError('BLOCKED_LONG_SEQUENCE_MODEL_INTERFACE')
 evidence={'status':'GATE2D2_V2_LONG_SEQUENCE_FORWARD_PASSED','physical_gpu':args.physical_gpu,'selected_samples':len(records),'summary':summary,'repeat_max_abs':repeat_max,'single_vs_padded_batch_max_abs':batch_max,'records':records,'wall_seconds':time.perf_counter()-started,'truncation':False,'smiles_modified':False,'sliding_window':False,'remote_network_during_load':False,'model_parameter_count':sum(p.numel() for p in model.parameters()),'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'optimizer_parameter_count':0,'token_length_correction_sha256':correction_sha,'test_artifacts_accessed':False,'main_parquet_accessed':False,'final673_accessed':False}
 out=ROOT/'logs/gate2d2_v2_long_sequence_smoke.json'; out.write_text(json.dumps(evidence,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({k:evidence[k] for k in ('status','selected_samples','repeat_max_abs','single_vs_padded_batch_max_abs','wall_seconds','model_parameter_count')},indent=2))
if __name__=='__main__': main()
