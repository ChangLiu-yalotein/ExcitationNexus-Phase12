#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import torch
from gate2d2_v2_model import ROOT,anon,load_model,pooled,sha,unique_inputs

OUT=ROOT/'runs/gate2d2_frozen_molformer/v2_embeddings/frozen_embeddings_v2.npz'
REG=ROOT/'data_registry/gate2d2_v2_embedding_registry.json'

def ah(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def sh(xs): return hashlib.sha256(('\n'.join(xs)+'\n').encode()).hexdigest()
def embed(values,tok,model,device,batch_size):
 rows=[]
 for s in values: rows.append((len(tok(s,add_special_tokens=True,truncation=False)['input_ids']),s))
 order=sorted(range(len(rows)),key=lambda i:(rows[i][0],rows[i][1])); result=np.empty((len(values),model.config.hidden_size),np.float32)
 for start in range(0,len(order),batch_size):
  idx=order[start:start+batch_size]; batch=tok([values[i] for i in idx],padding=True,truncation=False,return_tensors='pt').to(device); out=pooled(model,batch).float().cpu().numpy(); result[idx]=out
 return result,np.array([x[0] for x in rows],np.int16)
def collisions(x): return int(len(x)-len({np.ascontiguousarray(row).tobytes() for row in x}))
def diagnostic(raw,proj,lengths,seed):
 rng=np.random.default_rng(seed); n=len(raw); k=min(10000,max(1,n*(n-1)//2)); i=rng.integers(0,n,k); j=rng.integers(0,n,k); same=i==j; j[same]=(j[same]+1)%n
 rd=np.linalg.norm(raw[i]-raw[j],axis=1); pd=np.linalg.norm(proj[i]-proj[j],axis=1); valid=rd>1e-12; rel=np.abs(pd[valid]-rd[valid])/rd[valid]
 rn=np.linalg.norm(raw,axis=1); pn=np.linalg.norm(proj,axis=1); rcos=1-np.sum(raw[i]*raw[j],axis=1)/np.maximum(rn[i]*rn[j],1e-12); pcos=1-np.sum(proj[i]*proj[j],axis=1)/np.maximum(pn[i]*pn[j],1e-12)
 sample=proj[:min(len(proj),2048)].astype(np.float64); rank=int(np.linalg.matrix_rank(sample-sample.mean(0)))
 def stats(mask):
  vals=pn[mask]; return {'n':int(mask.sum()),'mean':float(vals.mean()),'median':float(np.median(vals)),'p90':float(np.quantile(vals,.9))} if mask.any() else {'n':0}
 return {'cosine_distance_spearman':float(spearmanr(rcos,pcos).statistic),'pairwise_relative_error_median':float(np.median(rel)),'pairwise_relative_error_p90':float(np.quantile(rel,.9)),'projected_norm':{'mean':float(pn.mean()),'median':float(np.median(pn)),'p90':float(np.quantile(pn,.9))},'empirical_rank':rank,'rank_sample_records':min(len(proj),2048),'length_support':{'at_most_202':stats(lengths<=202),'over_202':stats(lengths>202)}}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--physical-gpu',type=int,required=True); ap.add_argument('--batch-size',type=int,default=32); args=ap.parse_args()
 smoke=json.loads((ROOT/'logs/gate2d2_v2_long_sequence_smoke.json').read_text())
 if smoke['status']!='GATE2D2_V2_LONG_SEQUENCE_FORWARD_PASSED': raise RuntimeError('long-sequence gate not passed')
 if REG.exists() or OUT.exists(): raise RuntimeError('v2 embeddings already exist; refuse overwrite')
 device=torch.device('cuda:0'); c,tok,model=load_model(device); values=unique_inputs(); matrices=np.load(ROOT/'data_registry/gate2d2_v2_fixed_projection_matrices.npz',allow_pickle=False); rf=matrices['R_full']; rc=matrices['R_component']
 started=time.perf_counter(); raw={}; lengths={}
 for kind in ('full','donor','acceptor'): raw[kind],lengths[kind]=embed(values[kind],tok,model,device,args.batch_size)
 projected={'full':raw['full']@rf,'donor':raw['donor']@rc,'acceptor':raw['acceptor']@rc}
 for kind in raw:
  if not np.isfinite(raw[kind]).all() or not np.isfinite(projected[kind]).all() or np.any(np.linalg.norm(projected[kind],axis=1)==0): raise RuntimeError('BLOCKED_EMBEDDING_INTEGRITY')
 # batch-size sensitivity on fixed boundary subset
 batch_delta=0.0
 for kind in raw:
  idx=sorted(set([0,len(values[kind])//2,len(values[kind])-1])); subset=[values[kind][i] for i in idx]; check,_=embed(subset,tok,model,device,1)
  for pos,orig in enumerate(idx): batch_delta=max(batch_delta,float(np.max(np.abs(check[pos]-raw[kind][orig]))))
 if batch_delta>1e-5: raise RuntimeError('BLOCKED_EMBEDDING_INTEGRITY: batch size mismatch')
 OUT.parent.mkdir(parents=True,exist_ok=True); np.savez(OUT,full_strings=np.asarray(values['full']),donor_strings=np.asarray(values['donor']),acceptor_strings=np.asarray(values['acceptor']),full_raw=raw['full'],donor_raw=raw['donor'],acceptor_raw=raw['acceptor'],full_projected=projected['full'],donor_projected=projected['donor'],acceptor_projected=projected['acceptor'],full_lengths=lengths['full'],donor_lengths=lengths['donor'],acceptor_lengths=lengths['acceptor'])
 categories={}
 for n,kind in enumerate(('full','donor','acceptor')):
  categories[kind]={'identities':len(values[kind]),'identity_set_sha256':sh(values[kind]),'raw_shape':list(raw[kind].shape),'raw_content_sha256':ah(raw[kind]),'projected_shape':list(projected[kind].shape),'projected_content_sha256':ah(projected[kind]),'raw_exact_collision_count':collisions(raw[kind]),'projected_exact_collision_count':collisions(projected[kind]),'max_token_length':int(lengths[kind].max()),'over_202_count':int((lengths[kind]>202).sum()),'diagnostic':diagnostic(raw[kind],projected[kind],lengths[kind],20260720+n)}
 reg={'status':'GATE2D2_V2_EMBEDDINGS_FROZEN','model_revision':c['model']['revision'],'weight_sha256':sha(ROOT/'runs/gate2d2_frozen_molformer/model_asset_audit/model.safetensors'),'physical_gpu':args.physical_gpu,'batch_size':args.batch_size,'raw_embedding_dtype':'float32','projection_dtype':'float32','categories':categories,'batch_size_sample_max_abs':batch_delta,'artifact_path':str(OUT.relative_to(ROOT)),'artifact_sha256':sha(OUT),'wall_seconds':time.perf_counter()-started,'encoder_eval':not model.training,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'optimizer_parameters':0,'test_artifacts_accessed':False,'main_parquet_accessed':False,'final673_accessed':False}
 REG.write_text(json.dumps(reg,indent=2,sort_keys=True,allow_nan=False)+'\n'); print(json.dumps({'status':reg['status'],'wall_seconds':reg['wall_seconds'],'batch_size_sample_max_abs':batch_delta,'categories':{k:{'identities':v['identities'],'max_token_length':v['max_token_length'],'over_202_count':v['over_202_count']} for k,v in categories.items()}},indent=2))
if __name__=='__main__': main()
