#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy as np,torch
from gate2e1_common import ROOT,PRIMARY,MultiTaskNet,arrays,config,load_protocol,sha,write_json
def flat_grad(loss,params,retain=True):
 g=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=False);return torch.cat([x.reshape(-1) for x in g])
def cosine(a,b):return float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)+1e-20))
def main():
 c=config();reg=json.loads((ROOT/'data_registry/gate2e1_model_registry.json').read_text());device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');result={}
 for protocol in c['protocols']:
  frame,desc,bits=load_protocol(protocol,False);frame=frame.sort_values('molecule_id').iloc[:256].copy();result[protocol]={}
  for arm in ('M11','M15'):
   item=reg['models'][protocol][arm]['42'];ck=torch.load(ROOT/item['model_path'],map_location='cpu',weights_only=False);model=MultiTaskNet(arm,ck['tasks']);model.load_state_dict(ck['state_dict']);model.to(device).eval();x,y,w=arrays(frame,desc,bits,ck['feature_stats'],ck['target_stats'],ck['tasks']);x=torch.from_numpy(x).to(device);y=torch.from_numpy(y).to(device);w=torch.from_numpy(w).to(device);out=model(x);params=list(model.trunk.parameters());grads={}
   for j,t in enumerate(ck['tasks']):
    mask=torch.isfinite(y[:,j]);loss=(w[mask]*(out[t][mask]-y[mask,j]).abs()).sum()/w[mask].sum();grads[t]=flat_grad(loss,params)
   gp=grads[PRIMARY];secondary=ck['tasks'][1:12];masked=ck['tasks'][12:]
   gs=sum(grads[t] for t in secondary)/len(secondary);gm=sum((grads[t] for t in masked),torch.zeros_like(gp))/len(masked) if masked else None
   cosines={t:cosine(gp,grads[t]) for t in ck['tasks'][1:]};result[protocol][arm]={'batch_records':len(frame),'seed':42,'task_cosines':cosines,'primary_vs_aggregate_secondary':cosine(gp,gs),'primary_vs_aggregate_masked':cosine(gp,gm) if gm is not None else None,'primary_norm':float(torch.linalg.vector_norm(gp)),'aggregate_secondary_norm_ratio':float(torch.linalg.vector_norm(gs)/torch.linalg.vector_norm(gp)),'aggregate_masked_norm_ratio':float(torch.linalg.vector_norm(gm)/torch.linalg.vector_norm(gp)) if gm is not None else None,'negative_cosine_task_fraction':float(np.mean([v<0 for v in cosines.values()])),'parameters_updated':False}
 write_json('logs/gate2e1_gradient_metrics.json',{'protocols':result,'fixed_batch_rule':'first 256 train molecule_ids sorted','model_seed':42,'parameters_updated':False,'weights_or_tasks_changed':False,'test_accessed':False});print(json.dumps(result,indent=2))
if __name__=='__main__':main()
