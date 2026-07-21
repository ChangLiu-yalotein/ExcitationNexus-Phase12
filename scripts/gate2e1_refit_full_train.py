#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from gate2e1_common import ROOT,MultiTaskNet,arm_tasks,arrays,config,feature_stats,load_protocol,masked_loss,set_seed,sha,task_weights,weighted_stats,write_json
def run(protocol,arm,seed,physical_gpu):
 c=config(); inner=ROOT/c['local_root']/'inner_selection'/protocol/arm/f'seed{seed}'/'summary.json'; s=json.loads(inner.read_text()); epochs=s['best_epoch']; out=ROOT/c['local_root']/'full_refit'/protocol/arm/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
 frame,desc,bits=load_protocol(protocol,False);tasks=arm_tasks(arm);fs=feature_stats(frame,desc);ts=weighted_stats(frame,tasks);x,y,w=arrays(frame,desc,bits,fs,ts,tasks)
 set_seed(seed);model=MultiTaskNet(arm,tasks);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay']);g=torch.Generator();g.manual_seed(seed);dl=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(w)),batch_size=c['training']['batch_size'],shuffle=True,generator=g,drop_last=False,num_workers=0)
 start=time.time();peak=0;last=None
 for epoch in range(1,epochs+1):
  model.train();vals=[]
  for xb,yb,wb in dl:
   xb,yb,wb=xb.to(device),yb.to(device),wb.to(device);opt.zero_grad(set_to_none=True);loss,_=masked_loss(model(xb),yb,wb,tasks,task_weights(arm));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['gradient_clip_norm']);opt.step();vals.append(float(loss.detach()))
  last=float(np.mean(vals));
  if device.type=='cuda':peak=max(peak,torch.cuda.max_memory_allocated())
 checkpoint={'state_dict':model.state_dict(),'arm':arm,'protocol':protocol,'seed':seed,'epochs':epochs,'feature_stats':fs,'target_stats':ts,'tasks':tasks,'source_inner_summary_sha256':sha(inner)};torch.save(checkpoint,out/'model.pt')
 summary={'stage':'full_train_refit','protocol':protocol,'arm':arm,'seed':seed,'physical_gpu':physical_gpu,'epochs':epochs,'final_train_loss':last,'wall_seconds':time.time()-start,'peak_gpu_bytes':peak,'model_sha256':sha(out/'model.pt'),'normalization':{'feature':fs,'target':ts},'official_validation_accessed':False,'test_accessed':False,'finite':bool(np.isfinite(last))};write_json(out/'summary.json',summary);print(json.dumps({k:summary[k] for k in ['protocol','arm','seed','epochs','final_train_loss','wall_seconds']},indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--arm',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--physical-gpu',required=True);a=ap.parse_args();run(a.protocol,a.arm,a.seed,a.physical_gpu)
if __name__=='__main__':main()
