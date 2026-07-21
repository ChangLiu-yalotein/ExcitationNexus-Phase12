#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,TensorDataset
from gate2e1_common import ROOT,PRIMARY,MultiTaskNet,arm_tasks,arrays,config,feature_stats,group_macro_mae,load_protocol,masked_loss,set_seed,sha,task_weights,weighted_stats,write_json

def loader(x,y,w,batch,seed,shuffle):
 g=torch.Generator();g.manual_seed(seed); ds=TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(w),torch.arange(len(x)))
 return DataLoader(ds,batch_size=batch,shuffle=shuffle,generator=g,drop_last=False,num_workers=0)
def evaluate(model,dl,device,tstats,groups):
 model.eval(); pred=[];true=[];idx=[]
 with torch.inference_mode():
  for x,y,_,i in dl: pred.append(model(x.to(device))[PRIMARY].cpu().numpy());true.append(y[:,0].numpy());idx.append(i.numpy())
 p=np.concatenate(pred); y=np.concatenate(true); ii=np.concatenate(idx); order=np.argsort(ii); p=p[order]*tstats[PRIMARY]['std']+tstats[PRIMARY]['mean']; y=y[order]*tstats[PRIMARY]['std']+tstats[PRIMARY]['mean']
 return group_macro_mae(y,np.asarray(p),groups),p
def run(protocol,arm,seed,physical_gpu):
 c=config(); out=ROOT/c['local_root']/ 'inner_selection'/protocol/arm/f'seed{seed}'; out.mkdir(parents=True,exist_ok=False)
 frame,desc,bits=load_protocol(protocol,False); split_spec=json.loads((ROOT/'data_registry/gate2e1_inner_split_registry.json').read_text())['protocols'][protocol]
 if sha(split_spec['path'])!=split_spec['sha256']: raise RuntimeError('inner split hash')
 split=pd.read_parquet(ROOT/split_spec['path']); frame=frame.merge(split[['molecule_id','inner_partition']],on='molecule_id',validate='one_to_one')
 fit=frame[frame.inner_partition.eq('inner_fit')].copy(); check=frame[frame.inner_partition.eq('inner_checkpoint')].copy(); tasks=arm_tasks(arm)
 fs=feature_stats(fit,desc); ts=weighted_stats(fit,tasks); xf,yf,wf=arrays(fit,desc,bits,fs,ts,tasks); xc,yc,wc=arrays(check,desc,bits,fs,ts,tasks)
 set_seed(seed); model=MultiTaskNet(arm,tasks); initial_shapes={n:list(p.shape) for n,p in model.named_parameters() if n.startswith('trunk.') or n.startswith('primary_head.')}
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay'])
 train_dl=loader(xf,yf,wf,c['training']['batch_size'],seed,True); check_dl=loader(xc,yc,wc,c['training']['batch_size'],seed,False)
 best=float('inf');best_epoch=0;wait=0;curves=[];start=time.time();peak=0
 for epoch in range(1,c['training']['max_epochs']+1):
  model.train(); losses=[]
  for x,y,w,_ in train_dl:
   x,y,w=x.to(device),y.to(device),w.to(device);opt.zero_grad(set_to_none=True);loss,_=masked_loss(model(x),y,w,tasks,task_weights(arm));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['gradient_clip_norm']);opt.step();losses.append(float(loss.detach()))
  mae,_=evaluate(model,check_dl,device,ts,check.structure_group_id_v1.to_numpy())
  curves.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'checkpoint_primary_group_macro_mae_eV':mae})
  if mae < best-c['training']['min_delta_eV']:
   best,best_epoch,wait=mae,epoch,0;torch.save({'state_dict':model.state_dict(),'arm':arm,'protocol':protocol,'seed':seed,'epoch':epoch,'feature_stats':fs,'target_stats':ts,'tasks':tasks,'primary_shapes':initial_shapes},out/'best_checkpoint.pt')
  else: wait+=1
  if device.type=='cuda':peak=max(peak,torch.cuda.max_memory_allocated())
  if epoch>=c['training']['minimum_epochs'] and wait>=c['training']['patience']:break
 summary={'stage':'inner_selection','protocol':protocol,'arm':arm,'seed':seed,'physical_gpu':physical_gpu,'best_epoch':best_epoch,'best_primary_group_macro_mae_eV':best,'epochs_run':epoch,'wall_seconds':time.time()-start,'peak_gpu_bytes':peak,'checkpoint_sha256':sha(out/'best_checkpoint.pt'),'official_validation_accessed':False,'test_accessed':False,'finite':bool(np.isfinite(best)),'curve':curves}
 write_json(out/'summary.json',summary);print(json.dumps({k:summary[k] for k in ['protocol','arm','seed','best_epoch','best_primary_group_macro_mae_eV','wall_seconds']},indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--arm',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--physical-gpu',required=True);a=ap.parse_args();run(a.protocol,a.arm,a.seed,a.physical_gpu)
if __name__=='__main__':main()
