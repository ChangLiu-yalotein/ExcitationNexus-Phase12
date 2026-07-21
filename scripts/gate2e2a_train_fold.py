#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,TensorDataset
from gate2e2a_common import ROOT,PRIMARY,MultiTaskNet,arrays,config,feature_stats,group_macro_mae,load_train,masked_loss,set_seed,sha,task_weights,tasks,weighted_stats,write_json

def loader(x,y,w,batch,seed,shuffle):
 g=torch.Generator().manual_seed(seed);return DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(w),torch.arange(len(x))),batch_size=batch,shuffle=shuffle,generator=g,num_workers=0,drop_last=False)
def evaluate(model,dl,device,ts,groups):
 model.eval();pred=[];true=[];idx=[]
 with torch.inference_mode():
  for x,y,_,i in dl:pred.append(model(x.to(device))[PRIMARY].cpu().numpy());true.append(y[:,0].numpy());idx.append(i.numpy())
 ii=np.concatenate(idx);order=np.argsort(ii);p=np.concatenate(pred)[order]*ts[PRIMARY]['std']+ts[PRIMARY]['mean'];y=np.concatenate(true)[order]*ts[PRIMARY]['std']+ts[PRIMARY]['mean']
 return group_macro_mae(y,p,groups),p,y
def train_stage(frame,check,desc,bits,arm,seed,epochs=None):
 c=config();tt=tasks(arm);fs=feature_stats(frame,desc);ts=weighted_stats(frame,tt);xf,yf,wf=arrays(frame,desc,bits,fs,ts,tt);xc,yc,wc=arrays(check,desc,bits,fs,ts,tt)
 set_seed(seed);model=MultiTaskNet(arm,tt).cuda();opt=torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay']);dl=loader(xf,yf,wf,c['training']['batch_size'],seed,True);cdl=loader(xc,yc,wc,c['training']['batch_size'],seed,False)
 best=float('inf');best_epoch=0;wait=0;curves=[];best_state=None;limit=epochs or c['training']['max_epochs']
 for epoch in range(1,limit+1):
  model.train();losses=[]
  for x,y,w,_ in dl:
   x,y,w=x.cuda(),y.cuda(),w.cuda();opt.zero_grad(set_to_none=True);loss,_=masked_loss(model(x),y,w,tt,task_weights(arm));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['gradient_clip_norm']);opt.step();losses.append(float(loss.detach()))
  if epochs is None:
   mae,_,_=evaluate(model,cdl,torch.device('cuda:0'),ts,check.structure_group_id_v1.to_numpy());curves.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'checkpoint_mae_eV':mae})
   if mae<best-c['training']['min_delta_eV']:best,best_epoch,wait=mae,epoch,0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   else:wait+=1
   if epoch>=c['training']['minimum_epochs'] and wait>=c['training']['patience']:break
  else:curves.append({'epoch':epoch,'train_loss':float(np.mean(losses))})
 if epochs is None:return best_epoch,best,best_state,fs,ts,curves
 return model,fs,ts,curves
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--fold',type=int,required=True);ap.add_argument('--arm',choices=['S0','M11'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--physical-gpu',required=True);a=ap.parse_args();c=config();start=time.time();out=ROOT/c['local_root']/ 'neural'/a.protocol/f'fold{a.fold}'/a.arm/f'seed{a.seed}'
 if out.exists() and not (out/'summary.json').exists():
  abandoned=out.with_name(out.name+'_aborted_scheduler_overlap');i=1
  while abandoned.exists():abandoned=out.with_name(out.name+f'_aborted_scheduler_overlap_{i}');i+=1
  out.rename(abandoned)
 out.mkdir(parents=True,exist_ok=False)
 frame,desc,bits=load_train(a.protocol);reg=json.loads((ROOT/'data_registry/gate2e2a_fold_registry.json').read_text())['protocols'][a.protocol];outer=pd.read_parquet(ROOT/reg['outer_path']);inner=pd.read_parquet(ROOT/reg['inner_path']);frame=frame.merge(outer,on='molecule_id',validate='one_to_one');unit=c['protocols'][a.protocol]['unit'];held=frame[frame.outer_fold.eq(a.fold)].copy();outer_train=frame[~frame.outer_fold.eq(a.fold)].copy();imap=inner[inner.outer_fold.eq(a.fold)].set_index('unit_id').inner_partition.to_dict();outer_train['inner_partition']=outer_train[unit].astype(str).map(imap)
 if outer_train.inner_partition.isna().any() or held.empty:raise RuntimeError('fold binding')
 fit=outer_train[outer_train.inner_partition.eq('inner_fit')].copy();check=outer_train[outer_train.inner_partition.eq('inner_checkpoint')].copy();best_epoch,best,bstate,_,_,curves=train_stage(fit,check,desc,bits,a.arm,a.seed)
 model,fs,ts,refit_curves=train_stage(outer_train,held,desc,bits,a.arm,a.seed,epochs=best_epoch);tt=tasks(a.arm);xh,yh,wh=arrays(held,desc,bits,fs,ts,tt);mae,p,y=evaluate(model,loader(xh,yh,wh,c['training']['batch_size'],a.seed,False),torch.device('cuda:0'),ts,held.structure_group_id_v1.to_numpy())
 ck={'state_dict':model.state_dict(),'protocol':a.protocol,'fold':a.fold,'arm':a.arm,'seed':a.seed,'epoch':best_epoch,'feature_stats':fs,'target_stats':ts,'tasks':tt};torch.save(ck,out/'refit_checkpoint.pt')
 pred=pd.DataFrame({'molecule_id':held.molecule_id,'y':y,'prediction':p,'outer_fold':a.fold,'seed':a.seed,'arm':a.arm,unit:held[unit].astype(str),'structure_group_id_v1':held.structure_group_id_v1.astype(str)});pred.to_parquet(out/'oof.parquet',index=False)
 summary={'protocol':a.protocol,'fold':a.fold,'arm':a.arm,'seed':a.seed,'physical_gpu':a.physical_gpu,'best_epoch':best_epoch,'inner_checkpoint_mae_eV':best,'oof_structure_group_macro_mae_eV':mae,'held_records':len(held),'checkpoint_sha256':sha(out/'refit_checkpoint.pt'),'oof_sha256':sha(out/'oof.parquet'),'wall_seconds':time.time()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'inner_curve':curves,'refit_curve':refit_curves,'official_validation_accessed':False,'test_accessed':False,'finite':bool(np.isfinite(mae))};write_json(out/'summary.json',summary);print(json.dumps({k:summary[k] for k in ['protocol','fold','arm','seed','best_epoch','oof_structure_group_macro_mae_eV','wall_seconds']},indent=2))
if __name__=='__main__':main()
