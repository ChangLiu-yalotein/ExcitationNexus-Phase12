#!/usr/bin/env python3
from __future__ import annotations
import json,time
import numpy as np,pandas as pd,xgboost as xgb
from gate2e2a_common import ROOT,PRIMARY,config,descriptor_columns,load_train,sha,write_json
def main():
 c=config();summ=[]
 for protocol in c['protocols']:
  frame,desc,bits=load_train(protocol);reg=json.loads((ROOT/'data_registry/gate2e2a_fold_registry.json').read_text())['protocols'][protocol];frame=frame.merge(pd.read_parquet(ROOT/reg['outer_path']),on='molecule_id',validate='one_to_one');unit=c['protocols'][protocol]['unit']
  for fold in range(5):
   out=ROOT/c['local_root']/'xgb'/protocol/f'fold{fold}';out.mkdir(parents=True,exist_ok=False);tr=frame[~frame.outer_fold.eq(fold)].copy();te=frame[frame.outer_fold.eq(fold)].copy();w=tr.group_weight.to_numpy(float);mu=np.sum(tr[desc].to_numpy(float)*w[:,None],axis=0)/w.sum();sd=np.sqrt(np.sum((tr[desc].to_numpy(float)-mu)**2*w[:,None],axis=0)/w.sum());sd=np.where(sd>1e-12,sd,1.)
   def X(z):return np.concatenate([(z[desc].to_numpy(np.float32)-mu)/sd,z[bits].to_numpy(np.float32)],axis=1)
   model=xgb.XGBRegressor(**c['xgboost']);start=time.time();model.fit(X(tr),tr[PRIMARY],sample_weight=tr.group_weight);p=model.predict(X(te));pred=pd.DataFrame({'molecule_id':te.molecule_id,'y':te[PRIMARY],'prediction':p,'outer_fold':fold,unit:te[unit].astype(str),'structure_group_id_v1':te.structure_group_id_v1.astype(str)});model.save_model(out/'model.ubj');pred.to_parquet(out/'oof.parquet',index=False);item={'protocol':protocol,'fold':fold,'records':len(te),'model_sha256':sha(out/'model.ubj'),'oof_sha256':sha(out/'oof.parquet'),'wall_seconds':time.time()-start};write_json(out/'summary.json',item);summ.append(item);print(item,flush=True)
 write_json('logs/gate2e2a_xgb_registry.json',{'runs':summ,'count':len(summ),'official_validation_accessed':False,'test_accessed':False})
if __name__=='__main__':main()
