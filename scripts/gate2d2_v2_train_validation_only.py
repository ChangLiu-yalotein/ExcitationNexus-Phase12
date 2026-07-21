#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from gate2d1_common import PROTOCOLS,ROOT,TARGET,load_labels,metric_frame,read_json,resolve,sha,weighted_median,write_json

ARMS=('B_MF_Full_RP512','C_MF_Role_RP512')
RUN=ROOT/'runs/gate2d2_frozen_molformer/v2_models'
REG=ROOT/'data_registry/gate2d2_v2_model_registry.json'

def verify():
 c=read_json('configs/gate2d2_frozen_molformer_admission_v2.json'); lock=read_json('data_registry/gate2d2_v2_preregistration_lock.json'); emb=read_json('data_registry/gate2d2_v2_embedding_registry.json')
 if sha('configs/gate2d2_frozen_molformer_admission_v2.json')!=lock['config_sha256'] or emb['status']!='GATE2D2_V2_EMBEDDINGS_FROZEN': raise RuntimeError('v2 lock/embedding mismatch')
 if sha(emb['artifact_path'])!=emb['artifact_sha256']: raise RuntimeError('embedding artifact hash mismatch')
 return c,read_json('configs/gate2d1_role_aware_2d_v1.json'),emb
def descriptor_impute(x,w):
 med=np.array([weighted_median(x[:,j][np.isfinite(x[:,j])],w[np.isfinite(x[:,j])]) for j in range(x.shape[1])]); return med
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--physical-gpu',type=int,required=True); args=ap.parse_args(); c,d1,ereg=verify()
 if REG.exists() or RUN.exists(): raise RuntimeError('v2 models already exist; refuse overwrite')
 z=np.load(resolve(ereg['artifact_path']),allow_pickle=False); structs=pd.read_parquet(ROOT/'manifests/new15016_structure_groups_v1.parquet',columns=['molecule_id','canonical_structure_smiles_v1']); comps=pd.read_csv(ROOT/'manifests/component_identity_v1.csv',usecols=['molecule_id','donor_canonical_structure_smiles_v1','acceptor_canonical_structure_smiles_v1']); rows=structs.merge(comps,on='molecule_id',validate='one_to_one').sort_values('molecule_id',kind='mergesort')
 dcache=np.load(ROOT/'runs/gate2d1_role_aware_2d/features/role_aware_features_v1.npz',allow_pickle=False); desc=dcache['descriptors'].astype(np.float32)
 if desc.shape!=(len(rows),20) or not rows.molecule_id.is_monotonic_increasing: raise RuntimeError('D1 descriptor cache deterministic row-order contract failed')
 maps={kind:{str(s):i for i,s in enumerate(z[f'{kind}_strings'])} for kind in ('full','donor','acceptor')}
 full=z['full_projected'][[maps['full'][x] for x in rows.canonical_structure_smiles_v1.astype(str)]]; donor=z['donor_projected'][[maps['donor'][x] for x in rows.donor_canonical_structure_smiles_v1.astype(str)]]; acceptor=z['acceptor_projected'][[maps['acceptor'][x] for x in rows.acceptor_canonical_structure_smiles_v1.astype(str)]]
 lengths=z['full_lengths'][[maps['full'][x] for x in rows.canonical_structure_smiles_v1.astype(str)]]; donor_lengths=z['donor_lengths'][[maps['donor'][x] for x in rows.donor_canonical_structure_smiles_v1.astype(str)]]; acceptor_lengths=z['acceptor_lengths'][[maps['acceptor'][x] for x in rows.acceptor_canonical_structure_smiles_v1.astype(str)]]; matrices={'B_MF_Full_RP512':np.concatenate([desc,full],1),'C_MF_Role_RP512':np.concatenate([desc,donor,acceptor],1)}; index={x:i for i,x in enumerate(rows.molecule_id.astype(str))}
 registry={'status':'GATE2D2_V2_VALIDATION_MODELS_FROZEN','new_models':0,'physical_gpu':args.physical_gpu,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'protocols':{},'models_shared_between_protocols':False,'test_artifacts_accessed':False,'main_parquet_accessed':False,'final673_accessed':False,'scaler_used':False}
 for name in PROTOCOLS:
  spec=d1['protocols'][name]; manifest=pd.read_csv(ROOT/spec['manifest']); train_y,val_y=load_labels(d1,name,manifest); cols=['molecule_id','structure_group_id_v1','donor_structure_group_id_v1','acceptor_structure_group_id_v1','pair_group_id_v1','full_scaffold_group_id_v1','group_weight']; train=manifest.loc[manifest.partition.eq('train'),cols].merge(train_y,on='molecule_id',validate='one_to_one'); val=manifest.loc[manifest.partition.eq('val'),cols].merge(val_y,on='molecule_id',validate='one_to_one'); ti=np.array([index[x] for x in train.molecule_id.astype(str)]); vi=np.array([index[x] for x in val.molecule_id.astype(str)]); w=train.group_weight.to_numpy(float); cluster=d1['protocol_clusters'][name]
  arm_a=pd.read_csv(ROOT/spec['arm_a_validation']); arm_a=val[[*cols,TARGET]].merge(arm_a,on='molecule_id',validate='one_to_one'); protocol={'train_records':len(train),'validation_records':len(val),'manifest_sha256':spec['sha256'],'arms':{'A_C0_512_reference':{'validation':metric_frame(arm_a,arm_a.prediction.to_numpy(),cluster),'prediction_sha256':spec['arm_a_sha256']}},'train_weight_sum':float(w.sum())}; paired=val[[*cols,TARGET]].copy(); paired['token_length']=lengths[vi]; paired['donor_token_length']=donor_lengths[vi]; paired['acceptor_token_length']=acceptor_lengths[vi]; paired['A_C0_512_reference']=arm_a.set_index('molecule_id').loc[paired.molecule_id,'prediction'].to_numpy()
  for arm in ARMS:
   out=RUN/name/arm; out.mkdir(parents=True); xtr=matrices[arm][ti].astype(np.float32); xval=matrices[arm][vi].astype(np.float32); med=descriptor_impute(xtr[:,:20].astype(float),w); xtr[:,:20]=np.where(np.isfinite(xtr[:,:20]),xtr[:,:20],med); xval[:,:20]=np.where(np.isfinite(xval[:,:20]),xval[:,:20],med); np.save(out/'descriptor_medians.npy',med)
   params=dict(c['xgboost']); params.pop('row_subsampling'); params.pop('feature_subsampling'); params.pop('early_stopping'); model=XGBRegressor(**params); started=time.perf_counter(); model.fit(xtr,train[TARGET].to_numpy(float),sample_weight=w); seconds=time.perf_counter()-started; pred=model.predict(xval); model.save_model(out/'model.json'); pd.DataFrame({'molecule_id':val.molecule_id,'prediction':pred}).sort_values('molecule_id').to_parquet(out/'validation_predictions.parquet',index=False); paired[arm]=pred
   protocol['arms'][arm]={'validation':metric_frame(val,pred,cluster),'model_path':str((out/'model.json').relative_to(ROOT)),'model_sha256':sha(out/'model.json'),'prediction_path':str((out/'validation_predictions.parquet').relative_to(ROOT)),'prediction_sha256':sha(out/'validation_predictions.parquet'),'descriptor_medians_sha256':sha(out/'descriptor_medians.npy'),'feature_columns':532,'training_wall_seconds':seconds,'fit_partition':'train','inference_partition':'val','test_accessed':False}; registry['new_models']+=1
  path=RUN/name/'validation_paired.parquet'; paired.sort_values('molecule_id').to_parquet(path,index=False); protocol['paired_validation_path']=str(path.relative_to(ROOT)); protocol['paired_validation_sha256']=sha(path); registry['protocols'][name]=protocol
 write_json(REG,registry); print(json.dumps({'status':registry['status'],'new_models':registry['new_models'],'protocols':{n:{a:round(x['validation'].get('identity_macro_mae',x['validation'].get('acceptor_identity_macro_mae',0)),8) for a,x in p['arms'].items()} for n,p in registry['protocols'].items()}},indent=2))
if __name__=='__main__': main()
