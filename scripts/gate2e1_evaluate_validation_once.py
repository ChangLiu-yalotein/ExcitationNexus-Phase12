#!/usr/bin/env python3
from __future__ import annotations
import json,math,time
import numpy as np,pandas as pd,torch
from sklearn.metrics import mean_squared_error,r2_score
from gate2e1_common import ROOT,PRIMARY,MultiTaskNet,arrays,config,load_protocol,sha,write_json

def metric(y,p,groups,identities):
 e=np.abs(y-p); g=pd.DataFrame({'e':e,'sq':(y-p)**2,'g':groups}).groupby('g').mean(); ident=pd.DataFrame({'e':e,'id':identities}).groupby('id').e.mean();
 return {'record_mae':float(e.mean()),'record_rmse':float(mean_squared_error(y,p)**.5),'record_r2':float(r2_score(y,p)),'structure_group_macro_mae':float(g.e.mean()),'structure_group_macro_rmse':float(np.sqrt(g.sq.mean())),'identity_macro_mae':float(ident.mean()),'p90_absolute_error':float(np.quantile(e,.9)),'worst_decile_identity_mae':float(ident.nlargest(max(1,math.ceil(len(ident)*.1))).mean()),'records':len(y),'groups':len(g),'identities':len(ident),'normalized_identity_mae':float(ident.mean()/np.subtract(*np.quantile(y,[.75,.25])))}
def cluster_bootstrap_diff(y,pa,pb,clusters,reps,seed):
 d=pd.DataFrame({'ea':np.abs(y-pa),'eb':np.abs(y-pb),'c':clusters}).groupby('c').mean(); vals=(d.ea-d.eb).to_numpy(); keys=np.array(sorted(d.index.astype(str))); d=d.loc[keys];vals=(d.ea-d.eb).to_numpy();rng=np.random.default_rng(seed); out=np.empty(reps)
 for i in range(reps):out[i]=rng.choice(vals,size=len(vals),replace=True).mean()
 return {'point':float(vals.mean()),'ci95':[float(np.quantile(out,.025)),float(np.quantile(out,.975))],'clusters':len(vals),'replicates':reps}
def infer(checkpoint,frame,desc,bits,device):
 c=torch.load(checkpoint,map_location='cpu',weights_only=False);model=MultiTaskNet(c['arm'],c['tasks']);model.load_state_dict(c['state_dict']);model.to(device).eval();x,_,_=arrays(frame,desc,bits,c['feature_stats'],c['target_stats'],c['tasks']);out={t:[] for t in c['tasks']}
 with torch.inference_mode():
  for start in range(0,len(x),512):
   pred=model(torch.from_numpy(x[start:start+512]).to(device))
   for t,v in pred.items():out[t].append(v.cpu().numpy())
 for t in out:out[t]=np.concatenate(out[t])*c['target_stats'][t]['std']+c['target_stats'][t]['mean']
 return out
def main():
 c=config();unlock=ROOT/'data_registry/gate2e1_validation_unlock_v1.json'
 if unlock.exists():raise RuntimeError('official validation evaluator is locked after first invocation')
 registry=json.loads((ROOT/'data_registry/gate2e1_model_registry.json').read_text())
 if registry['model_count']!=18 or registry['official_validation_accessed']:raise RuntimeError('models not frozen')
 for protocol in c['protocols']:
  for arm in c['arms']:
   for seed in c['training']['seeds']:
    item=registry['models'][protocol][arm][str(seed)]
    if sha(item['model_path'])!=item['model_sha256']:raise RuntimeError('model hash mismatch')
 write_json(unlock,{'status':'OFFICIAL_VALIDATION_UNLOCKED_ONCE','model_registry_sha256':sha('data_registry/gate2e1_model_registry.json'),'models':18,'created_before_inference':True,'consumed':False,'test_access':False,'final673_access':False})
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); all_metrics={}; comparisons={}; auxiliary={}; local=ROOT/c['local_root']/'validation_once';local.mkdir(parents=True,exist_ok=False)
 for protocol,spec in c['protocols'].items():
  full,desc,bits=load_protocol(protocol,True);frame=full[full.partition.eq('val')].copy().sort_values('molecule_id').reset_index(drop=True); y=frame[PRIMARY].to_numpy(float); group=frame.structure_group_id_v1.to_numpy();identity=frame[spec['validation_unit']].to_numpy();preds={}; all_metrics[protocol]={}; auxiliary[protocol]={}
  for arm in c['arms']:
   preds[arm]={};seed_metrics=[]
   for seed in c['training']['seeds']:
    item=registry['models'][protocol][arm][str(seed)];out=infer(ROOT/item['model_path'],frame,desc,bits,device);preds[arm][seed]=out;seed_metrics.append(metric(y,out[PRIMARY],group,identity))
   ensemble={t:np.mean([preds[arm][s][t] for s in c['training']['seeds']],axis=0) for t in preds[arm][c['training']['seeds'][0]]};preds[arm]['ensemble']=ensemble
   em=metric(y,ensemble[PRIMARY],group,identity); em['seed_identity_mae_mean']=float(np.mean([z['identity_macro_mae'] for z in seed_metrics]));em['seed_identity_mae_sd']=float(np.std([z['identity_macro_mae'] for z in seed_metrics],ddof=1));all_metrics[protocol][arm]={'seeds':seed_metrics,'ensemble':em}
   for t,p in ensemble.items():
    valid=frame[t].notna().to_numpy();
    if t!=PRIMARY: auxiliary[protocol].setdefault(arm,{})[t]={'mae':float(np.abs(frame.loc[valid,t].to_numpy(float)-p[valid]).mean()),'valid':int(valid.sum())}
  xgb=pd.read_csv(ROOT/spec['xgb_validation_predictions']);
  if sha(spec['xgb_validation_predictions'])!=spec['xgb_validation_sha256'] or set(xgb.molecule_id)!=set(frame.molecule_id):raise RuntimeError('XGB reference mismatch')
  xgb=frame[['molecule_id']].merge(xgb,on='molecule_id',validate='one_to_one').prediction.to_numpy(float);all_metrics[protocol]['XGBoost_C0']={'ensemble':metric(y,xgb,group,identity)}
  reps=c['validation']['bootstrap_replicates'];seed=c['validation']['bootstrap_seed'];comparisons[protocol]={'M11_minus_S0':cluster_bootstrap_diff(y,preds['M11']['ensemble'][PRIMARY],preds['S0']['ensemble'][PRIMARY],identity,reps,seed),'M11_minus_XGBoost':cluster_bootstrap_diff(y,preds['M11']['ensemble'][PRIMARY],xgb,identity,reps,seed),'M15_minus_M11':cluster_bootstrap_diff(y,preds['M15']['ensemble'][PRIMARY],preds['M11']['ensemble'][PRIMARY],identity,reps,seed),'improvement_fraction_M11_vs_S0':float((np.abs(y-preds['M11']['ensemble'][PRIMARY])<np.abs(y-preds['S0']['ensemble'][PRIMARY])).mean())}
  pd.DataFrame({'molecule_id':frame.molecule_id,'y_primary':y,**{f'{arm}_seed{s}':preds[arm][s][PRIMARY] for arm in c['arms'] for s in c['training']['seeds']},**{f'{arm}_ensemble':preds[arm]['ensemble'][PRIMARY] for arm in c['arms']},'xgb_c0':xgb}).to_parquet(local/f'{protocol}_primary_predictions.parquet',index=False)
 acc=comparisons['acceptor_cold'];iid=comparisons['iid'];d=c['decisions'];m11_signal=acc['M11_minus_S0']['point']<=d['acceptor_M11_minus_S0_max'] and acc['M11_minus_S0']['ci95'][1]<0
 if m11_signal and acc['M11_minus_XGBoost']['point']<=d['acceptor_M11_minus_XGB_max'] and acc['M11_minus_XGBoost']['ci95'][1]<0 and iid['M11_minus_XGBoost']['ci95'][1]<=d['iid_M11_minus_XGB_CI_upper_max']:decision='PHYSICS_MULTITASK_ADMITTED'
 elif m11_signal:decision='MULTITASK_SUPERVISION_SIGNAL_ONLY'
 elif acc['M11_minus_S0']['point']<0 and acc['M11_minus_S0']['ci95'][1]>=0:decision='MULTITASK_SIGNAL_INCONCLUSIVE'
 else:decision='MULTITASK_NOT_ADMITTED'
 masked=acc['M15_minus_M11']; masked_signal=m11_signal and masked['point']<=d['masked_M15_minus_M11_max'] and masked['ci95'][1]<0 and iid['M15_minus_M11']['ci95'][1]<=d['masked_iid_CI_upper_max'];masked_decision='MASKED_FRAGMENT_AUXILIARY_ADDS_VALUE' if masked_signal else ('MASKED_FRAGMENT_SIGNAL_INCONCLUSIVE' if masked['point']<0 and masked['ci95'][1]>=0 else 'MASKED_FRAGMENT_AUXILIARY_NOT_ADMITTED')
 payload={'status':'GATE2E1_VALIDATION_EVALUATED_ONCE','decision':decision,'masked_decision':masked_decision,'metrics':all_metrics,'comparisons':comparisons,'auxiliary':auxiliary,'model_registry_sha256':sha('data_registry/gate2e1_model_registry.json'),'test_accessed':False,'final673_accessed':False};write_json('logs/gate2e1_validation_metrics.json',payload);write_json(unlock,{**json.loads(unlock.read_text()),'consumed':True,'completed':True,'metrics_sha256_pending_finalize':True});print(json.dumps({'decision':decision,'masked':masked_decision,'comparisons':comparisons},indent=2))
if __name__=='__main__':main()
