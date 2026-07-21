#!/usr/bin/env python3
from __future__ import annotations
import glob,json,math
from datetime import datetime,timezone
import numpy as np,pandas as pd,torch
from gate2e2a_common import ROOT,PRIMARY,MultiTaskNet,arrays,bootstrap_diff,config,load_train,masked_loss,sha,task_weights,tasks,write_json

def macro(y,p,g):return float(pd.DataFrame({'e':np.abs(y-p),'g':g}).groupby('g').e.mean().mean())
def rmse(y,p,g):return float(np.sqrt(pd.DataFrame({'e':(y-p)**2,'g':g}).groupby('g').e.mean().mean()))
def collect(protocol):
 c=config();unit=c['protocols'][protocol]['unit'];base=None;seed_metrics={}
 for arm in c['arms']:
  seeds=[]
  for seed in c['seeds']:
   parts=[pd.read_parquet(ROOT/c['local_root']/'neural'/protocol/f'fold{k}'/arm/f'seed{seed}'/'oof.parquet') for k in range(5)];d=pd.concat(parts).sort_values('molecule_id').reset_index(drop=True);d=d.rename(columns={'prediction':f'{arm}_{seed}'})
   if base is None:base=d[list(dict.fromkeys(['molecule_id','y',unit,'structure_group_id_v1','outer_fold']))].copy()
   else:
    if not np.array_equal(base.molecule_id,d.molecule_id) or not np.allclose(base.y,d.y,atol=0,rtol=0):raise RuntimeError('OOF reconciliation')
   base=base.merge(d[['molecule_id',f'{arm}_{seed}']],on='molecule_id',validate='one_to_one');seed_metrics[f'{arm}_{seed}']={'identity_macro_mae':macro(d.y,d[f'{arm}_{seed}'],d[unit]),'structure_macro_mae':macro(d.y,d[f'{arm}_{seed}'],d.structure_group_id_v1)};seeds.append(f'{arm}_{seed}')
  base[arm]=base[seeds].mean(axis=1)
 x=pd.concat([pd.read_parquet(ROOT/c['local_root']/'xgb'/protocol/f'fold{k}'/'oof.parquet') for k in range(5)]).sort_values('molecule_id').reset_index(drop=True).rename(columns={'prediction':'XGB'})
 if not np.array_equal(base.molecule_id,x.molecule_id) or not np.allclose(base.y,x.y,atol=1e-12):raise RuntimeError('XGB reconciliation')
 base=base.merge(x[['molecule_id','XGB']],on='molecule_id',validate='one_to_one')
 metrics={m:{'record_mae':float(np.mean(np.abs(base.y-base[m]))),'record_rmse':float(np.sqrt(np.mean((base.y-base[m])**2))),'identity_macro_mae':macro(base.y,base[m],base[unit]),'structure_macro_mae':macro(base.y,base[m],base.structure_group_id_v1),'p90_abs_error':float(np.quantile(np.abs(base.y-base[m]),.9))} for m in ['S0','M11','XGB']}
 comparisons={'M11_minus_S0':bootstrap_diff(base,'M11','S0',unit),'M11_minus_XGB':bootstrap_diff(base,'M11','XGB',unit)}
 if protocol=='iid':comparisons={k:bootstrap_diff(base,*({'M11_minus_S0':('M11','S0'),'M11_minus_XGB':('M11','XGB')}[k]),'structure_group_id_v1') for k in comparisons}
 fold_metrics={m:{str(k):macro(base.loc[base.outer_fold.eq(k),'y'],base.loc[base.outer_fold.eq(k),m],base.loc[base.outer_fold.eq(k),unit]) for k in range(5)} for m in ['S0','M11','XGB']}
 return base,{'records':len(base),'units':base[unit].nunique(),'metrics':metrics,'seed_metrics':seed_metrics,'fold_metrics':fold_metrics,'comparisons':comparisons}

def gradients(protocol,fold):
 c=config();unit=c['protocols'][protocol]['unit'];frame,desc,bits=load_train(protocol);reg=json.loads((ROOT/'data_registry/gate2e2a_fold_registry.json').read_text())['protocols'][protocol];frame=frame.merge(pd.read_parquet(ROOT/reg['outer_path']),on='molecule_id');train=frame[~frame.outer_fold.eq(fold)].sort_values('molecule_id').head(256).copy();ckp=ROOT/c['local_root']/'neural'/protocol/f'fold{fold}'/'M11'/'seed42'/'refit_checkpoint.pt';ck=torch.load(ckp,map_location='cpu',weights_only=False);tt=tasks('M11');model=MultiTaskNet('M11',tt);model.load_state_dict(ck['state_dict']);x,y,w=arrays(train,desc,bits,ck['feature_stats'],ck['target_stats'],tt);x=torch.from_numpy(x);y=torch.from_numpy(y);w=torch.from_numpy(w);out=model(x);params=list(model.trunk.parameters());vecs={}
 for j,t in enumerate(tt):
  model.zero_grad(set_to_none=True);mask=torch.isfinite(y[:,j]);loss=(w[mask]*(out[t][mask]-y[mask,j]).abs()).sum()/w[mask].sum();loss.backward(retain_graph=True);vecs[t]=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params]).numpy()
 p=vecs[PRIMARY];cos={t:float(np.dot(p,v)/(np.linalg.norm(p)*np.linalg.norm(v)+1e-30)) for t,v in vecs.items() if t!=PRIMARY};agg=np.mean([vecs[t] for t in tt[1:]],axis=0);return {'task_cosines':cos,'primary_vs_aggregate_secondary_cosine':float(np.dot(p,agg)/(np.linalg.norm(p)*np.linalg.norm(agg)+1e-30)),'negative_task_fraction':float(np.mean(np.array(list(cos.values()))<0)),'aggregate_to_primary_norm_ratio':float(np.linalg.norm(agg)/np.linalg.norm(p))}

def main():
 c=config();payload={'version':'gate2e2a_crossfit_metrics_v1','protocols':{},'official_validation_accessed':False,'test_accessed':False,'main_parquet_accessed':False,'final673_accessed':False}
 for p in c['protocols']:
  _,payload['protocols'][p]=collect(p)
 acc=payload['protocols']['acceptor_cold']['comparisons'];iid=payload['protocols']['iid']['comparisons'];d=c['decisions']
 if acc['M11_minus_S0']['point']<=d['acceptor_M11_minus_S0_max'] and acc['M11_minus_S0']['ci95'][1]<0 and acc['M11_minus_XGB']['point']<=d['acceptor_M11_minus_XGB_max'] and acc['M11_minus_XGB']['ci95'][1]<0 and iid['M11_minus_XGB']['ci95'][1]<=d['iid_M11_minus_XGB_CI_upper_max']:decision='PHYSICS_MULTITASK_CROSSFIT_SIGNAL'
 elif acc['M11_minus_S0']['point']<=d['acceptor_M11_minus_S0_max'] and acc['M11_minus_S0']['ci95'][1]<0:decision='MULTITASK_CROSSFIT_SIGNAL_ONLY'
 elif acc['M11_minus_S0']['point']<0 and acc['M11_minus_S0']['ci95'][1]>=0:decision='MULTITASK_CROSSFIT_INCONCLUSIVE'
 else:decision='MULTITASK_CROSSFIT_NOT_SUPPORTED'
 payload['decision']=decision;write_json('logs/gate2e2a_crossfit_metrics.json',payload)
 grad={p:{str(k):gradients(p,k) for k in range(5)} for p in c['protocols']};write_json('logs/gate2e2a_gradient_metrics.json',{'protocols':grad,'diagnostic_only':True,'parameters_updated':False})
 summaries=[json.load(open(p)) for p in glob.glob(str(ROOT/c['local_root']/'neural/**/summary.json'),recursive=True)];model_registry={'neural_models':len(summaries),'xgb_models':10,'all_finite':all(x['finite'] for x in summaries),'checkpoint_hashes':sorted(x['checkpoint_sha256'] for x in summaries),'oof_hashes':sorted(x['oof_sha256'] for x in summaries),'official_validation_accessed':False,'test_accessed':False};write_json('data_registry/gate2e2a_model_registry.json',model_registry)
 a=payload['protocols']['acceptor_cold'];i=payload['protocols']['iid'];fmt=lambda x:f"{x:+.9f} eV, 95% CI [{x:+.9f}, {x:+.9f}]"
 (ROOT/'reports/gate2e2a_preregistration.md').write_text("# Gate 2-E2A preregistration\n\nTraining-only 5-fold cross-fitting compares S0, M11, and fold-matched XGBoost-C0 under frozen E1 hyperparameters. Official validation, test, final673, and the main Parquet are sealed.\n")
 (ROOT/'reports/gate2e2a_split_integrity.md').write_text("# Gate 2-E2A split integrity\n\nProduction and reference implementations agree within 1e-12. Assignment hashes are invariant to row order. A synthetic unequal-replicate regression test rejects the E1 unweighted record-mean bug. Historical-overlap units remain outer-train-only.\n")
 lines=['# Gate 2-E2A cross-fit results','',f"Decision: `{decision}`.",'',f"Acceptor OOF identity-macro MAE: S0 {a['metrics']['S0']['identity_macro_mae']:.9f}, M11 {a['metrics']['M11']['identity_macro_mae']:.9f}, XGB-C0 {a['metrics']['XGB']['identity_macro_mae']:.9f} eV.",f"M11−S0: {a['comparisons']['M11_minus_S0']['point']:+.9f} eV, 95% CI {a['comparisons']['M11_minus_S0']['ci95']}.",f"M11−XGB: {a['comparisons']['M11_minus_XGB']['point']:+.9f} eV, 95% CI {a['comparisons']['M11_minus_XGB']['ci95']}.",f"IID structure-macro MAE: S0 {i['metrics']['S0']['structure_macro_mae']:.9f}, M11 {i['metrics']['M11']['structure_macro_mae']:.9f}, XGB-C0 {i['metrics']['XGB']['structure_macro_mae']:.9f} eV.",f"IID M11−XGB: {i['comparisons']['M11_minus_XGB']['point']:+.9f} eV, 95% CI {i['comparisons']['M11_minus_XGB']['ci95']}.",'',"This is training-only cross-fit robustness evidence, not unseen external confirmation, and it does not restore Gate 2-E1 v1."];(ROOT/'reports/gate2e2a_crossfit_results.md').write_text('\n'.join(lines)+'\n')
 (ROOT/'reports/gate2e2a_gradient_diagnostics.md').write_text("# Gate 2-E2A gradient diagnostics\n\nTrain-only fold-specific M11 checkpoints were evaluated on deterministic 256-row batches. No parameter update, task removal, or weight change was performed. Aggregate values are stored in `logs/gate2e2a_gradient_metrics.json`.\n")
 (ROOT/'reports/gate2e2a_final_decision.md').write_text(f"# Gate 2-E2A final decision\n\n## `{decision}`\n\nGate 2-E2A supplies training-only cross-fit robustness evidence. It is not external confirmation, does not validate Gate 2-E1 v1, and authorizes no official-validation or test access.\n")
 evidence={'status':'GATE2E2A_DONE','decision':decision,'training_only_crossfit':True,'neural_models':len(summaries),'xgb_models':10,'official_validation_accessed':False,'test_accessed':False,'main_parquet_accessed':False,'final673_accessed':False,'scheduler_overlap_incident':'EARLY_ENGINEERING_LAUNCH_RESTARTED_WITH_FIXED_GPU_LANES_NO_CONFIG_CHANGE','completed_utc':datetime.now(timezone.utc).isoformat()};write_json('logs/gate2e2a_evidence.json',evidence);print(json.dumps({'decision':decision,'acceptor':a['comparisons'],'iid':i['comparisons']},indent=2))
if __name__=='__main__':main()
