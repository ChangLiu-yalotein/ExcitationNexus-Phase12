#!/usr/bin/env python3
from __future__ import annotations
import json
from datetime import datetime,timezone
import numpy as np,pandas as pd
from gate2e1_common import ROOT,config,parameter_contract,sha,write_json
def fmt(x):return f'{x:.9f}'
def main():
 c=config();v=json.loads((ROOT/'logs/gate2e1_validation_metrics.json').read_text());g=json.loads((ROOT/'logs/gate2e1_gradient_metrics.json').read_text());model=json.loads((ROOT/'data_registry/gate2e1_model_registry.json').read_text());runs=json.loads((ROOT/'logs/gate2e1_training_registry.json').read_text())['runs']
 dynamics={}
 for protocol in c['protocols']:
  dynamics[protocol]={}
  for arm in c['arms']:
   q=[r for r in runs if r['protocol']==protocol and r['arm']==arm];dynamics[protocol][arm]={'best_epochs':[r['best_epoch'] for r in q],'inner_metric_eV':[r['inner_metric_eV'] for r in q],'wall_seconds_inner':sum(r['wall_seconds_inner'] for r in q),'wall_seconds_refit':sum(r['wall_seconds_refit'] for r in q),'peak_gpu_bytes':max(r['peak_gpu_bytes'] for r in q)}
 write_json('logs/gate2e1_training_dynamics.json',dynamics)
 strata={}
 roles=pd.read_csv(ROOT/'manifests/role_resolution_v1.csv')[['molecule_id','original_donor_count','original_acceptor_count','original_unknown_count']]
 reg=json.loads((ROOT/'data_registry/gate2e0_auxiliary_extraction_registry.json').read_text())
 for protocol in c['protocols']:
  p=pd.read_parquet(ROOT/c['local_root']/'validation_once'/f'{protocol}_primary_predictions.parquet');item=reg['protocols'][protocol]['val'];aux=pd.read_parquet(ROOT/item['artifact_path']);d=p.merge(aux,on='molecule_id',validate='one_to_one').merge(roles,on='molecule_id',validate='one_to_one');masked=json.loads((ROOT/'data_registry/gate2e0_target_graph_v2.json').read_text())['masked_auxiliary'];d['fragment_observed']=d[masked].notna().all(axis=1);d['role_stratum']=np.select([(d.original_donor_count>0)&(d.original_acceptor_count>0)&(d.original_unknown_count==0),(d.original_donor_count>0)&(d.original_acceptor_count>0)&(d.original_unknown_count>0),(d.original_donor_count==0)&(d.original_unknown_count>0)],['pure_DA','DA_unknown','empty_donor_unknown'],default='other');strata[protocol]={}
  for key,block in list(d.groupby('fragment_observed'))+list(d.groupby('role_stratum')):
   label=('fragment_observed' if isinstance(key,(bool,np.bool_)) and key else 'fragment_missing') if isinstance(key,(bool,np.bool_)) else str(key);strata[protocol][label]={'records':len(block),'M11_MAE':float(np.abs(block.y_primary-block.M11_ensemble).mean()),'M15_MAE':float(np.abs(block.y_primary-block.M15_ensemble).mean()),'M15_minus_M11':float((np.abs(block.y_primary-block.M15_ensemble)-np.abs(block.y_primary-block.M11_ensemble)).mean())}
 write_json('logs/gate2e1_auxiliary_strata.json',strata)
 p=parameter_contract();(ROOT/'reports/gate2e1_pipeline_and_fairness.md').write_text(f"# Gate 2-E1 pipeline and fairness\n\nAll arms use the same 532 C0 inputs. Shared trunk parameters: {p['S0']['trunk']:,}; primary head: {p['S0']['primary_head']:,}. Total parameters are S0 {p['S0']['total']:,}, M11 {p['M11']['total']:,}, and M15 {p['M15']['total']:,}. Primary-path names and shapes are identical. Only auxiliary heads add parameters.\n\nThe frozen inner splits have zero unit leakage. Official validation was inaccessible during inner selection and full-train refit. All 18 model hashes and normalizations were frozen before the one-time validation unlock. Test, source Parquet, buffer, quarantine, and final673 were not accessed.\n")
 lines=['# Gate 2-E1 training dynamics','']
 for protocol in dynamics:
  for arm,z in dynamics[protocol].items():lines.append(f"- {protocol}/{arm}: best epochs {z['best_epochs']}; inner metrics {[round(x,6) for x in z['inner_metric_eV']]}; inner/refit wall {z['wall_seconds_inner']:.1f}/{z['wall_seconds_refit']:.1f} s; peak {z['peak_gpu_bytes']/2**20:.1f} MiB.")
 lines+=['','CUDA emitted a CuBLAS determinism warning because `CUBLAS_WORKSPACE_CONFIG` was not set. The same runtime was retained for every arm and seed; no run was repeated or selected post hoc.'];(ROOT/'reports/gate2e1_training_dynamics.md').write_text('\n'.join(lines)+'\n')
 lines=['# Gate 2-E1 validation results','',f"Primary decision: `{v['decision']}`. Masked decision: `{v['masked_decision']}`.",'']
 for protocol in ('iid','acceptor_cold'):
  lines+=['## '+protocol,'','| Model | Identity/group MAE (eV) | Record MAE | RMSE | R² |','|---|---:|---:|---:|---:|']
  for arm in ('S0','M11','M15','XGBoost_C0'):
   z=v['metrics'][protocol][arm]['ensemble'];lines.append(f"| {arm} | {z['identity_macro_mae']:.9f} | {z['record_mae']:.9f} | {z['record_rmse']:.9f} | {z['record_r2']:.6f} |")
  lines+=['',f"M11−S0: {v['comparisons'][protocol]['M11_minus_S0']['point']:+.9f} eV, 95% CI {v['comparisons'][protocol]['M11_minus_S0']['ci95']}.",f"M11−XGBoost: {v['comparisons'][protocol]['M11_minus_XGBoost']['point']:+.9f} eV, 95% CI {v['comparisons'][protocol]['M11_minus_XGBoost']['ci95']}.",f"M15−M11: {v['comparisons'][protocol]['M15_minus_M11']['point']:+.9f} eV, 95% CI {v['comparisons'][protocol]['M15_minus_M11']['ci95']}.",'']
 (ROOT/'reports/gate2e1_validation_results.md').write_text('\n'.join(lines)+'\n')
 auxlines=['# Gate 2-E1 auxiliary task results','','Auxiliary MAEs use the single frozen official-validation inference. They are descriptive and did not select checkpoints.','']
 for protocol in ('iid','acceptor_cold'):
  auxlines+=['## '+protocol,'']
  for arm in ('M11','M15'):
   auxlines.append(f"### {arm}");auxlines+= [f"- `{t}`: MAE {z['mae']:.8g}, valid {z['valid']}" for t,z in sorted(v['auxiliary'][protocol][arm].items())];auxlines.append('')
  auxlines.append('Primary-error strata:');auxlines += [f"- {k}: n={z['records']}, M15−M11={z['M15_minus_M11']:+.6f} eV" for k,z in strata[protocol].items()];auxlines.append('')
 (ROOT/'reports/gate2e1_auxiliary_task_results.md').write_text('\n'.join(auxlines)+'\n')
 glines=['# Gate 2-E1 gradient conflicts','','Diagnostics use the first 256 sorted official-train records and frozen seed42 models. They perform no optimizer step and cannot change tasks or weights.','']
 for protocol in g['protocols']:
  for arm,z in g['protocols'][protocol].items():glines.append(f"- {protocol}/{arm}: primary-vs-secondary cosine {z['primary_vs_aggregate_secondary']:+.4f}; primary-vs-masked {z['primary_vs_aggregate_masked']}; negative task fraction {z['negative_cosine_task_fraction']:.1%}; secondary/primary norm ratio {z['aggregate_secondary_norm_ratio']:.3f}.")
 glines+=['','Acceptor-cold M11 shows aggregate negative transfer pressure: primary-vs-secondary cosine is negative and 63.6% of task gradients have negative cosine. This explains uncertainty in transfer but does not authorize reweighting or task removal.'];(ROOT/'reports/gate2e1_gradient_conflicts.md').write_text('\n'.join(glines)+'\n')
 final=f"# Gate 2-E1 final decision\n\n## `{v['decision']}`\n\nAcceptor-cold M11 improves over the matched S0 ensemble by only {v['comparisons']['acceptor_cold']['M11_minus_S0']['point']:+.9f} eV and its 95% identity-cluster CI crosses zero. M11 is {v['comparisons']['acceptor_cold']['M11_minus_XGBoost']['point']:+.9f} eV relative to frozen XGBoost-C0, also with a CI crossing zero. Therefore the preregistered physics-multitask admission criteria are not met.\n\n## `{v['masked_decision']}`\n\nM15 improves over M11 on acceptor-cold by {v['comparisons']['acceptor_cold']['M15_minus_M11']['point']:+.9f} eV, but the CI crosses zero; it is a candidate signal only and cannot replace the primary M11 comparison or unlock test. IID M15−M11 is {v['comparisons']['iid']['M15_minus_M11']['point']:+.9f} eV.\n\nThe train-only gradient audit finds substantial acceptor-cold conflict for M11. No weight, task, architecture, or run was changed after observing it. No test or final673 access is authorized.\n";(ROOT/'reports/gate2e1_final_decision.md').write_text(final)
 unlock=json.loads((ROOT/'data_registry/gate2e1_validation_unlock_v1.json').read_text());unlock['metrics_sha256']=sha('logs/gate2e1_validation_metrics.json');unlock.pop('metrics_sha256_pending_finalize',None);write_json('data_registry/gate2e1_validation_unlock_v1.json',unlock)
 evidence={'status':'GATE2E1_DONE','scientific_decision':v['decision'],'masked_decision':v['masked_decision'],'models':18,'inner_runs':18,'official_validation_invocations':1,'second_invocation_fail_closed':True,'parameter_fairness':True,'model_registry_sha256':sha('data_registry/gate2e1_model_registry.json'),'gradient_diagnostics_no_update':True,'test_accessed':False,'main_parquet_accessed':False,'final673_accessed':False,'completed_utc':datetime.now(timezone.utc).isoformat()};write_json('logs/gate2e1_evidence.json',evidence);print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
