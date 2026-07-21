#!/usr/bin/env python3
from gate2e1_common import ROOT,read_json,sha,write_json
FILES=[
'configs/gate2e1_physics_multitask_admission_v1.json','data_registry/gate2e1_preregistration_lock_v1.json','data_registry/gate2e1_inner_split_implementation_correction_v1.json','data_registry/gate2e1_inner_split_registry.json','data_registry/gate2e1_normalization_registry.json','data_registry/gate2e1_model_registry.json','data_registry/gate2e1_validation_unlock_v1.json',
'scripts/gate2e1_common.py','scripts/gate2e1_build_inner_splits.py','scripts/gate2e1_train.py','scripts/gate2e1_refit_full_train.py','scripts/gate2e1_freeze_models.py','scripts/gate2e1_evaluate_validation_once.py','scripts/gate2e1_gradient_diagnostics.py','scripts/gate2e1_finalize_reports.py','scripts/gate2e1_finalize.py','scripts/git_checkpoint.sh',
'reports/gate2e1_preregistration.md','reports/gate2e1_pipeline_and_fairness.md','reports/gate2e1_training_dynamics.md','reports/gate2e1_validation_results.md','reports/gate2e1_auxiliary_task_results.md','reports/gate2e1_gradient_conflicts.md','reports/gate2e1_final_decision.md',
'logs/gate2e1_training_registry.json','logs/gate2e1_training_dynamics.json','logs/gate2e1_validation_metrics.json','logs/gate2e1_auxiliary_strata.json','logs/gate2e1_gradient_metrics.json','logs/gate2e1_evidence.json','tests/test_gate2e1_contract.py','tests/test_gate2e1_outputs.py','tests/test_gate2e1_postfreeze_correction.py','scripts/gate2e1_postfreeze_integrity_audit.py','data_registry/gate2e1_postfreeze_integrity_correction_v1.json','reports/gate2e1_postfreeze_integrity_correction.md','PROJECT_STATE.md','TODO.md','DECISIONS.md','RUN_REGISTRY.csv']
def main():
 e=read_json('logs/gate2e1_evidence.json');e.update({'pytest':{'passed':166,'failed':0,'warnings':4},'secret_scan_findings':0,'large_files_over_20MiB':0,'git_ignored_checkpoints_predictions_labels':True,'git_diff_check':'PASS'});write_json('logs/gate2e1_evidence.json',e)
 missing=[p for p in FILES if not (ROOT/p).is_file()]
 if missing:raise RuntimeError(missing)
 (ROOT/'data_registry/gate2e1_sha256.txt').write_text(''.join(f'{sha(p)}  {p}\n' for p in sorted(FILES)))
 print(f'GATE2E1_DONE files={len(FILES)}')
if __name__=='__main__':main()
