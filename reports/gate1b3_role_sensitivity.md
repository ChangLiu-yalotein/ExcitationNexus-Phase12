# Gate 1-B3 Graph-Supported Role-Candidate Sensitivity

## Scope and status

Status: `ROLE_SENSITIVITY_COMPLETE_NO_RETRAINING`.

This is a secondary robustness analysis. The 198 mappings are graph-supported role candidates, not ground-truth donor labels. The primary scientific result continues to use the original explicit D/A/unknown roles. No model was trained, updated, selected, or rerun; `final673` was not accessed.

The candidate set contains 140 train, 26 validation, and 32 test records. All 189 `UNRESOLVED_AMBIGUOUS` records were excluded and no quarantine record entered a Dataset. Each candidate changed exactly 36 atoms from original unknown to donor while atom identities, coordinates, edges, ordering, and all scalar inputs remained frozen. The 198 records represent 196 structure groups, and no repeated structure group had an inconsistent candidate mapping.

## Frozen-input reconciliation

All six checkpoint hashes, the model registry, test-unlock record, and test-once artifacts matched their preregistered SHA-256 values. Original-role predictions on validation/test reconciled to the frozen artifacts with maximum absolute differences from `5.96e-08` to `1.19e-07 eV`, below the frozen `2e-6 eV` tolerance. Test truth was joined only from the frozen test-once prediction artifact; it was not reread from the master Parquet. The standard evaluator remains sealed.

## Prediction sensitivity

| Ensemble | Scope | Mean signed delta (eV) | Median abs delta (eV) | P90 | P95 | Max | abs(delta) > 0.01 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M3-Merged | all 198 | -0.072256 | 0.071053 | 0.095071 | 0.100254 | 0.206110 | 100.0% |
| M3-DAU-Shared | all 198 | -0.062097 | 0.059322 | 0.093857 | 0.103244 | 0.120642 | 99.5% |
| M3-Merged | test 32 | -0.067633 | 0.064338 | 0.088567 | 0.097200 | 0.099774 | 100.0% |
| M3-DAU-Shared | test 32 | -0.055797 | 0.055625 | 0.084603 | 0.092439 | 0.096916 | 96.9% |

Every ensemble record exceeded `0.001` and `0.005 eV`. Seed-pair sign agreement was 98.5–100% for M3-Merged and 94.9–97.0% for M3-DAU-Shared, although rank correlations were only moderate. The architecture ensembles agreed in delta sign for 98.0% of records, with Spearman `0.4663`. M3-Merged had a `0.00898 eV` larger mean absolute perturbation than M3-DAU-Shared.

## Error sensitivity

Candidate roles are an input perturbation, not an alternative model-selection route. Across all 198 records, mean group-level change in absolute error was `+0.01303 eV` for M3-Merged (95% group-bootstrap CI `[+0.00453,+0.02139]`) and `+0.02693 eV` for M3-DAU-Shared (`[+0.01968,+0.03408]`).

On the 32 already-unlocked test records, M3-Merged's point change was `-0.00665 eV` with CI `[-0.02532,+0.01254]`; M3-DAU-Shared's was `+0.00784 eV` with CI `[-0.00891,+0.02408]`. Both test intervals cross zero. These values cannot justify replacing the primary role annotation, retraining, or calling the candidates recovered ground truth.

## Interpretation

The models are materially sensitive to role definition: merely relabelling the graph-supported candidate atoms produces changes comparable to the overall prediction error scale. This identifies role semantics as a robustness limitation, particularly for the empty-donor subset. It does not establish that either role view is chemically correct.

Exact per-checkpoint, per-partition, threshold, bootstrap, seed-consistency, and stratified results are stored in `logs/gate1b3_role_sensitivity.json`; the paired prediction artifact is retained locally under `runs/` and is bound by SHA-256 `3034ec966e091cb06022bb63d0fe73adc137434fa631676a442c556bf6173294`.
