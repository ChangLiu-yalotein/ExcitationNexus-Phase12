# Gate 1-B3 frozen IID test comparison

Status: **TEST_EVALUATED_EXACTLY_ONCE; ROLE SENSITIVITY PENDING**.

The prediction target is `J_eh_screened_eV_eps3p5 proxy`, not experimental binding energy or catalytic performance. All six checkpoints, validation predictions, configuration hashes, and the model/environment registries were frozen before the explicit one-time test unlock. No retraining or test-guided configuration change occurred.

## Frozen IID test results

The test contains 2,319 calculation records and 2,195 canonical-structure groups.

| Model | Record MAE (eV) | Group-macro MAE (eV) | Record RMSE (eV) | Record R² |
|---|---:|---:|---:|---:|
| XGBoost-C0 | 0.085299163 | 0.084181475 | 0.115824301 | 0.569181 |
| M3-Merged seed 42 | 0.091600328 | 0.090219238 | 0.122625114 | 0.517103 |
| M3-Merged seed 123 | 0.091366359 | 0.090072808 | 0.122935726 | 0.514654 |
| M3-Merged seed 456 | 0.090450630 | 0.089051334 | 0.123604726 | 0.509357 |
| **M3-Merged ensemble** | **0.089044336** | **0.087664065** | **0.120634643** | **0.532653** |
| M3-DAU-Shared seed 42 | 0.092592277 | 0.091450654 | 0.123542321 | 0.509853 |
| M3-DAU-Shared seed 123 | 0.091473023 | 0.090343783 | 0.121937703 | 0.522502 |
| M3-DAU-Shared seed 456 | 0.091564943 | 0.090258712 | 0.124323207 | 0.503637 |
| **M3-DAU-Shared ensemble** | **0.089747465** | **0.088546137** | **0.120732900** | **0.531892** |

Across individual seeds, M3-Merged has group-macro MAE `0.089781127 ± 0.000636246 eV`; M3-DAU-Shared has `0.090684383 ± 0.000664972 eV` (sample standard deviation, ddof=1). Ensemble values are reported separately and are not included in those seed means.

## Pre-registered paired group bootstrap

The bootstrap resamples the 2,195 `structure_group_id_v1` units, uses 10,000 draws and seed 20260720, and estimates the paired group-macro MAE difference. Positive `3D − XGBoost` values favor XGBoost.

- M3-Merged ensemble minus XGBoost-C0: `+0.003482590 eV`, 95% CI `[+0.001367485, +0.005583528]`.
- M3-DAU-Shared ensemble minus XGBoost-C0: `+0.004364662 eV`, 95% CI `[+0.002194901, +0.006502449]`.
- M3-Merged ensemble minus M3-DAU-Shared ensemble: `−0.000882072 eV`, 95% CI `[−0.002186489, +0.000432603]`.

Thus, both frozen 3D ensembles are worse than XGBoost-C0 under this IID protocol, with paired confidence intervals excluding zero. The Merged-versus-DAU interval crosses zero, so explicit D/A/U separation does not show a reproducible IID advantage here. This is a valid negative architectural result and must not be hidden by retraining or post-hoc model selection.

## Diagnostic strata

The test contains 2,197 pure-D/A, 56 D+A+unknown, and 66 empty-donor+unknown records; 2,071 records are singleton structures and 248 belong to replicated groups. These strata were not used for model selection. Their small and unequal sizes, especially the two unknown-role strata, prevent strong comparative claims. The complete per-stratum metrics remain in `runs/gate1b3_test_once/metrics.json`.

The pre-registered 198-record resolved-role input sensitivity remains pending. It will reuse the six frozen checkpoints without retraining and cannot alter the primary original-role test result above.
