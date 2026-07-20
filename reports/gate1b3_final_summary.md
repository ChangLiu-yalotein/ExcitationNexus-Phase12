# Gate 1-B3 Final Summary

Final status: `GATE1B3_DONE`  
Scientific conclusion: `NEGATIVE_3D_ARCHITECTURE_RESULT_UNDER_FROZEN_IID_PROTOCOL`

## Primary frozen IID result

The primary target is the `J_eh_screened_eV_eps3p5 proxy`, not experimental exciton binding energy or catalytic performance.

| Model | Test structure-group-macro MAE (eV) |
|---|---:|
| XGBoost-C0 | 0.084181475 |
| M3-Merged three-seed ensemble | 0.087664065 |
| M3-DAU-Shared three-seed ensemble | 0.088546137 |

Paired structure-group bootstrap shows both small 3D ensembles are worse than XGBoost-C0 under the frozen IID protocol. M3-Merged versus M3-DAU-Shared crosses zero, so there is no evidence that explicit D/A/U separation improves this target. No result-driven rerun or model deletion was performed.

## Secondary robustness result

The preregistered 198-record graph-supported role-candidate perturbation was completed with all six frozen checkpoints. Original-role predictions reconciled within `1.20e-07 eV`; no checkpoint or parameter changed. Median absolute ensemble prediction changes were `0.07105 eV` for M3-Merged and `0.05932 eV` for M3-DAU-Shared. This is evidence of role-definition sensitivity, not evidence that candidate roles are ground truth.

## Boundary statement

Six formal checkpoints and the one-time IID test result are sealed. The standard test evaluator remains fail-closed. The analysis did not access `final673`, did not reread test labels from the master table, did not train on candidate roles, and did not enter attention, Engram, larger equivariant architectures, or OOD evaluation.

The next scientific gate should diagnose error mechanisms before architecture expansion: isolate whether XGBoost-C0 benefits from 2D composition, sample-efficiency, model capacity, or weak geometric dependence of the proxy.
