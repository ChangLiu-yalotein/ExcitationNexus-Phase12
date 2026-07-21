# Gate 2-G0 paper tables

These tables are consolidated from frozen aggregate artifacts. No test evaluator or new prediction was run.

## UQ 90% identity-level coverage and width

| Protocol | Identity-macro coverage | Cluster simultaneous coverage | Width (eV) | Normalized width | Calibration clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| IID | 0.905239 | 0.905239 | 0.375226 | 1.501880 | 2195 |
| donor-cold | 1.000000 | 1.000000 | 1.016671 | 4.797731 | 15 |
| acceptor-cold | 0.996414 | 0.805556 | 0.939559 | 3.259796 | 32 |
| pair-cold | 0.886726 | 0.886726 | 0.362799 | 1.504391 | 2275 |
| both-cold | 0.993095 (acceptor sensitivity) | 0.993186 record empirical | 0.630628 | 5.161946 | 288 |
| full-scaffold-cold | 0.964930 | 0.904762 | 0.531916 | 2.369092 | 731 |

Both-cold is crossed-cluster empirical evidence, not an exact conformal guarantee. Donor-cold has only 15 calibration identities and low inferential power.

## Parameter inventory

| Model | Trainable parameters | Frozen IID group-macro MAE | Scope |
| --- | ---: | ---: | --- |
| XGBoost-C0 | not meaningfully comparable | 0.084181 eV | 500 trees |
| M3-Merged | 36,689 | 0.087664 eV | new15016 IID |
| M3-DAU-Shared | 36,461 | 0.088546 eV | new15016 IID |
| B2-1 dual tower | 1,065,570 | historical ledger only | not comparable to new15016 |

## Cost inventory

| Model | Scope | Training time | Inference time | Peak memory |
| --- | --- | --- | --- | --- |
| XGBoost-C0 deployment | all 15,015 legal records | 1.494934 s | frozen registry | not uniformly captured |
| M3-Merged | new15016 IID, 3 seeds | per-run registry | per-seed frozen artifact | per-run registry |
| M3-DAU-Shared | new15016 IID, 3 seeds | per-run registry | per-seed frozen artifact | per-run registry |
| physics multitask E2A | training-only cross-fit | per-fold registry | OOF only | per-fold registry |

Missing cost cells are reported as missing; they are not estimated.

## Negative, blocked, and inconclusive methods

| Method | Status | Reason |
| --- | --- | --- |
| M3-Merged / M3-DAU | NEGATIVE | both weaker than XGBoost-C0 on frozen IID |
| role-aware Morgan | NOT_ADMITTED | lowest-similarity acceptors worsened |
| frozen MoLFormer | INCONCLUSIVE | did not beat C0 and failed IID guard |
| fixed-weight multitask | BLOCKED_THEN_INCONCLUSIVE | corrected acceptor cross-fit CI crossed zero |
| ground-state multifidelity/delta | INCONCLUSIVE_NO_DELTA_GAIN | small cross-fit effect; delta reparameterization no gain |

## Dataset and split mapping

Historical Layer G/B2 results and new15016 results remain separate. The former contains protocol-specific cheap/B2/Paper-A evidence; the latter contains IID, five frozen OOD protocols, UQ, and validation/training-only admission experiments. Cross-ledger numerical ranking is prohibited.
