# Gate 2-G0 paper-grade model and benchmark consolidation

## Scientific boundary

Historical Layer G and new15016 are separate ledgers and must never be ranked together. All metrics predict **J_eh_screened_eV_eps3p5 proxy**, not experimental Eb or catalytic activity. Gate 3's 16-item list is frozen as `EXPLORATORY_BASELINE_SHORTLIST_FROZEN`; experimental progression is paused.

## Historical ledger

| model | protocol | test_records | mae_eV | std_eV | checkpoint | status |
| --- | --- | --- | --- | --- | --- | --- |
| cheap no-dipole champion | Layer G paired 7313 | 1097 | 0.070210 |  | CHECKPOINT_MISSING_REPORT_ONLY | REPRODUCED_NUMERIC |
| B2-1 historical ensemble | historical 7316 | 1098 | 0.079409 | 0.002503 | PRESENT_3_SEEDS | HISTORICAL_ASSET_VALID |
| B2-1 new reproduction | historical 7316 | 1098 | 0.078123 | 0.001317 | PRESENT_3_SEEDS | FAILED_FROZEN_THRESHOLD_AGGREGATE |
| B2-0 | historical B2 | reported | report-only |  | CHECKPOINT_MAPPING_UNRESOLVED_REPORT_ONLY | REPORT_ONLY |
| B2-2a | historical B2-2a | reported | different historical scope |  | PRESENT_SEED42 | PROTOCOL_NOT_ALIGNED_WITH_NEW15016 |
| Paper A / Smoothed Memory | historical external-dev | protocol-specific | report-only |  | CHECKPOINT_MISSING_REPORT_ONLY | OWN_PROTOCOL_ONLY |

## new15016 IID benchmark

| model | records | groups | group_macro_mae_eV | group_macro_rmse_eV | group_macro_r2 | paper_status |
| --- | --- | --- | --- | --- | --- | --- |
| Weighted median | 2319 | 2195 | 0.141744 | 0.175043 | -0.000132 | BASELINE |
| Ridge-C0 | 2319 | 2195 | 0.090387 | 0.118550 | 0.541255 | BASELINE |
| XGBoost-C0 | 2319 | 2195 | 0.084181 | 0.114257 | 0.573874 | PRIMARY_BASELINE |
| XGBoost-C1.5-safe | 2319 | 2195 | 0.084240 | 0.113841 | 0.576977 | BASELINE |
| M3-Merged ensemble | 2319 | 2195 | 0.087664 | 0.118365 | 0.542686 | NEGATIVE_3D_BASELINE |
| M3-DAU-Shared ensemble | 2319 | 2195 | 0.088546 | 0.118500 | 0.541639 | NEGATIVE_3D_BASELINE |

## Asset integrity

Audited 228 local assets; load/smoke failures: 0. Phase-12 XGBoost/PyTorch assets received finite-forward smoke. Historical B2 state dictionaries were loaded and retain their prior frozen inference evidence. Missing model dumps remain report-only.

## Roadmap gaps

| item | state | evidence |
| --- | --- | --- |
| XGBoost-C0 new15016 | DONE | Gate 1-B1 / 2-A |
| M3-Merged and M3-DAU small invariant baselines | DONE_NEGATIVE | Gate 1-B3 |
| role-aware Morgan / frozen MoLFormer | NEGATIVE_OR_INCONCLUSIVE | Gate 2-D1/D2 |
| fixed-weight physics multitask | BLOCKED_THEN_INCONCLUSIVE | Gate 2-E1 correction / E2A |
| ground-state multifidelity/delta | NEGATIVE_OR_INCONCLUSIVE | Gate 2-F1 |
| Chemprop v2 D-MPNN strong 2D | NOT_IMPLEMENTED | roadmap gap |
| PaiNN/TensorNet2 strong 3D | NOT_IMPLEMENTED | roadmap gap |
| formal EquiformerV3 molecular baseline on new15016 | NOT_IMPLEMENTED | historical B2 is not this |
| explicit D/A interface cross-edge branch | NOT_IMPLEMENTED | M3-DAU has no interface branch |
| ReMEI-Net | NOT_IMPLEMENTED | no unified main model |
| PM6 gating / FiLM | NOT_IMPLEMENTED | C1.5 concat is not gating |
| A0-A10 parameter-matched ablation | NOT_IMPLEMENTED | roadmap gap |
| retrospective active-learning benchmark | NOT_IMPLEMENTED | roadmap gap |
| paper-grade model card and data card | NOT_IMPLEMENTED | roadmap gap |

## Decision

`BENCHMARK_CONSOLIDATED_READY_FOR_MAIN_MODEL`

Next permitted step is Gate 2-G1 preregistration; it is not started here.
