# Phase 12 TODO

## DONE

- [x] Gate 0-A GPU, system, environment, closed-loop data, field-contract, and historical overlap audit.
- [x] Clarify 15,016 calculation records versus 14,639 canonical structures.
- [x] Freeze V1 full/component/pair/role-aware identity keys.
- [x] Audit all 372 duplicate groups, enabled targets, sidecars, atom origins, DFT JSON/PDB geometry, and RMSD.
- [x] Quantify donor/acceptor aliases, conflicts, frequencies, Murcko scaffolds, and long tails.
- [x] Freeze group-level historical quarantine without final673 per-sample artifacts.
- [x] Freeze historical benchmark correction and structure-purged sensitivity policy.
- [x] Select evidence-backed replicate policy: retain records with `group_weight=1/group_size`.
- [x] Complete Gate 0-B success criteria.
- [x] Pre-register and lock Gate 0-C before split solving.
- [x] Generate and freeze six grouped Gate 0-C manifests.
- [x] Verify leakage invariants, repeatability, and input-order independence.
- [x] Run post-freeze balance, target-description, and Morgan OOD diagnostics without changing v1.
- [x] Gate 0-D reusable pipeline, firewall, normalization, masked loss, and group-aware metrics.
- [x] Gate 0-D 33 CPU tests, six joins, 72 PM6/DFT graph parses, and CPU model smoke.
- [x] Gate 0-D physical GPU 0 two-batch forward/backward/AdamW smoke.

## TODO

- [x] Gate 1-A1: strict historical reproduction of the 0.0702 no-dipole cheap champion.
- [x] Gate 1-A2: numerically reproduce frozen B2-1 seed42 checkpoint and one 80-epoch training run.
- [x] Gate 1-A3: fixed seeds 123/456 completed; status `FAILED_REPRODUCTION` because both seeds and the aggregate mean exceeded the preregistered 0.0010 eV tolerance.
- [x] Gate 1-B1: establish leakage-safe median, Ridge-C0, XGBoost-C0, and XGBoost-C1.5-safe baselines on the frozen new15016 grouped-IID split with one-time test evaluation.
- [x] Gate 1-B2: govern empty-donor roles, freeze a target-free DFT graph registry, and admit parameter-matched M3-Merged/M3-DAU-Shared through CPU and three-epoch GPU smoke.
- [x] Gate 1-B3: freeze six formal IID checkpoints and complete the one-time test, ensembles, XGBoost-C0 pairing, and structure-group bootstrap.
- [x] Gate 1-B3: complete the preregistered 198-record graph-supported role-candidate input sensitivity with frozen checkpoints and no retraining.

- [x] Gate 1-C1: diagnose frozen IID error mechanisms and geometry value; final decision `STOP_PURE_3D`.

## BLOCKED

- [!] Time/prospective split: `BLOCKED_NO_TRUSTED_TIMESTAMP`.

- [x] Gate 2-A: freeze 20 cheap/multi-fidelity assets and complete one-time evaluation on five OOD protocols.

- [x] Gate 2-B: audit OOD uncertainty at held-out donor/acceptor/pair/scaffold and crossed-cluster inference units.

- [x] Gate 2-C: freeze validation-only conformal calibrators and audit IID/OOD coverage plus target-free applicability-domain filtering.

- [x] Gate 2-D1: test equal-budget role-separated donor/acceptor 2D fingerprints on validation only; decision `ROLE_AWARE_2D_NOT_ADMITTED`.

- [!] Gate 2-D2: frozen continuous representation v1 is `BLOCKED_PREREGISTERED_PCA_INFEASIBLE`; donor PCA-256 cannot be fitted on 124-154 unique protocol-train donors. No embedding or validation model was run.

- [x] Gate 2-D2 v2: replace infeasible PCA with fixed Gaussian random projection, pass real 417/208/378-token forward and embedding integrity gates, freeze 12 validation-only models; decision `REPRESENTATION_SIGNAL_INCONCLUSIVE`.

- [x] Gate 2-E0: freeze `MULTITASK_TARGET_GRAPH_ADMITTED` with 11 secondary and 4 masked tasks after one protocol-local auxiliary-label extraction.

- [x] Gate 2-E1: freeze MULTITASK_SIGNAL_INCONCLUSIVE and MASKED_FRAGMENT_SIGNAL_INCONCLUSIVE from IID/acceptor-cold validation only.

- [!] Gate 2-E1 correction: BLOCKED_MULTITASK_PIPELINE_INTEGRITY; acceptor inner split violated the frozen group-weighted quantile rule after validation was consumed.

- [x] Gate 2-E2A: recover training-only cross-fit evidence with corrected group-weighted unit targets; decision MULTITASK_CROSSFIT_INCONCLUSIVE.
- [x] Gate 2-F0: audit PM6/DFT ground-state semantics, role/interface descriptors, matched deltas, duplicate dispersion, and acceptor-cold shift; decision `DELTA_FEATURE_GRAPH_ADMITTED`.
- [x] Gate 2-F1: test seven equal-contract ground-state/delta arms on frozen training-only folds; decisions `MULTIFIDELITY_SIGNAL_INCONCLUSIVE` and `DELTA_REPARAMETERIZATION_NO_GAIN`.
