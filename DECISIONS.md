# Phase 12 decisions

- No training or quantum calculation in Gate 0-A.
- A GPU is FREE only with no compute PID, <1 GiB used, and near-zero utilization.
- Bind every experiment with `CUDA_VISIBLE_DEVICES`; one independent seed/experiment per free GPU; single-GPU smoke first; no default seven-card DDP; recheck before launch; never kill non-project processes.
- Preserve raw data and both plans unchanged.
- Treat target as `J_eh_screened_eV_eps3p5 proxy`, never direct experimental Eb.
- Disable Tier 3 inputs, equivalent target transforms, `pm6_energy_raw`, and full-dataset statistics.
- Preserve final673 sealing: aggregate overlaps only; no per-sample blind membership artifact.
- Use normalized ID plus RDKit canonical structure for leakage boundaries.
- Split generation is blocked pending structure replicate/dedup policy and historical benchmark correction.

## Gate 0-B decisions

- 15,016 calculation records and 14,639 canonical structures are distinct counts; delete none.
- Use V1 structure/component hashes, not rows or numeric IDs.
- Keep one structure group wholly in one partition.
- Replicate recommendation `RETAIN_REPLICATES_WITH_GROUP_WEIGHT`; retained rows use `group_weight=1/group_size`.
- Preserve official metrics and add structure-purged sensitivity metrics.
- external/final overlap is a benchmark limitation, not a new-data blocker.
- No filesystem-mtime time split; `BLOCKED_NO_TRUSTED_TIMESTAMP`.

## Gate 0-C decisions

- Freeze preregistration v1 before any solver run; corrections require explicit v2.
- Preserve all 15,016 rows; isolate one external/legacy overlap as `historical_quarantine`.
- Force all 17 old-training overlaps to train; never merge old7316 into Phase 12 new-only training.
- Freeze six target-blind grouped protocols after independent reproducibility checks.
- Treat pair-cold as unseen-pair/seen-components, not strong component OOD.
- Use seen-component validation for both-cold; keep cross-component rows in explicit buffer.
- Do not post-hoc rebalance v1 from target distributions or diagnostic similarity.

## Gate 0-D decisions

- Bind loaders to frozen table/manifest hashes; retain the IID filename and record selected seed 123.
- Permit only explicit train/val/test; reject buffer and historical quarantine.
- Keep PM6 dipole disabled by default; DFT requires explicit tier2.
- Optimize one primary, twelve secondary, and four masked auxiliary targets; deterministic transforms are report-only.
- Fit train-only group-weighted normalization per split.
- Preserve unknown roles: for 387 no-donor-labelled records, never infer donor atoms; use unknown pooling and role-presence flags.
- TinyRoleAware3D is plumbing-only; smoke weights are not scientific hyperparameters.

## Gate 1-A1 decisions

- Reproduce only the fixed historical `B_direct_C1.5_no_dipole` configuration once; no test-guided rerun or tuning.
- Treat the byte-identical prediction vector as `REPRODUCED_STRICT` historical reproduction.
- Do not reinterpret the historical post-test-selected champion as leakage-corrected model selection.
- Require an independently sealed confirmation split for future superiority claims because Layer G test has already been inspected.
- Treat PM6 embedded `basename` as a pre-renumbering alias; use filename SID as the frozen join key, supported by atom-count and cross-fidelity checks.
- Keep historical PM6 energy only for old7313 reproduction; it remains disabled for new15016.

## Gate 1-A2 decisions

- Use original B2-1 ASE databases (5,120/1,098/1,098) for strict protocol reproduction; use only 1,097 common test SIDs for cheap pairing.
- Freeze both historical seed42 prediction vectors because the original run-level and later SID-regenerated artifacts differ; do not select the lower value post hoc.
- Describe the implemented B2-1 faithfully as shared graph-level scalar energy projection plus late fusion, not intermediate atom-embedding pooling.
- Count 1,065,570 model parameters separately from 1,075,318 checkpoint state tensor elements including buffers.
- Preserve the historical batch-macro validation/checkpoint behavior as a reproduction limitation; correct it only in a separately named future protocol.
- Classify Gate 1-A2 as REPRODUCED_NUMERIC; do not claim bitwise identity or cheap-model statistical superiority.

## Gate 1-A3 decisions

- Run seeds 123 and 456 exactly once in isolated output directories on separate physical GPUs; never rerun seed42 or tune after observing results.
- Preserve the historical checkpoint collision: seed123 historical inference uses the surviving epoch-80 `checkpoint.pt`; seed456 uses `best_checkpoint.pt`.
- Freeze the exact historical three-seed MAEs and recompute mean/sample standard deviation rather than using rounded legacy summaries.
- Keep batch-macro validation selection for reproduction and restrict cheap/B2-1 paired analysis to the common 1,097 test SIDs.
- Treat first-validation success as a launch gate only; no final reproduction status is assigned until both 80-epoch runs and fixed test inference finish.
- Freeze Gate 1-A3 as `FAILED_REPRODUCTION`; do not rerun or select lower results to convert it to success.
- Use the newly reproduced three-seed values as the engineering baseline for future Phase 12 comparisons, while retaining historical values as provenance rather than silently replacing them.
- Treat the 1,097-SID cheap/ensemble bootstrap as descriptive because the test set was already inspected; retain the locked p=0.145 publication wording.

## Gate 1-B1 decisions

- Mark the rounded historical `0.0750 ± 0.0025`, `13.5%`, p-value, Cohen's d, and `58% improved` claims as `SUPERSEDED_PENDING_RECOMPUTATION`; retain provenance without using them in the paper until prediction-vector reconciliation.
- Bind new IID experiments to manifest SHA-256 `f4572f2c1896d4228dd9eff67220adb7d0a02ad79b70c66766e6da876541c3f2`, not its historical seed-labelled filename.
- Keep the single historical-quarantine row outside every Dataset and all 17 historical-train-overlap rows train-only.
- Admit only deterministic RDKit 2D/Morgan features for C0 and resolved PM6 HOMO/LUMO/gap for C1.5-safe; permanently reject `pm6_energy_raw`, dipole, DFT, and Tier 3 inputs in this baseline.
- Use structure-group-macro MAE as the primary validation metric and group weights for all fitted models.
- Freeze all validation/model hashes before a one-time test unlock; never rerun or select models from test strata.
- Treat the identical three-seed XGBoost vectors as evidence that the inherited no-subsampling configuration is deterministic, not as independent stochastic replicates.
- Record that C1.5-safe did not improve MAE over C0 under the frozen baseline; make no PM6-orbital gain claim.

## Gate 1-B2 decisions

- Keep original D/A/unknown annotations as the primary analysis; allow uniquely graph-resolved roles only as a predeclared sensitivity view.
- Preserve all 387 empty-donor records and explicit unknown pooling. Never fold unknown into donor/acceptor or drop an unresolved record.
- Classify the 189 records with multiple element/bond-equivalent atom sets as `UNRESOLVED_AMBIGUOUS`; do not select a role map from performance.
- Restrict M3 inputs to atomic number, DFT S0 coordinates, explicit original role, distances, and role-presence flags; exclude every scalar quantum property and target equivalent.
- Use the same backbone family and optimization contract for M3-Merged and M3-DAU-Shared; parameter difference must remain within 5%.
- Name the separated model `M3-DAU-Shared`; it is not old B2-1. Its donor/acceptor/unknown calls share weights and empty roles use zero embeddings with presence flags.
- Treat Gate 1-B2 checkpoints and three-epoch validation as plumbing-only. Formal hyperparameters and test access remain unfrozen until Gate 1-B3 preregistration.


## Gate 1-B3 decisions

- Freeze original explicit D/A/U roles, seeds 42/123/456, the common budget, and validation group-macro checkpoint selection before formal training.
- Preserve all six formal results without result-driven reruns; DAU seed42 epoch-33 patience stopping is compliant.
- Unlock IID test exactly once after all model/config/environment hashes are frozen; prohibit second evaluation and test-guided retraining.
- Record that both 3D ensembles are worse than XGBoost-C0 under paired group bootstrap, while Merged versus DAU crosses zero; do not claim role-separation benefit.
- Keep resolved-role inference sensitivity-only; it cannot replace the original-role primary result.
- Freeze Gate 1-B3 as `NEGATIVE_3D_ARCHITECTURE_RESULT_UNDER_FROZEN_IID_PROTOCOL`; diagnose error mechanisms before adding attention, Engram, or larger equivariant models.
- Treat the 198 graph-supported role candidates as a robustness perturbation only. Their large prediction shifts expose role-definition sensitivity and cannot replace original roles or justify retraining.

## Gate 1-C1 decisions

- Freeze `GATE1C1_DONE_STOP_PURE_3D`: do not scale or continue a pure-3D model family under the current role contract.
- Do not reinterpret `STOP_PURE_3D` as absence of geometric signal; validation counterfactuals and duplicate groups show geometry dependence.
- Reject `SCALE_3D` because both architectures do not satisfy the preregistered underfit rule.
- Reject `FUSE_2D_3D` at this gate because only target Q4, not two distinct adequately powered subgroups, shows stable 3D advantage. Oracle-min remains non-deployable.
- Preserve original roles as primary. Graph-supported role candidates remain robustness perturbations, never ground truth.
- Any future fusion must be separately preregistered and select weights on validation only; Gate 1-C1 authorizes no further training.

## Gate 2-A decisions

- Keep XGBoost-C0 primary; C1.5-safe remains a secondary PM6 orbital control.
- Keep 512-bit non-chiral training Morgan features separate from 2048-bit chirality-enabled OOD diagnostics; diagnostics never enter models.
- Join labels only to each protocol's train/val rows. After all 20 assets freeze, evaluate the deduplicated union of five test sets with one Arrow target read and fail closed thereafter.
- Use paired structure-group bootstrap only within a protocol and independent structure-group bootstrap for IID-to-OOD descriptive degradation.
- Record acceptor-cold as the only clear C0 degradation; other protocol difference CIs cross zero.
- Make no PM6 orbital gain claim; acceptor-cold C1.5-safe is significantly worse.
- Preserve both-cold's 587-group limitation/3,291-record buffer and do not overstate the 75.60%-singleton scaffold protocol.

## Gate 2-B decisions

- Use held-out donor, acceptor, pair, or scaffold identity as the corresponding cold protocol's primary inference unit; structure-group inference is secondary.
- Use two-way donor/acceptor pigeonhole multiplicity bootstrap for both-cold, plus donor-only and acceptor-only sensitivities.
- Freeze `ACCEPTOR_OOD_FAILURE_CONFIRMED`: the 36-acceptor degradation CI remains wholly above zero and error rises as similarity falls.
- Mark donor-cold and the donor side of both-cold `LOW_CLUSTER_POWER` at 15 identities; make no strong donor-OOD significance claim.
- Freeze `BOTH_COLD_LOW_SKILL_WARNING`: narrow targets hide poor normalized error, R², median skill, and regression-to-mean.
- Freeze `PM6_ORBITAL_SHIFT_RISK` as a warning only; LUMO/gap shift plus weak SHAP/error association is not causal and authorizes no feature change.
- Future UQ coverage must respect the protocol's scientific inference unit rather than treating molecular records as independent.


## Gate 2-C decisions

- Authorize exactly one minimal validation-label extraction from the frozen source Parquet: `molecule_id` plus the primary target for the six-protocol validation union only. Keep the row-level artifact local and Git-ignored.
- Use finite-sample ranks without clipping. Report donor-cold 95% identity conformal as `UNATTAINABLE_FINITE_SAMPLE`; never substitute the maximum residual.
- Treat IID as approximately exchangeable, OOD identity coverage as empirical with unverified exchangeability, and both-cold as crossed-cluster unsupported rather than assigning an exact conformal guarantee.
- Freeze `UQ_EMPIRICALLY_CALIBRATED_OOD` for adequately powered identity protocols while preserving the exchangeability limitation. This does not repair acceptor-cold point-prediction degradation.
- Do not assign `ACCEPTOR_UQ_UNDERCOVERAGE`: the preregistered acceptor-identity coverage criterion did not fail. Preserve the wide interval as the cost of empirical coverage.
- Freeze `BOTH_COLD_UQ_UNSUPPORTED` despite acceptable marginal coverage, because 15 independent donors and crossed identities cannot support a reliable guarantee.
- Freeze `AD_SCORE_NOT_VALIDATED`: validation-locked similarity cutoffs did not stably reduce record and identity risk across all six protocols; do not select a test-derived cutoff.
- The first calibrator process exited after writing only a deterministic target-free similarity cache. Reuse was permitted only because no calibrator registry or quantile existed; the cache was schema/count checked and its hash frozen before calibration resumed.


## Gate 2-D1 decisions

- Treat each split as a protocol-local experiment. Labels may be used when an ID is train in that protocol, even if the same ID is test elsewhere; models, preprocessors, residuals, and selection evidence remain strictly protocol-specific.
- Freeze exactly three arms: C0-512 reference, full-molecule Wide-1536 capacity control, and equal-budget full/donor/acceptor RA2D-1536. Add no post hoc representation.
- Record three donor and one acceptor 512-bit excess-structure collisions as fixed-representation limitations, not parser leakage. No acceptor-cold validation identity is involved in the acceptor collision.
- Freeze `ROLE_AWARE_2D_NOT_ADMITTED`: acceptor-cold C−B has the wrong point direction and a CI crossing zero, although IID non-inferiority passes.
- Do not interpret the negative admission as role irrelevance. The acceptor block has substantial gain/SHAP use, but fails specifically in the low-similarity acceptor regime.
- Do not unlock test, add memory, expand 3D, fuse models, or add learned representations on the basis of this Gate.

## Gate 2-D2 decisions

- Lock MoLFormer to revision `a14249e5ad9e3e7c3b1bb604393e914cfcebd2c8`, Apache-2.0, safetensors, audited custom code, deterministic eval, mask-aware final-hidden-state mean pooling, and no truncation at max length 512.
- Preserve the documented pretraining domain mismatch for inputs above 202 tokens; do not silently truncate or change molecular/component strings.
- Freeze Gate 2-D2 v1 as `BLOCKED_PREREGISTERED_PCA_INFEASIBLE`: unique/equal-weight donor PCA-256 exceeds every protocol's identifiable rank.
- Do not manufacture PCA dimensions by duplicating component rows, zero-padding non-identifiable components, using held-out structures, changing the 256/256 allocation after lock, or substituting another encoder.
- Do not interpret the blocker as evidence against continuous representations. A future v2 requires a new preregistration and a mathematically feasible target-free compression rule; no v2 is authorized by this Gate.

## Gate 2-D2 v2 decisions

- Preserve v1 unchanged and authorize only fixed PCG64 Gaussian projections: 768×512 seed 20260720 and one shared 768×256 donor/acceptor matrix seed 20260721.
- Correct the target-free sequence-length audit before any molecular forward: the immutable tokenizer gives maxima 417/208/378. Preserve the original v2 lock and the explicit correction lock.
- Keep every >202-token input labeled `OUTSIDE_PRETRAINING_LENGTH_SUPPORT` despite successful forward execution.
- Treat exact embedding aliases as frozen-tokenizer representation aliases because every collision has identical token IDs; do not drop structures or replace the tokenizer post hoc.
- Freeze `REPRESENTATION_SIGNAL_INCONCLUSIVE`: C significantly improves over equal-budget full continuous embedding on acceptor-cold validation, but C versus C0 is below the preregistered effect threshold with CI crossing zero, and IID non-inferiority fails.
- Do not unlock test or try another foundation encoder from this result. Any next intervention requires a new scientific Gate.

## Gate 2-E0 decisions

- Close generic representation replacement and retain frozen C0-512 as the default input.
- Keep the screened Coulomb proxy as the sole optimized member of its algebra family.
- Admit 11 secondary tasks; keep `t_index` report-only because `t = D - H_CT` within source rounding.
- Retain all four fragment fractions as masked-only; never impute missing labels or erase unknown/unassigned contribution.
- Freeze Gate 2-E1 weights before performance: primary 1.0, secondary total 0.5, masked total 0.25; no dynamic balancing.
- Validation is coverage-only in E0; no test access or training is authorized.

## Gate 2-E1 decisions

- Keep identical C0-512 inputs and primary paths across S0, M11, and M15.
- Freeze MULTITASK_SIGNAL_INCONCLUSIVE because the favorable M11 acceptor point estimate has a CI crossing zero and does not beat XGBoost.
- Freeze MASKED_FRAGMENT_SIGNAL_INCONCLUSIVE; M15 cannot become a post hoc primary champion.
- Preserve gradient conflict as explanation only; do not reweight, remove tasks, use PCGrad/GradNorm, or rerun.
- Do not unlock test or continue dynamic weighting, 3D, foundation encoders, fusion, memory, or retrieval from this result.
