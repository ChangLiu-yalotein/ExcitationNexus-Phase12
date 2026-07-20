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
