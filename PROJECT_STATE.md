# Phase 12 project state

Current stage: **BLOCKED_MULTITASK_PIPELINE_INTEGRITY**; Gate 2-E1 metrics are diagnostic-only after a post-freeze inner-split weighting violation was found.

## Frozen facts

- 15,016 complete PM6+DFT+TDDFT calculation records remain intact.
- These records form 14,639 canonical structure groups.
- Duplicate distribution is 14,267 singleton, 369 doubleton, 1 triplet, and 2 quadruplet groups.
- All 372 duplicate groups have resolved atom/role/bond geometry correspondence; PDB/JSON checks passed.
- Replicate policy is `RETAIN_REPLICATES_WITH_GROUP_WEIGHT`.
- new15016 vs final673 aggregate intersection is ID=0 and structure=0.
- external2698/final673 share 18 structures and old7316/external2698 share 17; these are historical benchmark corrections.
- Gate 0-C created only frozen split assets; no training, CUDA computation, final-label access, or raw-data modification occurred.

## Gate 0-C result

Six immutable target-blind grouped splits are frozen in `data_registry/SPLIT_REGISTRY_V1_FROZEN.json`. Every split covers all 15,016 records; one record is historical quarantine and all 17 old-training-overlap records are train-only. Independent leakage checks, repeat generation, and shuffled-input generation passed. Both-cold retains 78.0767% in train/val/test with 3,291 explicit buffer records. Time/prospective split remains `BLOCKED_NO_TRUSTED_TIMESTAMP`.

## Gate 0-D result

- 33/33 CPU tests passed.
- Six full-table joins, train-only group-weighted normalization registries, and 72 PM6/DFT graph parses passed.
- Tiny plumbing model CPU forward/backward and translation/rotation/permutation invariance passed.
- Physical GPU 0 completed two FP32 batches with finite/nonzero gradients and a real AdamW parameter update.
- 387 records have explicit unknown atoms and no donor-labelled atom; unknown is never inferred as donor.
- Code snapshot SHA-256: `432cebe0fdd1088ec798fc130968fab0f37e3dd635deaa3833f29b104649b392`.

No formal epoch, model comparison, checkpoint selection, final673 access, or frozen split modification occurred.

Project and EquiformerV3 directories are not Git worktrees; source commit provenance remains unavailable and must not be fabricated.

## Gate 1-A1 result

- Historical `B_direct_C1.5_no_dipole` was reproduced once on physical GPU 0.
- Test MAE/RMSE/R²: 0.07020991162281436 / 0.1033576733887722 / 0.7574868831988364 on 1,097 records.
- New and historical prediction CSVs are byte-identical; SHA-256 `16a7e9a8c60176ae0f5c2f31ca6be10ece374967131996a1da6096b0d06818ea`.
- The result is strict historical reproduction, not leakage-corrected confirmation: the old workflow inspected test metrics when selecting the no-dipole champion.
- No B2-1, new15016, final673, extra seed, hyperparameter search, or checkpoint was run.

## Gate 1-A2 result

- Historical B2-1 seed42 checkpoint inference reproduced the 1,098-SID vector numerically: MAE 0.07656089216470718 eV.
- Exactly one seed42 training completed 80 epochs / 51,200 steps on physical GPU 0.
- New test MAE/RMSE/R²: 0.07745347172021866 / 0.12340133565776104 / 0.6542971134185791 on 1,098 records.
- Absolute MAE difference from the original historical run is 0.0008628666400909424 eV, within the 0.0010 eV numeric-reproduction tolerance.
- Original B2-1 uses 5,120/1,098/1,098 records; its three additional SIDs explain the difference from Layer G 7,313.
- The implemented model uses shared graph-level scalar energy projections, not intermediate atom-embedding pooling; parameter count is 1,065,570.
- No other seed, new15016, final673, B2-0, B2-2a, or hyperparameter search was run.

## Gate 1-A3 result

- Historical seeds 123/456 assets passed audit; the shared checkpoint-directory collision is explicitly preserved.
- V1 preregistration was frozen before inference/training with aggregate SHA-256 `9765a811d782c3d05b82ae7c53a19b11435e9ae0a9ce5f2fa5d6389a129366e3`.
- Historical exact seed123/456 test MAEs are 0.0813705250620842 and 0.08026598393917084 eV; the historical exact three-seed mean/sample std is 0.07940903802712758 ± 0.0025025339119785065 eV.
- 41 tests, both CPU forward smokes, and both 1,098-SID historical checkpoint inferences passed.
- Seed123 completed 80 epochs on physical GPU 0; seed456 completed 80 epochs on physical GPU 1. Both produced independent best checkpoints and fixed 1,098-SID test predictions.
- New seed123/456 test MAEs are 0.07964026927947998 and 0.07727525383234024 eV. Their historical absolute deltas, 0.0017302557826042175 and 0.002990730106830597 eV, exceed the 0.0010 eV tolerance.
- The new exact three-seed mean/sample std is 0.0781229982773463 ± 0.0013170132399967678 eV; its mean differs from the historical exact aggregate by 0.0012860397497812814 eV.
- Gate 1-A3 is `FAILED_REPRODUCTION`: training succeeded technically, but seeds 123/456 and the aggregate mean did not numerically reproduce within the preregistered threshold.
- The three-seed B2-1 ensemble MAE on 1,097 common SIDs is 0.0741149457130847 eV versus cheap 0.07020991155682628 eV; paired bootstrap CI crosses zero, so no superiority claim is made.
- No seed42 rerun, new15016, final673, B2-0, B2-2a, tuning, or result-driven restart is permitted.

## Gate 1-B1 result

- The frozen IID manifest retained 10,387/2,309/2,319 train/val/test records, one historical-quarantine record, and 10,248/2,195/2,195 effective structure groups.
- C0-open uses 20 deterministic RDKit descriptors and 512 Morgan bits; C1.5-safe adds only resolved PM6 HOMO, LUMO, and gap. `pm6_energy_raw`, dipoles, DFT, and all TDDFT/Multiwfn fields remain forbidden.
- On the one-time 2,319-record / 2,195-group test, weighted median, Ridge-C0, XGBoost-C0, and XGBoost-C1.5-safe group-macro MAEs are 0.141744444, 0.090387231, 0.084181475, and 0.084240369 eV, respectively.
- The three XGBoost seeds are prediction-identical because the frozen historical transfer configuration has no row/feature subsampling; random seed is operationally inert in this protocol.
- C1.5-safe did not improve group-macro MAE over C0, so no PM6-orbital benefit is claimed.
- Test targets were accessed exactly once after all eight models, validation metrics, and hashes were frozen; no test-guided retraining occurred.
- Results describe `J_eh_screened_eV_eps3p5 proxy` on new15016 grouped IID only, not experimental Eb, catalytic performance, or direct progress over old Layer G.

## Gate 1-B2 result

- Original role classes reproduce 14,263 pure D/A, 366 D+A+unknown, and 387 empty-donor+unknown records.
- Conservative label-free graph governance resolves 198/387 empty-donor records to one unique/symmetry-equivalent atom set; 189 remain ambiguous. Original roles are primary, resolved roles sensitivity-only, and no record is removed.
- The target-free DFT cache/registry covers 15,016 records, 3,524,839 atoms, and 3,738,352 source bonds; maximum PDB/JSON rounding delta is within 0.000501 Å.
- M3-Merged and M3-DAU-Shared share a distance-only backbone and differ by only 0.621% in parameters (36,689 vs 36,461).
- Twelve CPU tests and both GPU two-batch/three-epoch smokes passed with finite gradients, real parameter changes, and translation/rotation/permutation invariance.
- Three-epoch validation is plumbing-only and cannot rank architectures. No test target, final673, formal multi-seed run, scalar quantum input, or paper checkpoint was used.


## Gate 1-B3 final result

- Six formal M3 checkpoints were trained once under the frozen common budget and original explicit D/A/U roles.
- On 2,319 IID records / 2,195 groups, XGBoost-C0, M3-Merged ensemble, and M3-DAU-Shared ensemble group-macro MAEs are 0.084181475, 0.087664065, and 0.088546137 eV.
- Paired structure-group bootstrap shows both 3D ensembles are worse than XGBoost-C0; Merged versus DAU crosses zero, so no explicit D/A/U separation benefit is claimed.
- No retraining, second test evaluation, or final673 access occurred.
- The 198 graph-supported role-candidate perturbations changed ensemble predictions materially (median absolute delta 0.07105 eV Merged; 0.05932 eV DAU), documenting role-definition sensitivity without treating candidates as ground truth.
- Final scientific status: `NEGATIVE_3D_ARCHITECTURE_RESULT_UNDER_FROZEN_IID_PROTOCOL`.

## Gate 1-C1 result

- Gate 1-C1 used only frozen test prediction artifacts; it created no new test predictions and ran counterfactuals on validation only.
- Geometry is real but insufficient to justify scaling: duplicate-group geometry RMSD versus target range has strong association, and zeroing coordinates strongly degrades validation predictions.
- Only three of six formal runs meet the preregistered underfit rule; both architectures therefore do not meet `SCALE_3D`.
- The only adequately powered stable 3D-winning subgroup is target Q4. Two model wins in that one subgroup do not satisfy the requirement for two distinct subgroups, so `FUSE_2D_3D` is not admitted.
- Role-candidate perturbations are 67.0%–81.1% of the corresponding IID MAE, confirming a major role-definition stability risk.
- Final status: `GATE1C1_DONE_STOP_PURE_3D`. This pauses the pure-3D path; it does not claim that geometry is physically irrelevant.

## Gate 2-A result

- Five frozen OOD protocols each contain weighted median, Ridge-C0, XGBoost-C0, and XGBoost-C1.5-safe, for 20 frozen baseline assets.
- XGBoost-C0 structure-group-macro MAEs are 0.084589716 donor-cold, 0.098030360 acceptor-cold, 0.085474007 pair-cold, 0.084398558 both-cold, and 0.081749508 eV full-scaffold-cold.
- Only acceptor-cold shows clear descriptive degradation from IID under independent structure-group bootstrap. Cross-protocol comparisons are not paired.
- C1.5-safe satisfies no protocol's preregistered PM6-orbital gain criterion.
- All 20 assets and ten preprocessors were frozen before one deduplicated 7,669-ID Arrow target read; the evaluator now fails closed.
- Both-cold buffer and historical quarantine received no Dataset, prediction, or metric; `final673` remained sealed.
- Final status: `GATE2A_DONE_OOD_BASELINES`.

## Gate 2-B result

- Cold-protocol primary inference now uses held-out identities; structure-group bootstrap remains secondary.
- Acceptor-cold has 36 identities: XGBoost-C0 identity MAE 0.097319544 eV and independent IID difference CI [0.004276044, 0.023622853] eV. Similarity correlations are -0.545 acceptor and -0.583 full. Label: `ACCEPTOR_OOD_FAILURE_CONFIRMED`.
- Donor-cold has only 15 identities and remains `LOW_CLUSTER_POWER`; its degradation CI crosses zero.
- Both-cold crosses 15 donor/40 acceptor identities. MAE 0.084398558 eV masks normalized MAE 0.691, R² 0.253, median skill 0.070, and regression-to-mean slope 0.310. Label: `BOTH_COLD_LOW_SKILL_WARNING`.
- PM6 LUMO/gap show locked acceptor-OOD shifts; weak frozen TreeSHAP/error association crosses the preregistered threshold. Label: `PM6_ORBITAL_SHIFT_RISK`, not causality.
- No training, new prediction, main Parquet read, Gate 2-A evaluator call, or final673 access occurred.
- Final status: `GATE2B_DONE_HIERARCHICAL_OOD_AUDIT`.


## Gate 2-C result

- The missing calibration-label artifact blocker was resolved through one authorized Arrow read of only `molecule_id` and the primary target for the 9,151-ID validation union. The row-level artifact remains local and Git-ignored.
- IID is only approximately exchangeable; donor-cold has 15 calibration donor identities and cannot attain 95% identity conformal coverage; OOD exchangeability remains unverified; both-cold is a crossed design without an exact conformal guarantee.
- The 90% acceptor-cold record interval covered 0.9093 of records. Its conservative acceptor-identity maximum-residual interval covered 0.9964 identity-macro with CI [0.9937, 0.9987], at width 0.9396 eV. `ACCEPTOR_UQ_UNDERCOVERAGE` was not triggered.
- Both-cold achieved empirical 90% record coverage 0.9370 with a two-way CI [0.8948, 0.9693], but remains `BOTH_COLD_UQ_UNSUPPORTED` because crossed-cluster exchangeability and donor power are inadequate.
- Validation-locked similarity filtering was not stable across all protocols; `AD_SCORE_NOT_VALIDATED` is frozen.
- No point model was trained or modified, no new point prediction was generated, the Gate 2-A evaluator was not called, and final673 remained sealed.
- Final status: `GATE2C_DONE_UQ_APPLICABILITY_AUDIT`; labels: `UQ_EMPIRICALLY_CALIBRATED_OOD`, `BOTH_COLD_UQ_UNSUPPORTED`, `AD_SCORE_NOT_VALIDATED`.


## Gate 2-D1 result

- One authorized two-column Arrow read supplied 1,017 protocol-local missing train labels. Every protocol reached 100% train-label coverage with zero local val/test/buffer/quarantine overlap; row-level artifacts remain local and ignored.
- The three frozen representation arms contain 532 / 1,556 / 1,556 columns. Arm B and C have the same 1,536-bit budget, and Arm A exactly reproduces all 15,016 frozen C0-512 fingerprints.
- Twelve independent XGBoost models were trained on physical GPU 0, one Arm B and one Arm C per protocol. No model, preprocessor, label-derived asset, or validation result was shared across protocols.
- Acceptor-cold identity MAEs are 0.096529 / 0.097919 / 0.100244 eV for A/B/C. Primary C−B is +0.002325 eV with acceptor-cluster CI [-0.003722, +0.010601], failing the admission rule.
- IID C−B is -0.000905 eV with CI [-0.002070, +0.000241], so IID non-inferiority passes but cannot rescue the failed primary endpoint.
- The acceptor fingerprint block is used (gain fraction 0.405; absolute TreeSHAP fraction 0.204), yet the lowest-similarity acceptor quartile worsens by +0.020295 eV. Fixed RA2D therefore does not solve acceptor OOD.
- Final decision: `ROLE_AWARE_2D_NOT_ADMITTED`. No test artifact, post-extraction source Parquet, or final673 asset was accessed.

## Gate 2-D2 result

- MoLFormer asset security and tokenizer gates passed at immutable revision `a14249e5ad9e3e7c3b1bb604393e914cfcebd2c8`; license is Apache-2.0 and safetensors is available.
- Full/donor/acceptor tokenizer audit covered 14,639 / 154 / 352 unique strings with zero unknown tokens, reconstruction failures, or truncations at the preregistered max length 512.
- Gate 2-D2 v1 stopped before executing remote code or extracting embeddings: the frozen Arm C requests 256 donor PCs from unique protocol-train donors, but only 124-154 unique donors exist (maximum centered PCA ranks 123-153).
- No PCA, XGBoost model, validation prediction, test artifact, main Parquet, or final673 asset was accessed. This is a protocol blocker, not a negative MoLFormer result.
- Final status: `BLOCKED_PREREGISTERED_PCA_INFEASIBLE`. Any retry requires an explicit v2 compression contract; v1 cannot be silently edited.

## Gate 2-D2 v2 result

- V1 remains frozen as `BLOCKED_PREREGISTERED_PCA_INFEASIBLE`; v2 replaces only the infeasible PCA with two fixed, target-free Gaussian random projections.
- The immutable safetensors SHA matched. A target-free correction made before any molecular forward established true tokenizer maxima 417/208/378 rather than the earlier regex counts 399/208/372.
- The long-sequence gate passed without truncation: repeat delta 0 and single-versus-padded-batch maximum delta 2.3544e-6. Inputs above 202 tokens remain outside documented pretraining length support.
- Frozen embeddings cover 14,639 full, 154 donor, and 352 acceptor identities. All exact embedding aliases arise from identical tokenizer ID sequences; none joins distinct token sequences.
- Twelve protocol-local validation-only XGBoost models were frozen. Acceptor C−A is -0.0012744 eV with CI [-0.0054454, +0.0026585]; C−B is -0.0030333 eV with CI [-0.0059965, -0.0001475]. IID C−A CI upper is +0.0029984 eV.
- Final v2 decision: `REPRESENTATION_SIGNAL_INCONCLUSIVE`. The role-separated continuous arm beats the full continuous control on acceptor-cold validation, but misses the frozen C0 effect threshold and IID non-inferiority rule.
- No test artifact, source Parquet, buffer/quarantine, final673, encoder fine-tuning, projection search, or post-result model change was used.

## Gate 2-E0 result

- The generic representation-replacement route is closed; C0-512 remains the default structural input.
- Exactly one authorized Arrow read selected molecule_id plus 16 auxiliary labels for the 15,015-ID six-protocol train/validation union. The primary was not reread; row-level artifacts remain local and ignored.
- Eleven secondary targets are admitted. tddft_t_index_angstrom is report-only redundant because t = D - H_CT within 0.001 angstrom rounding.
- All four fragment fractions remain masked-only because donor+acceptor is not strictly one for every jointly observed record.
- Gate 2-E1 weights are frozen: primary 1.0, secondary total 0.5, masked total 0.25; no dynamic weighting.
- Final decision: MULTITASK_TARGET_GRAPH_ADMITTED. No model, prediction, GPU, test artifact, buffer/quarantine, or final673 asset was used.

## Gate 2-E1 result

- Eighteen inner-selection runs and 18 same-seed full-train refits compare S0, M11, and M15 on identical C0-512 inputs and primary paths.
- Acceptor-cold M11-S0 is -0.000620306 eV with CI [-0.001623608, +0.000376664]; M11-XGBoost is +0.000569607 eV with CI crossing zero. Decision: MULTITASK_SIGNAL_INCONCLUSIVE.
- Acceptor-cold M15-M11 is -0.003349093 eV with CI [-0.008248288, +0.000815910]. Masked decision: MASKED_FRAGMENT_SIGNAL_INCONCLUSIVE.
- Train-only acceptor M11 aggregate-secondary cosine is -0.1426 with 63.6 percent negative task cosines; no weight or task changed.
- Official validation ran once after all models froze; second invocation failed closed. Test, source Parquet, buffer, quarantine, and final673 remained inaccessible.

## Gate 2-E1 post-freeze correction

- The acceptor-cold inner split used record mean rather than the preregistered structure-group-weighted primary mean. Correct reconstruction changes 146 record assignments; IID is unchanged.
- The defect was found after official validation unlock. V1 cannot be repaired or retrained without post-validation contamination.
- Final status is BLOCKED_MULTITASK_PIPELINE_INTEGRITY. Existing metrics are diagnostic only; no multitask admission or rejection conclusion is valid.
- No rerun, test access, force push, or published tag rewrite occurred.

## Gate 2-E2A result

- Gate 2-E1 v1 remains permanently BLOCKED_MULTITASK_PIPELINE_INTEGRITY; no old artifact or tag was changed.
- Training-only 5-fold cross-fitting produced 60 neural refits and 10 fold-matched XGBoost-C0 models using only IID and acceptor-cold official-train rows.
- Acceptor M11-S0 is -0.000352594 eV with identity CI [-0.001056860, +0.000369232]; M11-XGB is -0.000328942 eV with CI [-0.001937995, +0.001382827].
- IID M11-S0 is -0.000647392 eV and M11-XGB is -0.002402325 eV, with both structure-group CIs below zero. IID evidence does not replace the primary acceptor endpoint.
- Final decision: MULTITASK_CROSSFIT_INCONCLUSIVE. This is development-time robustness evidence, not external confirmation, and it does not restore Gate 2-E1 v1.
- Official validation, test, main Parquet, buffer, quarantine, and final673 were not accessed.
