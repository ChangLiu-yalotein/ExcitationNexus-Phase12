# Gate 1-A1 cheap champion reproduction preregistration

Status: **FROZEN BEFORE TRAINING**.

## Objective

Run exactly one fixed historical reproduction of `B_direct_C1.5_no_dipole` on the frozen Layer G split. This Gate does not run B2-1, new15016, a hyperparameter search, or final673.

## Frozen inputs and protocol

- Split: historical `split_paired_7313_seed42.json`, 5,118 train / 1,098 validation / 1,097 test.
- Label: historical `eb_eV`, label only.
- Features in deterministic order: 20 RDKit descriptors, 512-bit Morgan fingerprint (radius 2, no chirality), then nine no-dipole PM6/QC fields.
- Total features: 541.
- No donor/acceptor IDs, dipoles, TDDFT/Multiwfn fields, Coulomb target transforms, final673 fields, or source paths enter the feature matrix.
- Historical PM6 energy is retained solely to reproduce the frozen old7313 protocol. This is not authorization to use the unresolved new15016 `pm6_energy_raw` field.
- Median imputer and StandardScaler fit on the 5,118 training records only.
- XGBoost: 500 estimators, depth 6, learning rate 0.05, `tree_method=hist`, `device=cuda`, seed 42, no early stopping.
- Formal training count: one. Test is prediction-only and is not passed to fit, early stopping, or tuning.

The historical PM6 directory contains three absent files and one invalid JSON, all handled by the frozen train-only median imputer plus missing flag. Readable sidecars retain pre-renumbering `basename` values. A cross-fidelity audit found that PM6 atom counts match the current filename SID structures for every comparable record, while the embedded values behave as old aliases; the filename SID is therefore the frozen join key. This provenance limitation remains explicit in the evidence.

Exact paths and SHA-256 values are frozen in `configs/gate1a1_cheap_reproduction_v1.json` and `data_registry/gate1a1_historical_asset_registry.json`.

## Historical selection limitation

The original Stage 2C script trained several configurations and selected `best_config` using test MAE. The prior Stage 2B-A ablation also inspected test metrics. Therefore the 0.0702 result is a historical, post-test-selected champion. This Gate can verify numerical or prediction-vector reproducibility, but it cannot retroactively make that model-selection process leakage-corrected.

A leakage-corrected confirmation must select the feature family using train/validation only, freeze it, and evaluate a genuinely untouched confirmation split once. Layer G test is no longer untouched, so future claims must use an independently sealed split. This corrected protocol is designed here but is not run in Gate 1-A1.

## Status rules

- `REPRODUCED_STRICT`: all asset/config/environment checks match and the ID-aligned prediction vector exactly matches the old vector.
- `REPRODUCED_NUMERIC`: protocol matches and absolute test-MAE difference is at most 0.0010 eV, but bitwise predictions differ for a documented numerical reason.
- `BLOCKED_ASSET`: a required input, split, implementation detail, prediction, or environment fact is unavailable.
- `FAILED_REPRODUCTION`: confirmed protocol differs by more than 0.0010 eV or the historical conclusion does not hold.

The run result must report full-precision MAE/RMSE/R², record counts, feature list/hash, prediction hash, ID-aligned old/new comparison, environment, GPU, observed peak memory, wall time, and all input hashes.
