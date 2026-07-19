# Gate 1-A1 historical cheap champion reproduction

Final status: **REPRODUCED_STRICT**.

## Result

The fixed historical `B_direct_C1.5_no_dipole` configuration was reproduced once on physical RTX 3090 GPU 0. The newly serialized, ID-aligned prediction CSV is byte-identical to the historical prediction asset.

| Metric | Train | Validation | Test |
|---|---:|---:|---:|
| Records | 5,118 | 1,098 | 1,097 |
| MAE (eV) | 0.03243142377937654 | 0.0719305354805247 | **0.07020991162281436** |
| RMSE (eV) | 0.04404783813039372 | 0.1124846772750741 | **0.1033576733887722** |
| R² | 0.9485807200997725 | 0.6809708158025116 | **0.7574868831988364** |

- Historical test MAE: `0.07020991162281436`.
- Absolute test-MAE difference: `0.0 eV`.
- Old/new prediction SHA-256: `16a7e9a8c60176ae0f5c2f31ca6be10ece374967131996a1da6096b0d06818ea`.
- Old/new serialized prediction files: byte-identical.
- Raw in-memory float32 versus reloaded CSV maximum difference: `5.7189941449209414e-08`; this is serialization dtype representation, not a prediction difference in the frozen artifact.

## Exact feature contract

Feature count: 541, in this order:

1. Twenty RDKit descriptors: `MolWt`, `MolLogP`, `MolMR`, `TPSA`, `NumHDonors`, `NumHAcceptors`, `NumRotatableBonds`, `NumAromaticRings`, `NumAliphaticRings`, `NumAromaticHeterocycles`, `NumAliphaticHeterocycles`, `NumSaturatedRings`, `NumHeteroatoms`, `HeavyAtomCount`, `NumValenceElectrons`, `NHOHCount`, `NOCount`, `FractionCSP3`, `RingCount`, `HallKierAlpha`, each prefixed with `pair_`.
2. `pair_morgan_0` through `pair_morgan_511`, Morgan radius 2, 512 bits, chirality disabled.
3. Nine PM6/QC fields: `pm6_homo_hartree`, `pm6_lumo_hartree`, `pm6_homo_lumo_gap_hartree`, `pm6_homo_lumo_gap_ev`, `pm6_pm6_energy_hartree`, `pm6_num_atoms`, `pm6_normal_termination`, `pm6_n_warnings`, `pm6_missing_flag`.

The complete ordered list is frozen in `data_registry/gate1a1_feature_columns_v1.json`; ordered-column SHA-256 is `ac31990f3a45f2dad441cbabe94601094e445d7686ed65982404f21fb81e252a`.

No dipole, TDDFT/Multiwfn, Coulomb-equivalent target, donor/acceptor ID, final673, source path, or target column entered the matrix. The historical PM6 energy field is retained only for exact old7313 reproduction and remains forbidden for new15016 until its semantics are resolved.

## Preprocessing and model

- `SimpleImputer(strategy=median)` fit only on the 5,118 training records.
- `StandardScaler` fit only on the imputed training matrix.
- XGBoost 3.2.0: 500 estimators, depth 6, learning rate 0.05, `tree_method=hist`, `device=cuda`, seed 42.
- No early stopping, evaluation set, grid search, additional seed, or checkpoint selection.
- Formal models trained in this Gate: one.

The historical and current software environments match: Python 3.10.19, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, XGBoost 3.2.0, and RDKit 2025.09.6.

## PM6 provenance finding

Three Layer G PM6 files are absent and one JSON is invalid; the frozen missing flag and train-only median imputer handle them. All 7,309 readable sidecars carry a pre-renumbering `basename` rather than the current filename SID. This is not an unexamined assumption: for all 7,296 atom-count-comparable cases, PM6 `num_atoms` matches the current filename SID structure; PM6 and DFT sidecars also share the same old alias in all 7,309 readable cases. Current filename SID is therefore the frozen model join key, while the stale embedded alias remains a documented provenance limitation.

## Historical model-selection limitation

The model training itself did not use validation or test data. However, the original Stage 2C script selected the best configuration using test MAE across multiple candidates, and the preceding no-dipole ablation had also inspected test results. This result is therefore a strict **historical reproduction**, not a leakage-corrected independent confirmation.

The old Stage 2C comparison reported a nominal no-dipole-versus-B2-1 Wilcoxon p-value of 0.0489, while the broader frozen project statement uses the C1.5 comparison p=0.145. Neither supports a new superiority claim here because the no-dipole configuration was selected after test inspection and this Gate ran no B2-1 model. The permitted statement remains: the cheap model numerically reaches B2-1-level performance at lower input cost.

A leakage-corrected confirmation requires train/validation-only selection followed by one evaluation on an independently sealed split; Layer G test can no longer serve as untouched confirmation data.

## Runtime and scope

- Physical GPU: 0, NVIDIA GeForce RTX 3090.
- Observed process peak GPU memory: 370 MiB.
- Wall time: 43.113010614179075 seconds.
- No model/checkpoint was saved.
- No B2-1, new15016, final673, extra seed, hyperparameter search, or additional formal run occurred.

Primary evidence: `logs/gate1a1_evidence.json`, `logs/gate1a1_asset_audit.json`, and `runs/gate1a1_cheap_reproduction/`.
