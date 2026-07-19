# Gate 1-B1 new IID cheap baselines

Status: **GATE1B1_DONE**. The target is `J_eh_screened_eV_eps3p5 proxy`; it is not experimental Eb or catalytic efficiency.

Frozen IID test: 2,319 records / 2,195 structure groups. The historical-quarantine record never entered a Dataset. Test labels were unlocked once only after all eight model and validation hashes were frozen.

| Model | Record MAE (eV) | Group-macro MAE (eV) | Record RMSE (eV) | R² |
|---|---:|---:|---:|---:|
| weighted_median | 0.143113163 | 0.141744444 | 0.176468084 | -0.000064 |
| ridge_c0 | 0.091447580 | 0.090387231 | 0.120060718 | 0.537089 |
| xgb_c0_seed42 | 0.085299163 | 0.084181475 | 0.115824301 | 0.569181 |
| xgb_c1p5_safe_seed42 | 0.085371395 | 0.084240369 | 0.115448270 | 0.571974 |
| xgb_c0_seed123 | 0.085299163 | 0.084181475 | 0.115824301 | 0.569181 |
| xgb_c1p5_safe_seed123 | 0.085371395 | 0.084240369 | 0.115448270 | 0.571974 |
| xgb_c0_seed456 | 0.085299163 | 0.084181475 | 0.115824301 | 0.569181 |
| xgb_c1p5_safe_seed456 | 0.085371395 | 0.084240369 | 0.115448270 | 0.571974 |

XGBoost-C0 group-macro MAE: `0.084181475 ± 0.000000000 eV`; XGBoost-C1.5-safe: `0.084240369 ± 0.000000000 eV` (sample std, ddof=1). All three seed predictions are identical because the frozen historical transfer configuration uses no row or feature subsampling; the seed is therefore operationally inert. C1.5-safe does not improve this frozen baseline, so no PM6-orbital gain is claimed.

These values are an internal new15016 grouped-IID comparison and are not directly comparable to old Layer G as evidence of model progress. Role, replicate, component-frequency, and train-quartile target strata are recorded in the metrics JSON for diagnosis only and did not guide selection.

## Coverage and runtime evidence

- All 15,016 rows have finite values for the 532 C0-open and 535 C1.5-safe features; the admitted-feature missing fraction is zero. The old 541-column feature set is not reproduced because raw PM6 energy and control/redundant fields fail the current contract.
- C0 training wall times for seeds 42/123/456 were 3.2345/3.1408/3.1081 s; C1.5-safe times were 1.6174/1.5894/1.6327 s. Observed peak GPU memory was 395 MiB and maximum peak RSS was 717,212 KiB.
- Frozen-model test inference took 0.0024 s for Ridge and 0.0031–0.0209 s for each XGBoost model. C0 and C1.5 model/prediction hashes are identical across seeds, consistent with the deterministic configuration.
- Test role strata contain 2,253 donor+acceptor+unknown records, 66 empty-donor+unknown records, and zero pure donor/acceptor records. Singleton and replicated strata contain 2,071 and 248 records. These post-freeze summaries are diagnostic only; sparse strata are not used for scientific ranking.
