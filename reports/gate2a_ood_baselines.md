# Gate 2-A frozen OOD cheap baselines

Status: **GATE2A_DONE_OOD_BASELINES**.

Primary model: XGBoost-C0 (532 frozen C0 columns). C1.5-safe is a secondary PM6-orbital control. The target is `J_eh_screened_eV_eps3p5 proxy`, not experimental Eb.

| Protocol | Test records | Test groups | Median | Ridge-C0 | XGB-C0 | XGB-C1.5-safe |
|---|---:|---:|---:|---:|---:|---:|
| donor-cold | 2,251 | 2,224 | 0.126801203 | 0.125077233 | 0.084589716 | 0.083375093 |
| acceptor-cold | 2,237 | 2,208 | 0.160932843 | 0.120586675 | 0.098030360 | 0.099214340 |
| pair-cold | 2,309 | 2,191 | 0.143151375 | 0.091863096 | 0.085474007 | 0.085280034 |
| both-cold | 587 | 587 | 0.090703214 | 0.169790392 | 0.084398558 | 0.083740806 |
| full-scaffold-cold | 2,250 | 2,197 | 0.135345369 | 0.089517602 | 0.081749508 | 0.081409208 |

All table values are structure-group-macro MAE in eV. Full record/group metrics, target mean/SD/IQR, normalized MAE, bootstrap intervals, replicate/role/frequency strata, and three-view similarity strata are frozen in the metrics JSON (SHA-256 `7cf8761f6dc1d7f478930cbf93985971df6d4464d4773374aea51b08f18bf606`).

C0 training fingerprints are Morgan radius 2 / 512 bits / no chirality. OOD diagnostics are separate Morgan radius 2 / 2048 bits / chirality-enabled assets and never enter a model.

Cross-protocol degradation is descriptive and uses independent structure-group bootstrap because IID and OOD test sets differ. No paired cross-protocol claim is made.

Both-cold contains 587 test records/groups; 3,291 records / 3,234 groups remain buffer and received no predictions. Full-scaffold-cold is not described as deep scaffold extrapolation because 75.60% of full scaffolds are singletons.

## Interpretation

- Donor-cold tests unseen donor structures. C0 is 0.084589716 eV and its independent-bootstrap degradation CI crosses zero.
- Acceptor-cold tests unseen acceptor structures. C0 rises to 0.098030360 eV; the difference-from-IID CI is [0.009069126, 0.018681001] eV and is the only clear protocol-specific degradation.
- Pair-cold tests seen components in unseen combinations, not strong structural OOD. Its degradation CI crosses zero.
- Both-cold tests unseen donor and acceptor components, but its 587-group test yields a wide CI; no degradation claim is supported.
- Full-scaffold-cold is singleton-heavy. Its lower point estimate has a difference CI crossing zero and is not evidence of improved OOD generalization.

XGBoost-C0 beats Ridge-C0 and weighted median within every protocol under paired structure-group bootstrap. C1.5-safe passes no preregistered orbital-gain gate; in acceptor-cold it is modestly but significantly worse than C0.
