# Gate 1-A2 cheap versus B2-1 paired audit

This is an audit on the already inspected Layer G test set, not a new confirmatory superiority test.

The new B2-1 seed42 prediction was aligned to the frozen Gate 1-A1 cheap prediction on 1,097 common SIDs. `D-33_A-79` exists only in the 7,316-record B2-1 test database and is excluded from this paired table.

| Model | Common-SID MAE (eV) | RMSE (eV) | R² |
|---|---:|---:|---:|
| Cheap no-dipole | 0.07020991155682628 | 0.10335767339729517 | 0.7574868831588407 |
| New B2-1 seed42 | 0.07748756768072791 | 0.12345163819514829 | 0.6540259713798231 |

The paired MAE difference `cheap - B2-1` is -0.007277656123901632 eV. A same-vector Wilcoxon audit gives nominal p=0.013949873834921597, but it is not a preregistered independent confirmation: the test set and cheap configuration were previously inspected, and the new B2-1 run is one seed. It must not be used to claim that the cheap champion is significantly superior.

The frozen project-level wording remains: the cheap model numerically reaches B2-1-level performance at lower input cost, and the locked broader comparison p=0.145 remains the publication statement until an independently sealed confirmation is run.

The ID-aligned prediction/error table is `runs/gate1a2_b21_seed42/published/gate1a2_cheap_vs_b21_paired_1097.csv`, SHA-256 `fac1514fba8bcd89f74eab35024d25c208a35052e76fbd0cdc9ad6307b167064`.
