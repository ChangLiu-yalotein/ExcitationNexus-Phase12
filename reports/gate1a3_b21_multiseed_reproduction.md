# Gate 1-A3 B2-1 multiseed reproduction

Final status: **FAILED_REPRODUCTION**. Both fixed 80-epoch runs completed successfully, but numerical reproduction failed the preregistered tolerance.

| Seed | Historical MAE (eV) | New MAE (eV) | Absolute delta | Threshold pass |
|---:|---:|---:|---:|:---:|
| 42 | 0.076590605080 | 0.077453471720 | 0.000862866640 | True |
| 123 | 0.081370525062 | 0.079640269279 | 0.001730255783 | False |
| 456 | 0.080265983939 | 0.077275253832 | 0.002990730107 | False |

New mean/sample std: `0.078122998277 ± 0.001317013240` eV.  Historical exact mean/sample std: `0.079409038027 ± 0.002502533912` eV.  The aggregate mean delta is `0.001286039750` eV, also above 0.001 eV.

Seed123 selected epoch 30 with validation batch-macro MAE 0.076926558697; seed456 selected epoch 13 with 0.079292809105. No run was restarted or tuned after test inspection. Final673 and new15016 were not accessed.

Seed123 ran for 9937.56 s with an observed peak of 14837 MiB and maximum sampled temperature 84°C. Seed456 ran for 10991.56 s with an observed peak of 14821 MiB; its maximum sampled temperature was 90°C, but the sustained-temperature stop condition was not met. Both runs completed normally with no observed NaN, OOM, or Xid event.
