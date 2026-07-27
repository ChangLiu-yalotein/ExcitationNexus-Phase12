# Gate 2-G1C Chemprop formal admission smoke

Status: **GATE2G1C_CHEMPROP_FORMAL_ADMISSION_SMOKE_PASS**.

The smoke CSV was generated from actual Gate 2-G1B eligible development records, but only from the outer-train side of the initial unified fold. No outer-validation records, final673 labels, Gate 3 candidate assets, or frozen benchmark test targets were accessed.

| smoke | accelerator | train batches | epochs | result |
| --- | --- | ---: | ---: | --- |
| actual-data CPU | cpu | 2 | 1 | finite validation loss and finite test path; test RMSE 0.15585218369960785 |
| actual-data GPU | gpu 0 via `CUDA_VISIBLE_DEVICES=0` | 2 | 1 | finite validation loss and finite test path; test RMSE 0.19455985724925995 |

The smoke metrics are engineering dependency checks only, not scientific benchmark results. GPU 6 was not used.
