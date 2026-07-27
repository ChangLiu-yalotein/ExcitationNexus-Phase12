# Gate 2-G1C Chemprop v2 dependency recovery amendment

Status: **GATE2G1C_CHEMPROP_V2_DEPENDENCY_RECOVERED**.

The earlier blocker commit remains part of the audit trail. The blocker is now reclassified as **PYTORCH_WHEEL_NETWORK_DOWNLOAD_SLOW_OR_INTERRUPTED**, not a Chemprop v2 incompatibility. The root issue was the slow and interrupted download of the Python 3.11 CUDA PyTorch wheel and its CUDA dependencies. The original `ML` environment was not modified.

Recovered environment:

| item | value |
| --- | --- |
| environment | `/home/changliu/miniconda3/envs/chemprop-v2-g1c` |
| Python | 3.11.15 |
| Chemprop | 2.3.0 |
| Chemprop path | `/home/changliu/miniconda3/envs/chemprop-v2-g1c/lib/python3.11/site-packages/chemprop/__init__.py` |
| Torch | 2.8.0+cu128 |
| CUDA | 12.8 |
| CUDA visible to torch | true, 7 devices |
| RDKit | 2026.03.4 |
| Lightning | 2.6.5 |
| NumPy | 2.4.6 |
| pip check | No broken requirements found. |

The torch wheel was downloaded to a persistent cache and verified before installation:

```text
/home/changliu/.cache/chemprop-g1c/torch-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl
sha256=039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed
```

Admission smoke tests used only synthetic molecules in `/tmp`, not final673 labels, not Gate 3 candidates, and not any frozen benchmark test targets.

| smoke | accelerator | train batches | epochs | result |
| --- | --- | ---: | ---: | --- |
| CPU | cpu | 2 | 1 | finite train/validation/test path, test RMSE 0.707740306854248 |
| GPU | gpu 0 via `CUDA_VISIBLE_DEVICES=0` | 2 | 1 | finite train/validation/test path, test RMSE 0.7240775227546692 |

GPU 6 was not used. These smoke metrics are dependency checks only and are not scientific model results.

This amendment resolves the dependency blocker, but it does not complete Gate 2-G1C. The formal Chemprop v2 D-MPNN blind and observable-provenance arms remain to be run under the frozen Gate 2-G1B split contracts. XGBoost models and predictions from the prior G1C subset are not retrained or recalculated here.
