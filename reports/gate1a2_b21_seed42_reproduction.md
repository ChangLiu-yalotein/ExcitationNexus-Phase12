# Gate 1-A2 B2-1 seed42 reproduction

Final status: **REPRODUCED_NUMERIC**.

## Primary result

The historical checkpoint was first evaluated without retraining. Its 1,098-SID test inference produced MAE 0.07656089216470718 eV, RMSE 0.12207155608497411 eV, and R² 0.6617075800895691. The maximum difference from the later historical SID vector was 3.5540e-7 eV, establishing numerical checkpoint compatibility.

Exactly one new seed42 run then completed 80 epochs and 51,200 steps on physical RTX 3090 GPU 0. Its best checkpoint was selected at epoch 80 using validation only. Fixed test inference produced:

| Metric | Historical original run | New seed42 run |
|---|---:|---:|
| Records | 1,098 | 1,098 |
| MAE (eV) | 0.07659060508012772 | **0.07745347172021866** |
| RMSE (eV) | 0.12215884029865265 | **0.12340133565776104** |
| R² | not stored in the original JSON | **0.6542971134185791** |

The absolute MAE difference is 0.0008628666400909424 eV, below the preregistered 0.0010 eV threshold. This is numeric, not bitwise, reproduction. The new checkpoint SHA-256 is `e893da44bcc0a2b9ef756381e2c550c9083737ffae6f3d99072062a3680b66e2`; the published prediction SHA-256 is `134e9ac1d51a72c77d2b62745bcd55f062dba2b656e599a5c9554eb1d89e01d3`.

## Protocol and runtime

- Original B2-1 databases: 5,120 train / 1,098 validation / 1,098 test, 7,316 disjoint SIDs.
- Shared EquiformerV3 tower, invoked separately for donor and acceptor; each graph-level scalar energy is projected to 64 dimensions, then late-concatenated and fused.
- Model parameters: 1,065,570. The checkpoint state has 1,075,318 tensor elements including buffers.
- DFT S0 coordinates, 8 Å cutoff, maximum 20 neighbors.
- Historical fixed normalization: mean 0.800596 eV, standard deviation 0.194270 eV.
- AdamW, batch 8, seed 42, FP32, 80 epochs, no test-based checkpoint selection.
- Best validation batch-macro MAE: 0.07653653435409069 at epoch 80.
- Training wall time: 9,941.873630168848 s (2.7616 h).
- Maximum GPU memory observed by periodic external polling: 14,767 MiB. This is a lower-bound observation, not a continuous-profiler peak.
- Checkpoint inference throughput: approximately 72 records/s.

Python 3.10.19, PyTorch 2.8.0+cu128, CUDA 12.8, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, and scikit-learn 1.7.2 were used.

## Historical provenance corrections

The historical B2-1 protocol is not the 7,313-record PM6-complete Layer G protocol. B2-1 includes two extra train SIDs (`D-113_A-136`, `D-90_A-117`) and one extra test SID (`D-33_A-79`). Cheap/B2-1 comparison is therefore restricted to 1,097 common test SIDs.

Two historical seed42 prediction artifacts also differ. The original 09:11 metrics/vector gives MAE 0.07659060508012772 eV; the later 13:38 SID vector used for paired analysis recomputes to 0.07656088892531875 eV, with maximum per-prediction difference 0.0082203 eV. Both remain frozen. The documented shared checkpoint-directory collision and nondeterministic GPU inference are plausible contributors, but causality is unresolved.

The historical trainer averages validation batch means equally, overweighting the final two-record batch. Strict reproduction retained this selection behavior; it must be corrected only in a separately named future protocol.

No final673, new15016, other B2-1 seed, B2-0, B2-2a, or hyperparameter search was accessed or run.
