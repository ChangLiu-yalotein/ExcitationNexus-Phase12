# Gate 1-A2 B2-1 seed42 reproduction preregistration

Status: **FROZEN BEFORE CHECKPOINT INFERENCE OR FORMAL TRAINING**.

## Objective and immutable protocol

This Gate audits and reproduces the historical B2-1 seed42 shared-weight dual-tower model. It first runs the historical checkpoint on the original frozen test database, then performs at most one formal seed42 training run with the unchanged historical configuration. No other seed, model, new15016 record, or final673 asset is used.

The historical B2-1 training assets contain 5,120 train, 1,098 validation, and 1,098 test records (7,316 unique SIDs). This differs from the PM6-complete Layer G manifest, which contains 5,118 / 1,098 / 1,097 records. Strict B2-1 reproduction therefore uses the original 7,316 ASE databases. The cheap-versus-B2-1 paired audit is restricted to their 1,097 common test SIDs and must not claim that the two original training protocols had identical coverage.

## Architecture and optimization

- One shared EquiformerV3 tower is applied separately to donor and acceptor graphs.
- Each side is mean pooled; embeddings are concatenated and passed through a two-layer fusion MLP and energy head.
- 1,075,318 trainable parameters; DFT S0 coordinates; 8 Å cutoff; at most 20 neighbors.
- Historical normalization is fixed at mean 0.800596 eV and standard deviation 0.194270 eV.
- AdamW, batch 8, seed 42, at most 80 epochs, initial learning rate 1e-3, 500-step warmup, milestone decay at 2,000/4,000/6,000 steps, EMA 0.999, gradient clipping 10, FP32.
- Test is excluded from training and checkpoint selection.

The historical trainer's validation method averages batch means equally. Since 1,098 is not divisible by eight, its final two-record batch is overweighted. The historical best checkpoint records validation MAE 0.07783559350755767 under this implementation. Strict historical reproduction retains and reports this limitation; a future corrected protocol must be a separately named experiment.

## Frozen expected result and status rules

The original 09:11 evaluation has 1,098 rows and records MAE 0.07659060508012772 eV and RMSE 0.12215884029865265 eV. The later 13:38 SID-regenerated vector, used by the formal paired comparison, recomputes to MAE 0.07656088892531875 eV and differs from the original vector by up to 0.0082203 eV per prediction. This is a frozen historical provenance discrepancy; it is not resolved by choosing the lower number. Both assets and their roles are retained. Full input paths and SHA-256 values are frozen in `configs/gate1a2_b21_seed42_reproduction_v1.json`.

- `REPRODUCED_STRICT`: frozen checkpoint inference gives ID-aligned predictions equal within serialization tolerance and an identical full-precision metric.
- `REPRODUCED_NUMERIC`: protocol matches and test MAE differs by at most 0.0010 eV.
- `BLOCKED_ASSET`: a required frozen asset or implementation fact is unavailable.
- `FAILED_REPRODUCTION`: confirmed protocol differs by more than 0.0010 eV.

The 0.07020991162281436 eV cheap result remains a historical post-test-selected champion. This Gate may compare the already frozen common prediction pair, but it does not establish a new superiority claim and does not replace the locked project statement `p=0.145`.
