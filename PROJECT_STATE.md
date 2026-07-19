# Phase 12 project state

Current stage: **GATE0D_DONE**; Gate 1-A baseline reproduction is ready under a new instruction.

## Frozen facts

- 15,016 complete PM6+DFT+TDDFT calculation records remain intact.
- These records form 14,639 canonical structure groups.
- Duplicate distribution is 14,267 singleton, 369 doubleton, 1 triplet, and 2 quadruplet groups.
- All 372 duplicate groups have resolved atom/role/bond geometry correspondence; PDB/JSON checks passed.
- Replicate policy is `RETAIN_REPLICATES_WITH_GROUP_WEIGHT`.
- new15016 vs final673 aggregate intersection is ID=0 and structure=0.
- external2698/final673 share 18 structures and old7316/external2698 share 17; these are historical benchmark corrections.
- Gate 0-C created only frozen split assets; no training, CUDA computation, final-label access, or raw-data modification occurred.

## Gate 0-C result

Six immutable target-blind grouped splits are frozen in `data_registry/SPLIT_REGISTRY_V1_FROZEN.json`. Every split covers all 15,016 records; one record is historical quarantine and all 17 old-training-overlap records are train-only. Independent leakage checks, repeat generation, and shuffled-input generation passed. Both-cold retains 78.0767% in train/val/test with 3,291 explicit buffer records. Time/prospective split remains `BLOCKED_NO_TRUSTED_TIMESTAMP`.

## Gate 0-D result

- 33/33 CPU tests passed.
- Six full-table joins, train-only group-weighted normalization registries, and 72 PM6/DFT graph parses passed.
- Tiny plumbing model CPU forward/backward and translation/rotation/permutation invariance passed.
- Physical GPU 0 completed two FP32 batches with finite/nonzero gradients and a real AdamW parameter update.
- 387 records have explicit unknown atoms and no donor-labelled atom; unknown is never inferred as donor.
- Code snapshot SHA-256: `432cebe0fdd1088ec798fc130968fab0f37e3dd635deaa3833f29b104649b392`.

No formal epoch, model comparison, checkpoint selection, final673 access, or frozen split modification occurred.

Project and EquiformerV3 directories are not Git worktrees; source commit provenance remains unavailable and must not be fabricated.
