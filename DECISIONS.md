# Phase 12 decisions

- No training or quantum calculation in Gate 0-A.
- A GPU is FREE only with no compute PID, <1 GiB used, and near-zero utilization.
- Bind every experiment with `CUDA_VISIBLE_DEVICES`; one independent seed/experiment per free GPU; single-GPU smoke first; no default seven-card DDP; recheck before launch; never kill non-project processes.
- Preserve raw data and both plans unchanged.
- Treat target as `J_eh_screened_eV_eps3p5 proxy`, never direct experimental Eb.
- Disable Tier 3 inputs, equivalent target transforms, `pm6_energy_raw`, and full-dataset statistics.
- Preserve final673 sealing: aggregate overlaps only; no per-sample blind membership artifact.
- Use normalized ID plus RDKit canonical structure for leakage boundaries.
- Split generation is blocked pending structure replicate/dedup policy and historical benchmark correction.

## Gate 0-B decisions

- 15,016 calculation records and 14,639 canonical structures are distinct counts; delete none.
- Use V1 structure/component hashes, not rows or numeric IDs.
- Keep one structure group wholly in one partition.
- Replicate recommendation `RETAIN_REPLICATES_WITH_GROUP_WEIGHT`; retained rows use `group_weight=1/group_size`.
- Preserve official metrics and add structure-purged sensitivity metrics.
- external/final overlap is a benchmark limitation, not a new-data blocker.
- No filesystem-mtime time split; `BLOCKED_NO_TRUSTED_TIMESTAMP`.

## Gate 0-C decisions

- Freeze preregistration v1 before any solver run; corrections require explicit v2.
- Preserve all 15,016 rows; isolate one external/legacy overlap as `historical_quarantine`.
- Force all 17 old-training overlaps to train; never merge old7316 into Phase 12 new-only training.
- Freeze six target-blind grouped protocols after independent reproducibility checks.
- Treat pair-cold as unseen-pair/seen-components, not strong component OOD.
- Use seen-component validation for both-cold; keep cross-component rows in explicit buffer.
- Do not post-hoc rebalance v1 from target distributions or diagnostic similarity.

## Gate 0-D decisions

- Bind loaders to frozen table/manifest hashes; retain the IID filename and record selected seed 123.
- Permit only explicit train/val/test; reject buffer and historical quarantine.
- Keep PM6 dipole disabled by default; DFT requires explicit tier2.
- Optimize one primary, twelve secondary, and four masked auxiliary targets; deterministic transforms are report-only.
- Fit train-only group-weighted normalization per split.
- Preserve unknown roles: for 387 no-donor-labelled records, never infer donor atoms; use unknown pooling and role-presence flags.
- TinyRoleAware3D is plumbing-only; smoke weights are not scientific hyperparameters.
