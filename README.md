# ExcitationNexus Phase 12

Reproducible code, immutable split manifests, contracts, audit reports, and small evidence files for ExcitationNexus Phase 12.

## Repository boundary

This repository intentionally excludes raw PM6/DFT/TDDFT files, the primary Parquet table, sealed evaluation assets, checkpoints, and large runtime outputs. Those assets remain on the research server and Google Drive. Their identities are bound through paths, schemas, versions, and SHA-256 registries.

Current research state: `GATE0D_DONE`. Gate 0-C froze six zero-leakage grouped splits; Gate 0-D established the reusable data pipeline, target firewall, group-weighted normalization/loss/metrics, and CPU/single-GPU smoke tests.

## Local verification

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
/home/changliu/miniconda3/envs/ML/bin/python -m pytest -q tests
sha256sum -c data_registry/gate0c_split_sha256.txt
sha256sum -c data_registry/gate0d_code_snapshot_sha256.txt
```

Gate checkpoint publication is performed only after DONE status with `scripts/git_checkpoint.sh`.
