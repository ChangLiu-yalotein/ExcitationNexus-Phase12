# ExcitationNexus Phase 12

Reproducible code, immutable split manifests, contracts, audit reports, and small evidence files for ExcitationNexus Phase 12.

## Current research state

Current state: `BENCHMARK_CONSOLIDATED_READY_FOR_MAIN_MODEL`.

- The frozen dataset contains 15,016 calculation records and 14,639 canonical structure groups.
- The reported target is **J_eh_screened_eV_eps3p5 proxy**. It is not experimental exciton binding energy, catalytic efficiency, or measured photocatalytic activity.
- The strongest reliable new15016 IID baseline is XGBoost-C0 at structure-group-macro MAE 0.084181 eV.
- The small M3-Merged and M3-DAU-Shared 3D ensembles are trained baselines, not ReMEI-Net, and are weaker than XGBoost-C0.
- Acceptor-cold is the confirmed component-OOD failure mode. Pair-cold means unseen donor-acceptor combinations of seen components.
- The 16 Gate 3 candidates remain frozen as `EXPLORATORY_BASELINE_SHORTLIST_FROZEN`. Experimental progression is paused until the main-model benchmark is complete.

## Benchmark entry points

- [Consolidated model benchmark](reports/gate2g0_model_benchmark_consolidation.md)
- [Checkpoint inventory](reports/gate2g0_checkpoint_inventory.md)
- [Roadmap gap audit](reports/gate2g0_roadmap_gap_audit.md)
- [Historical benchmark CSV](data_registry/gate2g0_historical_benchmark.csv)
- [new15016 IID benchmark CSV](data_registry/gate2g0_new15016_iid_benchmark.csv)
- [new15016 OOD benchmark CSV](data_registry/gate2g0_new15016_ood_benchmark.csv)
- [UQ benchmark CSV](data_registry/gate2g0_uq_benchmark.csv)
- [Model registry](data_registry/gate2g0_model_registry.json)

Historical Layer G and new15016 are separate benchmark ledgers. Results from different data and split protocols must not be ranked as model improvements.

## Implemented and missing models

Implemented assets include historical cheap/B2 reproduction evidence, new15016 median/Ridge/XGBoost baselines, six small M3 3D checkpoints, validation-only representation experiments, training-only multitask and multifidelity cross-fit assets, and the Gate 3-A1 deployment/stability XGBoost models.

Not implemented: Chemprop v2 D-MPNN, PaiNN/TensorNet2, a formal EquiformerV3 molecular baseline on new15016, an explicit D/A interface cross-edge model, PM6 FiLM/gating, ReMEI-Net, and the A0-A10 parameter-matched ablation matrix.

## Repository boundary

This repository excludes raw PM6/DFT/TDDFT data, the primary Parquet table, sealed evaluation assets, checkpoints, full predictions, candidate structures, SMILES, and large runtime caches. They remain on the research server. Registries bind local assets through paths, schemas, versions, and SHA-256.

Checkpoints are intentionally absent from GitHub. `data_registry/gate2g0_checkpoint_inventory.csv` records the audited local path, hash, size, loadability, and smoke scope.

## Verification

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
/home/changliu/miniconda3/envs/ML/bin/python -m pytest -q tests
sha256sum -c data_registry/gate2g0_sha256.txt
```

Safe CPU-only inventory smoke, which does not train or call a test evaluator:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /home/changliu/miniconda3/envs/ML/bin/python \
  scripts/gate2g0_consolidate_model_benchmarks.py
/home/changliu/miniconda3/envs/ML/bin/python scripts/gate2g0_ubj_audit.py
/home/changliu/miniconda3/envs/ML/bin/python scripts/gate2g0_supplemental.py
```

Gate checkpoint publication occurs only after DONE status and all hash, test, secret, large-file, and Git checks pass.
