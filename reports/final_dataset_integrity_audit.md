# Final15016 dataset integrity audit

## Result

File-level closed-loop integrity passed, but identity uniqueness failed. The dataset is complete as 15,016 calculation records; it is not 15,016 unique canonical molecular structures.

- PM6/DFT/TDDFT directory IDs: 15,016 each; sets identical.
- Parquet/CSV rows: 15,016 each.
- `molecule_id`: unique.
- `canonical_smiles_sha256`: **not unique**; 14,639 unique hashes.
- Duplicate structure groups: 372; 749 rows participate; 377 rows exceed one-per-structure; maximum group size 4.
- RDKit standardization (`RemoveHs`, canonical isomeric SMILES) gives the same 14,639 unique structures and zero parse failures.
- Upload manifest: 165,176 rows; 0 unknown fidelity, duplicate destination, missing file, empty file, or size mismatch.
- Primary label non-null: 15,016/15,016.
- Sidecar flag: only `D81_A28=True`.
- PM6 semantics flag: 15,016/15,016 unresolved; `pm6_energy_raw` remains disabled.
- Fragment fractions: 7,750/15,016 = 51.6116%.
- `Q_A_to_D` and `net_CT_D_to_A`: 0/15,016; allowed null and disabled.

Evidence: `logs/gate0a_integrity.json`, upstream `tables/verify_server_dataset.py` returned 0 errors, and `data_registry/final15016_sha256.txt`.

## Gate consequence

Do not freeze row-random splits. First define a canonical-structure group policy so all 749 duplicate rows remain in one partition, and decide whether repeated calculations are retained as replicates, aggregated, or reduced by a deterministic representative rule. Raw data remains unchanged.
