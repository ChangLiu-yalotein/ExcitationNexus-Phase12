# Historical identity and overlap audit

Identity was checked first by normalized molecule ID (`D-3_A-1` → `D3_A1`), then by RDKit canonical structure after explicit-H removal. RDKit version: 2025.09.6. The 64,543-row structure registry was parsed without malformed rows; all 10,687 historical IDs had SMILES.

| Comparison | ID overlap | Canonical-structure overlap | Row-level combined identity |
|---|---:|---:|---:|
| new15016 vs old7316 | 0 | 17 | 17 new rows |
| new15016 vs external2698 | 0 | 1 | 1 new row |
| new15016 vs final673 | 0 | 0 | 0 new rows |
| new15016 vs legacy3371 | 0 | 1 | 1 new row |
| old7316 vs external2698 | 0 | 17 structures | 17 rows on each side |
| external2698 vs final673 | 0 | 18 structures | 18 rows on each side |
| legacy3371 vs external2698 | 2,698 | 2,659 structures | all 2,698 external rows; 2,716 legacy rows |
| legacy3371 vs final673 | 673 | 670 structures | all 673 final rows; 691 legacy rows |

The 3,371 legacy IDs partition exactly by ID into external2698 + final673. Structure duplicates explain why the unique-structure and directional legacy-row counts differ.

## Interpretation

- No final673 identity occurs in new15016 by either method.
- One new row is structurally identical to external-dev and legacy; it is forbidden for new training/model selection.
- Seventeen new rows duplicate old training structures; they are not independent additions and must be structure-grouped/deduplicated.
- More seriously, external-dev and final-blind share 18 canonical structures under different IDs. Therefore the historical final-blind boundary has a structure-level leakage risk even though ID sets are disjoint.
- The sealed set was accessed only as an ID column in memory. No blind ID, SMILES, label, row, or per-sample membership was emitted. The public membership file keeps `in_final673=REDACTED_SEALED_SET`; otherwise eligible rows remain `SEALED_CHECK_REQUIRED`.

Evidence: `logs/gate0a_overlap.json` and `manifests/new15016_historical_membership.csv`.
