# Gate 0-C split generation summary

Overall status: **DONE**. Six target-blind grouped split manifests were preregistered, generated, independently verified, and frozen. No training or CUDA computation occurred.

| Split | Status | Train records/groups | Val records/groups | Test records/groups | Buffer | Quarantine |
|---|---:|---:|---:|---:|---:|---:|
| iid_group_seed42_v1 | DONE | 10,387 / 10,248 | 2,309 / 2,195 | 2,319 / 2,195 | 0 | 1 |
| donor_cold_v1 | DONE | 10,530 / 10,218 | 2,234 / 2,196 | 2,251 / 2,224 | 0 | 1 |
| acceptor_cold_v1 | DONE | 10,543 / 10,225 | 2,235 / 2,205 | 2,237 / 2,208 | 0 | 1 |
| pair_cold_v1 | DONE | 10,387 / 10,257 | 2,319 / 2,190 | 2,309 / 2,191 | 0 | 1 |
| both_cold_external_test_v1 | DONE | 9,345 / 9,195 | 1,792 / 1,622 | 587 / 587 | 3,291 / 3,234 | 1 |
| full_scaffold_cold_v1 | DONE | 10,511 / 10,246 | 2,254 / 2,195 | 2,250 / 2,197 | 0 | 1 |

Groups and effective weights are equal by construction; all retained records use `group_weight=1/structure_group_size`. All 15,016 records occur exactly once in every manifest. The one external/legacy-overlap record is `historical_quarantine`; the 17 old-training-overlap records are train-only.

The IID and pair-cold record ratios deviate by about 0.8 percentage points from 70/15/15 while their effective structure-weight ratios are close to target. Identity constraints and the frozen dual record/weight objective take precedence; v1 was not post-hoc altered.

Although the IID manifest name retains `seed42`, the preregistered candidate search selected seed 123 by the frozen graph/count objective. No target statistic was used.

Time/prospective split remains `BLOCKED_NO_TRUSTED_TIMESTAMP`.
