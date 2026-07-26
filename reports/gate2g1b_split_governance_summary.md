# Gate 2-G1B source-aware split governance

Decision: **GATE2G1B_SOURCE_AWARE_SPLITS_FROZEN_READY_FOR_MAIN_MODEL**.

This gate freezes split governance only. It performs no training, GPU use, prediction, candidate rescoring, or final673 label access.

## Counts

| metric | value |
| --- | --- |
| status | GATE2G1B_SOURCE_AWARE_SPLITS_FROZEN_READY_FOR_MAIN_MODEL |
| universe_records | 25703 |
| development_registry_records_before_quarantine | 25030 |
| final673_records | 673 |
| final673_structure_groups | 670 |
| final673_structure_overlap_quarantine_groups | 22 |
| final673_structure_overlap_quarantine_development_records | 22 |
| final673_structure_overlap_quarantine_development_records_by_source | {'old7316': 4, 'external2698': 18} |
| eligible_development_records_after_quarantine | 25008 |
| eligible_development_unique_structures_after_quarantine | 24543 |
| legacy3371_policy | membership alias only; not appended |
| rdkit_version | 2025.09.6 |

## Assignment hashes

| protocol | assignment_sha256 |
| --- | --- |
| unified_grouped_5fold_oof | bfe90ae4e768ec754d0dfdc482dbd8da9e57c157e1c1a45af713ed9a0b9cdd8e |
| source_stratified_acceptor_cold_5fold | d5a3fdaa9e3c44a2825eae0cd858740e17705ed51db99296965a8e9aa0f86471 |
| donor_cold_5fold | aee0312deafd2dba5d7d17a5bf8fac0759dbfa17ae765887d4a2d9bc21cfb7c4 |
| scaffold_cold_5fold | 9553bed5487b6ded7dcd70dbb3a0d9f3226c3231d3ec114fa86375d7e5d0663c |
| pair_cold_5fold | c4933c5703925d1a5641516c347129357f1d6a9dad080f85700abd3e52007c9d |
| both_cold_5fold | 4b765d511db3bc35240ec3f257e3517f1dfa82662b67261bee07ceb143b83945 |
| leave_one_cohort_out | 421cb4999683cbfd288624d93929dfd25d4b12df3111c018113bfbd8f26bd06c |
| final673_structure_overlap_quarantine | 9f0b65bf35224a874b9d81bf681af28d19769cecbb85e5e71c2eb03f3f7933b7 |
| eligible_development_after_final673_quarantine | 1031ffe7493e23e9be3682ba1051cf9c268dd5764445b7bd49c561d991e71806 |
