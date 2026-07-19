# Gate 0-B structure governance

Status: **DONE**.

- calculation records: **15,016** complete PM6+DFT+TDDFT records.
- canonical structures: **14,639** RDKit V1 groups.
- raw records deleted: 0.

## Hash and duplicate audit

- V1/source exact hash matches: 0/15,016.
- Source hashes stored explicit-H strings: 15,016/15,016.
- Partitions equivalent in both directions: **True**.
- Size distribution: {"1": 14267, "2": 369, "3": 1, "4": 2}
- Duplicate groups/records/extra: 372/749/377
- Role/atom-count/sidecar/PDB-JSON inconsistent groups: 219/0/0/0
- Geometry UNRESOLVED groups: 0; RMSD max-group summary: {"count": 372.0, "mean": 0.21758462389950908, "std": 1.01278910674903, "min": 2.230853662118533e-07, "50%": 0.0010794863432472643, "90%": 0.2991091862096972, "95%": 0.7994360917268841, "99%": 5.506373140759095, "max": 10.298962795075514}. D81_A28 in duplicate group: False.

V1 hashes `RemoveHs` canonical strings; source hashes explicit-H strings. Equal counts were not treated as proof. RMSD uses index+element+role+bond correspondence or complete role-aware isomorphism; truncation is UNRESOLVED.

## Target dispersion and policy

Primary: **J_eh_screened_eV_eps3p5 proxy**, not experimental Eb.

- Range bins: {"le_1e-6_eV": 32, "gt_1e-6_le_1e-3_eV": 212, "gt_1e-3_le_1e-2_eV": 76, "gt_1e-2_eV": 52}
- Primary meaningful groups: 128; all enabled meaningful group-target pairs: 3370.
- Raw and screened Coulomb are deterministic transforms, not independent evidence.

### Quantitative comparison

- deterministic representative: 14,639 training-view rows, removes 377 repeated rows; only 12/372 duplicate groups satisfy the strict role+geometry+target rule.
- retain replicates + group weight: 15,016 rows, effective total weight 14639; duplicate rows sum to effective weight 372.
- group target aggregation: 14,639 structure rows, but 219 role-aware-inconsistent groups and 330 groups with meaningful target disagreement make global averaging unsafe.

Recommendation: **RETAIN_REPLICATES_WITH_GROUP_WEIGHT** because role/geometry uncertainty or meaningful target dispersion exists; aggregation could erase distinct records. Keep groups in one partition. Retained replicates require `group_weight=1/group_size` and record-level plus structure-group-macro metrics.
