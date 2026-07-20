# Gate 2-B hierarchical OOD inference

Status: **GATE2B_DONE_HIERARCHICAL_OOD_AUDIT**.

Diagnostic labels: `ACCEPTOR_OOD_FAILURE_CONFIRMED, BOTH_COLD_LOW_SKILL_WARNING, PM6_ORBITAL_SHIFT_RISK`.

| Protocol | Primary inference unit | Clusters | Power | XGB-C0 identity/two-way MAE | IID difference CI | Skill vs median |
|---|---|---:|---|---:|---|---:|
| iid | structure_group_id_v1 | 2195 | ADEQUATE_CLUSTER_COUNT | 0.084181475 | reference | 0.4061 |
| donor_cold | donor_structure_group_id_v1 | 15 | LOW_CLUSTER_POWER | 0.087608538 | [-0.006089832872154736, 0.014156890282027065] | 0.3278 |
| acceptor_cold | acceptor_structure_group_id_v1 | 36 | ADEQUATE_CLUSTER_COUNT | 0.097319544 | [0.004276044049854374, 0.0236228533437702] | 0.3906 |
| pair_cold | pair_group_id_v1 | 2260 | ADEQUATE_CLUSTER_COUNT | 0.085295590 | [-0.0034141259403917385, 0.005661165665815125] | 0.4060 |
| both_cold | two_way_donor_acceptor | 15 | LOW_CLUSTER_POWER | 0.084398558 | [-0.011610177197209413, 0.013284458299610384] | 0.0695 |
| full_scaffold_cold | full_scaffold_group_id_v1 | 672 | ADEQUATE_CLUSTER_COUNT | 0.082086970 | [-0.007314234838233269, 0.0032548304439715837] | 0.3919 |

Structure-group bootstrap remains a secondary Gate 2-A sensitivity. Cold-protocol inference now clusters on held-out identities. IID-to-OOD comparisons use independent resampling and are descriptive, never paired or causal.
