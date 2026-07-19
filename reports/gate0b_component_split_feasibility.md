# Gate 0-B component and split feasibility

No split was generated.

- donor IDs/structures: 155/154
- acceptor IDs/structures: 354/352
- ID pairs/structure pairs: 15016/14861
- donor/acceptor same-ID multi-structure: 0/0
- donor/acceptor alias groups: 1/2

Cold splits use component structure groups, never numeric IDs. Alias IDs collapse before OOD assignment.

```json
{
  "donor_structure": {
    "unique": 154,
    "singleton_entities": 0,
    "low_frequency_le5_entities": 3,
    "median_records_per_entity": 82.5,
    "p95_records_per_entity": 192.35,
    "max_records_per_entity": 216
  },
  "acceptor_structure": {
    "unique": 352,
    "singleton_entities": 1,
    "low_frequency_le5_entities": 4,
    "median_records_per_entity": 33.0,
    "p95_records_per_entity": 91.0,
    "max_records_per_entity": 165
  },
  "ordered_structure_pair": {
    "unique": 14861,
    "singleton_entities": 14706,
    "low_frequency_le5_entities": 14861,
    "median_records_per_entity": 1.0,
    "p95_records_per_entity": 1.0,
    "max_records_per_entity": 2
  },
  "full_structure": {
    "unique": 14639,
    "singleton_entities": 14267,
    "low_frequency_le5_entities": 14639,
    "median_records_per_entity": 1.0,
    "p95_records_per_entity": 1.0,
    "max_records_per_entity": 4
  },
  "donor_scaffold": {
    "unique": 117,
    "singleton_entities": 0,
    "low_frequency_le5_entities": 3,
    "median_records_per_entity": 98.0,
    "p95_records_per_entity": 369.99999999999994,
    "max_records_per_entity": 947
  },
  "acceptor_scaffold": {
    "unique": 285,
    "singleton_entities": 0,
    "low_frequency_le5_entities": 3,
    "median_records_per_entity": 43.0,
    "p95_records_per_entity": 99.60000000000002,
    "max_records_per_entity": 416
  },
  "full_scaffold": {
    "unique": 10005,
    "singleton_entities": 7564,
    "low_frequency_le5_entities": 9782,
    "median_records_per_entity": 1.0,
    "p95_records_per_entity": 4.0,
    "max_records_per_entity": 40
  },
  "unique_donor_ids": 155,
  "unique_acceptor_ids": 354,
  "unique_id_pairs": 15016,
  "donor_id_multi_structure_count": 0,
  "acceptor_id_multi_structure_count": 0,
  "donor_structure_alias_groups": 1,
  "acceptor_structure_alias_groups": 2,
  "scaffold_parse_failures": {
    "donor": 0,
    "acceptor": 0,
    "full": 0
  }
}
```

donor-cold: FEASIBLE_BY_STRUCTURE_GROUP.
acceptor-cold: FEASIBLE_BY_STRUCTURE_GROUP.
pair-cold: FEASIBLE after alias collapse.
both-cold: Gate 0-C must solve group-disjoint assignment and verify power.
scaffold-cold: FEASIBLE_BY_MURCKO_GROUP.
time/prospective: **BLOCKED_NO_TRUSTED_TIMESTAMP**.
