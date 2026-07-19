# Gate 0-C OOD severity

Morgan: radius=2, nBits=2048, useChirality=True. Similarity is diagnostic only.

## iid_group_seed42_v1

- val: full_molecule median=0.8803, p05=0.7651; donor_component median=1.0000, p05=1.0000; acceptor_component median=1.0000, p05=1.0000
- test: full_molecule median=0.8830, p05=0.7640; donor_component median=1.0000, p05=1.0000; acceptor_component median=1.0000, p05=1.0000

## donor_cold_v1

- val: full_molecule median=0.8230, p05=0.6433; donor_component median=0.6897, p05=0.1414; acceptor_component median=1.0000, p05=1.0000
- test: full_molecule median=0.8192, p05=0.5817; donor_component median=0.5532, p05=0.1754; acceptor_component median=1.0000, p05=1.0000

## acceptor_cold_v1

- val: full_molecule median=0.8288, p05=0.7250; donor_component median=1.0000, p05=1.0000; acceptor_component median=0.5193, p05=0.2414
- test: full_molecule median=0.8158, p05=0.6226; donor_component median=1.0000, p05=1.0000; acceptor_component median=0.4828, p05=0.2034

## pair_cold_v1

- val: full_molecule median=0.8830, p05=0.7613; donor_component median=1.0000, p05=1.0000; acceptor_component median=1.0000, p05=1.0000
- test: full_molecule median=0.8818, p05=0.7639; donor_component median=1.0000, p05=1.0000; acceptor_component median=1.0000, p05=1.0000

## both_cold_external_test_v1

- val: full_molecule median=0.8850, p05=0.7607; donor_component median=1.0000, p05=1.0000; acceptor_component median=1.0000, p05=1.0000
- test: full_molecule median=0.7129, p05=0.5000; donor_component median=0.3333, p05=0.1205; acceptor_component median=0.7805, p05=0.4589
- buffer: full_molecule median=0.8167, p05=0.6697; donor_component median=1.0000, p05=0.2904; acceptor_component median=1.0000, p05=0.7168

## full_scaffold_cold_v1

Overall, 7,564 of 10,005 full Murcko scaffolds (75.60%) are singletons. This split must not be over-described as deep scaffold extrapolation.


- val: full_molecule median=0.8421, p05=0.7297; donor_component median=1.0000, p05=1.0000; acceptor_component median=1.0000, p05=1.0000
- test: full_molecule median=0.8478, p05=0.7353; donor_component median=1.0000, p05=1.0000; acceptor_component median=1.0000, p05=1.0000
