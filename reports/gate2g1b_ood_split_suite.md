# Gate 2-G1B OOD split suite

| protocol | metric | value |
| --- | --- | --- |
| both_cold_5fold | retained_validation_records | 5034 |
| both_cold_5fold | buffer_records | 19974 |
| both_cold_5fold | retained_fraction | 0.201296 |
| both_cold_5fold | retained_donors | 158 |
| both_cold_5fold | retained_acceptors | 355 |
| pair_cold_5fold | retained_validation_records | 25000 |
| pair_cold_5fold | buffer_insufficient_component_support_records | 8 |
| pair_cold_5fold | min_test_donor_train_support | 1 |
| pair_cold_5fold | min_test_acceptor_train_support | 1 |
| leave_one_cohort_out_external2698 | local_structure_quarantine_records | 18 |
| leave_one_cohort_out_new15016 | local_structure_quarantine_records | 18 |
| leave_one_cohort_out_old7316 | local_structure_quarantine_records | 34 |

Acceptor-cold is the primary OOD endpoint. Pair-cold is seen-components/unseen-combination and must not be described as strong component OOD. Both-cold may leave a buffer that is not a Dataset.
