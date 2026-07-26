# Gate 2-G1C unified strong 2D benchmark

Decision: **GATE2G1C_BLOCKED_CHEMPROP_V2_UNAVAILABLE_XGB_SUBSET_DONE**.

XGBoost subset completed on 25008 eligible records. Chemprop v2 D-MPNN arms were not run because official Chemprop v2 is not available in the local environment; Chemprop v1 was not substituted.

## XGBoost aggregate metrics

| protocol | arm | records | groups | group_macro_mae | record_mae | source_cohort_macro_mae | worst_source_mae | source_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| donor_cold_5fold | xgb_c0_blind | 25008 | 24543 | 0.096300 | 0.096330 | 0.093467 | 0.097927 | 0.013050 |
| donor_cold_5fold | xgb_c0_observable_provenance | 25008 | 24543 | 0.088714 | 0.088728 | 0.087204 | 0.091039 | 0.005822 |
| leave_one_cohort_out_external2698 | xgb_c0_blind | 2680 | 2641 | 0.107711 | 0.107469 | 0.107469 | 0.107469 | 0.000000 |
| leave_one_cohort_out_external2698 | xgb_c0_observable_provenance | 2680 | 2641 | 0.239503 | 0.238495 | 0.238495 | 0.238495 | 0.000000 |
| leave_one_cohort_out_new15016 | xgb_c0_blind | 15016 | 14639 | 0.176136 | 0.177351 | 0.177351 | 0.177351 | 0.000000 |
| leave_one_cohort_out_new15016 | xgb_c0_observable_provenance | 15016 | 14639 | 0.169222 | 0.170307 | 0.170307 | 0.170307 | 0.000000 |
| leave_one_cohort_out_old7316 | xgb_c0_blind | 7312 | 7298 | 0.154559 | 0.154671 | 0.154671 | 0.154671 | 0.000000 |
| leave_one_cohort_out_old7316 | xgb_c0_observable_provenance | 7312 | 7298 | 0.099645 | 0.099741 | 0.099741 | 0.099741 | 0.000000 |
| pair_cold_5fold | xgb_c0_blind | 25000 | 24535 | 0.086744 | 0.086710 | 0.081564 | 0.091804 | 0.027072 |
| pair_cold_5fold | xgb_c0_observable_provenance | 25000 | 24535 | 0.080774 | 0.080736 | 0.075465 | 0.084883 | 0.022210 |
| scaffold_cold_5fold | xgb_c0_blind | 25008 | 24543 | 0.088739 | 0.088790 | 0.084461 | 0.094502 | 0.025088 |
| scaffold_cold_5fold | xgb_c0_observable_provenance | 25008 | 24543 | 0.082894 | 0.082956 | 0.078699 | 0.086079 | 0.018090 |
| source_stratified_acceptor_cold_5fold | xgb_c0_blind | 25008 | 24543 | 0.094503 | 0.094394 | 0.093381 | 0.099788 | 0.012464 |
| source_stratified_acceptor_cold_5fold | xgb_c0_observable_provenance | 25008 | 24543 | 0.088678 | 0.088535 | 0.088017 | 0.090066 | 0.004711 |
| unified_grouped_5fold_oof | xgb_c0_blind | 25008 | 24543 | 0.086761 | 0.086855 | 0.081892 | 0.092238 | 0.026869 |
| unified_grouped_5fold_oof | xgb_c0_observable_provenance | 25008 | 24543 | 0.081030 | 0.081093 | 0.076148 | 0.084904 | 0.020889 |


final673 labels, Gate 3 candidate assets, and quarantined records were not used.

## Interpretation

The observable-provenance XGBoost arm improves unified grouped OOF and acceptor-cold point estimates, but it fails the deployment-admission logic because Chemprop arms are blocked and leave-one-cohort-out external2698 degrades sharply. Treat this as source/pipeline calibration evidence only.
