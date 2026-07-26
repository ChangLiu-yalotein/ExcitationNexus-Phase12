# Gate 2-G1A target harmonization

| source_cohort | records | target_field | epsilon | unit | semantics | status |
| --- | --- | --- | --- | --- | --- | --- |
| new15016 | 15016 | tddft_coulomb_attraction_eV_eps3p5_proxy | 3.500000 | eV | J_eh_screened_eV_eps3p5 proxy | COMPATIBLE_PROXY_NEW_PIPELINE |
| old7316 | 7316 | coulomb_attraction_screened_eV / eb_screened_eV | 3.500000 | eV | legacy screened proxy; same nominal epsilon but parser/method lineage differs | SOURCE_AWARE_REQUIRED |
| external2698 | 2698 | label_eV | 3.500000 | eV | legacy external screened proxy; historical model-selection use means not blind | SOURCE_AWARE_REQUIRED |
| final673 | 673 | REDACTED_NOT_READ_FOR_GATE2G1A | 3.500000 | eV | final confirmation labels remain sealed | SEALED_CONFIRMATION_ONLY |

Conclusion: the target is nominally the same screened Coulomb proxy, but parser/method lineage differs across old/external/new cohorts. Future training must retain source or method information.
