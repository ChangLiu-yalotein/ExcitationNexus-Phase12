# Gate 2-G1A method and domain shift

| source_cohort | records | method | basis | program | geometry_fidelity |
| --- | --- | --- | --- | --- | --- |
| external2698 | 2698 | CAM-B3LYP | 6-31G(d) | legacy_external_holdout_manifest | both |
| final673 | 673 | CAM-B3LYP | 6-31G(d) | legacy_external_holdout_manifest | both |
| new15016 | 15016 | CAM-B3LYP | 6-31G(d) | Gaussian 16, Revision C.01 | pm6_opt |
| old7316 | 7316 | CAM-B3LYP | legacy_6-31G(d)_assumed_from_external_protocol | legacy_teacher_table | legacy_dft_pm6 |

Method and parser differences must not be hidden in random splits. Use source-aware or cohort-aware training designs in Gate 2-G1B/G1C.
