# Gate 2-E0 missingness and coverage

All figures are protocol-local. Train drives task admission; validation is used only to establish future evaluability. Missing auxiliary labels are masked and never zero/mean-imputed or exposed as model inputs.

## IID coverage

| Task | Train records | Train group-weighted completeness | Validation records | Validation group-weighted completeness |
|---|---:|---:|---:|---:|
| `tddft_excitation_energy_ev` | 10387 | 100.0000% | 2309 | 100.0000% |
| `tddft_oscillator_strength` | 10387 | 100.0000% | 2309 | 100.0000% |
| `tddft_transition_dipole_au` | 10387 | 100.0000% | 2309 | 100.0000% |
| `tddft_Sm` | 10387 | 100.0000% | 2309 | 100.0000% |
| `tddft_Sr` | 10387 | 100.0000% | 2309 | 100.0000% |
| `tddft_D_index_angstrom` | 10387 | 100.0000% | 2309 | 100.0000% |
| `tddft_H_CT_angstrom` | 10387 | 100.0000% | 2309 | 100.0000% |
| `tddft_t_index_angstrom` | 10387 | 100.0000% | 2309 | 100.0000% |
| `tddft_HDI` | 10387 | 100.0000% | 2309 | 100.0000% |
| `tddft_EDI` | 10387 | 100.0000% | 2309 | 100.0000% |
| `tddft_Q_D_to_A_au` | 10387 | 100.0000% | 2309 | 100.0000% |
| `tddft_dipole_change_norm_au` | 10387 | 100.0000% | 2309 | 100.0000% |
| `tddft_hole_on_donor_fraction` | 5376 | 51.7955% | 1167 | 50.8884% |
| `tddft_hole_on_acceptor_fraction` | 5376 | 51.7955% | 1167 | 50.8884% |
| `tddft_electron_on_donor_fraction` | 5376 | 51.7955% | 1167 | 50.8884% |
| `tddft_electron_on_acceptor_fraction` | 5376 | 51.7955% | 1167 | 50.8884% |

All 12 secondary targets are fully observed in every protocol train/validation partition. The four fragment targets remain approximately 51.6% observed and retain explicit masks. Protocol- and identity-level counts are frozen in `logs/gate2e0_missingness.json`; no raw identity or target values are published.
