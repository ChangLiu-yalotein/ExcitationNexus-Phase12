# Gate 2-E1 auxiliary task results

Auxiliary MAEs use the single frozen official-validation inference. They are descriptive and did not select checkpoints.

## iid

### M11
- `tddft_D_index_angstrom`: MAE 2.4598165, valid 2309
- `tddft_EDI`: MAE 0.47928733, valid 2309
- `tddft_HDI`: MAE 0.54121526, valid 2309
- `tddft_H_CT_angstrom`: MAE 1.0245272, valid 2309
- `tddft_Q_D_to_A_au`: MAE 0.00046126591, valid 2309
- `tddft_Sm`: MAE 0.081218983, valid 2309
- `tddft_Sr`: MAE 0.085350761, valid 2309
- `tddft_dipole_change_norm_au`: MAE 4.6440368, valid 2309
- `tddft_excitation_energy_ev`: MAE 0.075289891, valid 2309
- `tddft_oscillator_strength`: MAE 0.3540003, valid 2309
- `tddft_transition_dipole_au`: MAE 1.3679032, valid 2309

### M15
- `tddft_D_index_angstrom`: MAE 2.4627578, valid 2309
- `tddft_EDI`: MAE 0.47400777, valid 2309
- `tddft_HDI`: MAE 0.53096888, valid 2309
- `tddft_H_CT_angstrom`: MAE 1.0166858, valid 2309
- `tddft_Q_D_to_A_au`: MAE 0.00040762246, valid 2309
- `tddft_Sm`: MAE 0.079992019, valid 2309
- `tddft_Sr`: MAE 0.084506271, valid 2309
- `tddft_dipole_change_norm_au`: MAE 4.6504631, valid 2309
- `tddft_electron_on_acceptor_fraction`: MAE 0.070003371, valid 1167
- `tddft_electron_on_donor_fraction`: MAE 0.070117591, valid 1167
- `tddft_excitation_energy_ev`: MAE 0.067346043, valid 2309
- `tddft_hole_on_acceptor_fraction`: MAE 0.12039909, valid 1167
- `tddft_hole_on_donor_fraction`: MAE 0.12074834, valid 1167
- `tddft_oscillator_strength`: MAE 0.33931541, valid 2309
- `tddft_transition_dipole_au`: MAE 1.3682129, valid 2309

Primary-error strata:
- fragment_missing: n=1142, M15−M11=-0.000139 eV
- fragment_observed: n=1167, M15−M11=+0.000923 eV
- DA_unknown: n=50, M15−M11=-0.001919 eV
- empty_donor_unknown: n=55, M15−M11=+0.004828 eV
- pure_DA: n=2204, M15−M11=+0.000340 eV

## acceptor_cold

### M11
- `tddft_D_index_angstrom`: MAE 2.5654862, valid 2235
- `tddft_EDI`: MAE 0.63156281, valid 2235
- `tddft_HDI`: MAE 0.4086762, valid 2235
- `tddft_H_CT_angstrom`: MAE 1.0155091, valid 2235
- `tddft_Q_D_to_A_au`: MAE 0.00063835564, valid 2235
- `tddft_Sm`: MAE 0.094230467, valid 2235
- `tddft_Sr`: MAE 0.1003739, valid 2235
- `tddft_dipole_change_norm_au`: MAE 4.8550143, valid 2235
- `tddft_excitation_energy_ev`: MAE 0.15174997, valid 2235
- `tddft_oscillator_strength`: MAE 0.41141163, valid 2235
- `tddft_transition_dipole_au`: MAE 1.3148442, valid 2235

### M15
- `tddft_D_index_angstrom`: MAE 2.4338353, valid 2235
- `tddft_EDI`: MAE 0.59767411, valid 2235
- `tddft_HDI`: MAE 0.39403954, valid 2235
- `tddft_H_CT_angstrom`: MAE 0.98498923, valid 2235
- `tddft_Q_D_to_A_au`: MAE 0.00061301073, valid 2235
- `tddft_Sm`: MAE 0.089179162, valid 2235
- `tddft_Sr`: MAE 0.094758678, valid 2235
- `tddft_dipole_change_norm_au`: MAE 4.5877515, valid 2235
- `tddft_electron_on_acceptor_fraction`: MAE 0.12438429, valid 1205
- `tddft_electron_on_donor_fraction`: MAE 0.12485456, valid 1205
- `tddft_excitation_energy_ev`: MAE 0.1459492, valid 2235
- `tddft_hole_on_acceptor_fraction`: MAE 0.10357356, valid 1205
- `tddft_hole_on_donor_fraction`: MAE 0.10436317, valid 1205
- `tddft_oscillator_strength`: MAE 0.38475319, valid 2235
- `tddft_transition_dipole_au`: MAE 1.2827143, valid 2235

Primary-error strata:
- fragment_missing: n=1030, M15−M11=+0.000828 eV
- fragment_observed: n=1205, M15−M11=-0.006576 eV
- DA_unknown: n=23, M15−M11=+0.007494 eV
- empty_donor_unknown: n=13, M15−M11=+0.002452 eV
- pure_DA: n=2199, M15−M11=-0.003309 eV

