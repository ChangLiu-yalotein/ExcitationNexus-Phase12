# Target contract V1

Primary:

- Source column: `tddft_coulomb_attraction_eV_eps3p5_proxy`
- Reporting name: `J_eh_screened_eV_eps3p5 proxy`
- Unit: eV
- Interpretation: fixed-dielectric screened electron–hole Coulomb-attraction proxy; not experimental Eb.

Secondary (100% complete): excitation energy, wavelength, oscillator strength, transition-dipole norm, raw Coulomb attraction, Sm, Sr, D index, H_CT, t index, HDI, EDI, Q_D_to_A, and dipole-change norm.

Masked auxiliary (7,750/15,016): hole-on-donor, hole-on-acceptor, electron-on-donor, and electron-on-acceptor fractions. Loss is computed only where labels are non-null.

Disabled: `Q_A_to_D`, `net_CT_D_to_A`, `pm6_energy_raw`, semantic-unresolved fields, and provenance/control fields. The primary label, raw Coulomb values, and every equivalent transform are forbidden as model inputs.
