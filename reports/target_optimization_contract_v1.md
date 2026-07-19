# Target optimization contract V1

The primary optimization target is `tddft_coulomb_attraction_eV_eps3p5_proxy`, reported as **J_eh_screened_eV_eps3p5 proxy**. It is not experimental exciton binding energy.

The optimization graph contains one primary, twelve non-redundant secondary, and four masked auxiliary tasks. `tddft_wavelength_nm`, raw Coulomb au, and raw Coulomb eV are report-only deterministic transforms and never enter total multitask loss. `Q_A_to_D`, `net_CT_D_to_A`, `pm6_energy_raw`, and provenance/control fields remain disabled.

Gate 0-D weights are `SMOKE_ONLY_NOT_SCIENTIFICALLY_TUNED` and are not frozen scientific hyperparameters.
