# Gate 0-D target and loss contract

Optimization uses one primary, twelve secondary, and four masked auxiliary tasks. The primary is `tddft_coulomb_attraction_eV_eps3p5_proxy`, reported only as **J_eh_screened_eV_eps3p5 proxy**.

`tddft_wavelength_nm`, Coulomb au, and unscreened Coulomb eV are report-only deterministic transforms. They never enter total loss. `Q_A_to_D`, `net_CT_D_to_A`, `pm6_energy_raw`, and provenance/control fields remain disabled.

For each optimization task:

`loss_t = sum(group_weight * mask * base_loss) / sum(group_weight * mask)`

Empty tasks are skipped safely. MSE, MAE, and SmoothL1 are supported. Loss operates in train-normalized target space; metrics use inverse-transformed original units.

Smoke weights are primary 1.0, all secondary tasks together 0.5, and all auxiliary tasks together 0.1. Status: `SMOKE_ONLY_NOT_FROZEN_FOR_EXPERIMENTS`.

The synthetic weighted masked MAE equals the hand-calculated 2.5. An all-empty mask is finite and differentiable, and report-only predictions cannot change total loss.
