# Input tier contract V1

- Tier 0: canonical SMILES, atom/bond graph, donor/acceptor role and 2D descriptors. Allowed.
- Tier 1.5: PM6 geometry, HOMO, LUMO, gap, and optionally dipole. PM6 dipole requires paired with/without ablation. `pm6_energy_raw` is forbidden.
- Tier 2: DFT S0 geometry and ground-state HOMO/LUMO/gap/dipole. Allowed only in an explicitly named high-cost tier.
- Tier 3: TDDFT/Multiwfn results. Supervision only; never main-model input.

IDs may be used for grouping and audit, not predictive features. Method, basis, source, parser version, root, termination and file paths are provenance/control fields. All preprocessing, memory and statistics are train-only after immutable grouping.
