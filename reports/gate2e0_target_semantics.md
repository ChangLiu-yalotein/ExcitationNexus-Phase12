# Gate 2-E0 target semantics

The primary remains `J_eh_screened_eV_eps3p5 proxy`; it is not experimental Eb or catalytic efficiency.

The 12 candidate secondary fields were traced to the frozen schema and TDDFT property/excitation JSON semantics. Eleven remain optimization candidates. `tddft_t_index_angstrom` is report-only redundant because the IID-train source values satisfy `t = D - H_CT` within 0.001 Å (source rounding).

Wavelength and raw Coulomb au/eV remain report-only deterministic. No member of the primary Coulomb unit/dielectric family is admitted as an auxiliary loss.

The four fragment fractions remain masked-only: missing values are never imputed, and donor+acceptor does not equal one for every jointly observed record, so unknown/unassigned contribution is preserved rather than erased.

## Fraction assignment audit

- hole: jointly observed 5376; median unassigned 1.11e-16; P99 |unassigned| 0.0665169; max 0.619151; strict complement=False.
- electron: jointly observed 5376; median unassigned 1.11e-16; P99 |unassigned| 0.0215921; max 0.376607; strict complement=False.
