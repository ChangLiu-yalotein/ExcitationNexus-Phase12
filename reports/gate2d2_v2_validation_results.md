# Gate 2-D2 v2 validation results

All values are validation-only hierarchical MAE in eV under each protocol's frozen inference unit. No test artifact was read.

| Protocol | C0-512 A | MF-Full-RP512 B | MF-Role-RP512 C |
|---|---:|---:|---:|
| acceptor_cold | 0.096528672 | 0.098287554 | 0.095254274 |
| both_cold | 0.084448152 | 0.090194769 | 0.085452625 |
| donor_cold | 0.089734998 | 0.094210920 | 0.093752469 |
| full_scaffold_cold | 0.086087363 | 0.091668553 | 0.085754576 |
| iid | 0.086661704 | 0.091245124 | 0.088208628 |
| pair_cold | 0.084917397 | 0.088743113 | 0.084718101 |

Primary acceptor-cold comparisons (10,000 acceptor-identity bootstrap replicates):

- C−A: -0.001274398 eV; 95% CI [-0.005445439957307638, 0.002658489158518558].
- C−B: -0.003033280 eV; 95% CI [-0.005996529975795716, -0.00014749916801554578].
- IID C−A: +0.001546925 eV; 95% CI [8.803258780654579e-05, 0.0029984058395145065].

Arm C beats the equal-budget continuous full-molecule control on acceptor-cold validation, but it neither reaches the frozen 0.003 eV improvement over C0 nor establishes IID non-inferiority. The evidence therefore does not admit this representation.
