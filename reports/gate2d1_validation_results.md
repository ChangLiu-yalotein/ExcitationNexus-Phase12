# Gate 2-D1 validation-only results

No test prediction, test target, main source Parquet, or final673 asset was read.

| Protocol | A C0-512 | B Wide-1536 | C RA2D-1536 | C−B (eV) | 95% cluster CI |
|---|---:|---:|---:|---:|---|
| iid | 0.086662 | 0.087319 | 0.086413 | -0.000905 | [-0.002070, +0.000241] |
| donor_cold | 0.089735 | 0.091033 | 0.088447 | -0.002587 | [-0.006973, +0.000940] |
| acceptor_cold | 0.096529 | 0.097919 | 0.100244 | +0.002325 | [-0.003722, +0.010601] |
| pair_cold | 0.084917 | 0.083896 | 0.084661 | +0.000765 | [-0.000349, +0.001906] |
| both_cold | 0.089335 | 0.088712 | 0.088514 | -0.000347 | [-0.002611, +0.001954] |
| full_scaffold_cold | 0.086087 | 0.086426 | 0.084263 | -0.002163 | [-0.003546, -0.000713] |

Metrics are protocol-specific identity-macro MAE; both-cold displays the mean of donor- and acceptor-identity macro MAE and uses the preregistered two-way bootstrap. These validation results cannot be presented as frozen test performance.
