# Gate 2-D2 final decision

Gate status: **BLOCKED_PREREGISTERED_PCA_INFEASIBLE**.

The frozen Arm C requires donor PCA to 256 dimensions while fitting unique protocol-train donor structures with equal weight. Across protocols there are only 124-154 unique train donors, so centered PCA has at most 123-153 identifiable components.

| protocol | unique donor | max donor PCs | unique acceptor | max acceptor PCs |
|---|---:|---:|---:|---:|
| iid | 154 | 153 | 351 | 350 |
| donor_cold | 124 | 123 | 352 | 351 |
| acceptor_cold | 154 | 153 | 284 | 283 |
| pair_cold | 154 | 153 | 352 | 351 |
| both_cold | 139 | 138 | 312 | 311 |
| full_scaffold_cold | 154 | 153 | 352 | 351 |

No result label such as FROZEN_CONTINUOUS_REPRESENTATION_NOT_ADMITTED is assigned, because the representation experiment was never run. Repetition, zero-padding, transductive PCA, or post-lock dimension changes would violate the preregistration. A v2 requires an explicit new compression contract; the cleanest candidate is a target-free fixed random projection with the original 512/256/256 output budgets, but it is not authorized here.

No test artifact, main Parquet, final673, remote model code, embedding, PCA, or regression model was accessed or executed.
