# Leakage risk audit

Status: **BLOCKED for immutable split generation**.

P0 risks:

1. New15016 contains 377 extra rows across 372 duplicate canonical-structure groups. A row split would leak identical structures.
2. external2698 and final673 share 18 canonical structures under different IDs. Historical identity separation is insufficient at structure level.
3. One new row matches external-dev/legacy by structure and must not enter training.
4. Seventeen new rows match old7316 structures and cannot be claimed as independent new samples.
5. All TDDFT/Multiwfn columns, including the primary and linear transforms, are Tier 3 labels only.
6. `pm6_energy_raw` semantics are unresolved for all rows and the field is disabled.
7. Any memory/mean/statistic must be fit after split using training rows only.
8. Source paths, method/basis, termination, root, and parser version are provenance, not predictive inputs.

Required remediation before split freeze: publish a structure-group identity key; resolve/declare replicate policy; create sealed in-memory exclusion by ID and canonical structure; revise historical benchmark interpretation for the 18 external/final shared structures; and test that no canonical group crosses train/val/test.
