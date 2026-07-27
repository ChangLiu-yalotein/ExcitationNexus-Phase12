# Gate 2-G1C Chemprop formal recovery preregistration

Status: **GATE2G1C_CHEMPROP_FORMAL_INPUTS_PREPARED**.

Chemprop v2 dependency recovery is complete, but the formal D-MPNN benchmark is not yet complete. This amendment prepares frozen Chemprop inputs for the unified grouped OOF protocol and launches only after graph compatibility and actual-data smoke pass.

Eligible records: 25008; eligible structures: 24543.

The initial formal launch is `unified_grouped_5fold_oof / blind / seed42 / fold0`. OOD Chemprop protocols are pending an explicit inner-validation policy because Gate 2-G1B only materialized `inner_fold` for the unified grouped OOF manifest.

Observable provenance uses one-hot `method`, `basis`, `program`, `geometry_fidelity`, and `target_semantics_version` descriptors. Raw `source_cohort`, molecule IDs, file paths, final673 membership, and historical split labels are forbidden model inputs.
