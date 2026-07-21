# Gate 2-E0 multitask feasibility

Decision: `MULTITASK_TARGET_GRAPH_ADMITTED`.

Admitted optimization graph: one primary, 11 non-redundant secondary targets, and 4 masked fragment targets. Of these, 11 secondaries exceed 95% IID-train structure-group completeness and 4 masked targets exceed 45%.

Gate 2-E1 frozen initial weights: primary 1.0; secondary total 0.5 (equal per admitted task); masked total 0.25 (equal per admitted task). All normalization is protocol-train-only, group-weighted, and observed-label-only. Dynamic loss weighting is prohibited in the first admission experiment.

Validation was used only for coverage/evaluability; no validation relationship or performance selected tasks.
