# Gate 1-C1 Decision

Final status: `GATE1C1_DONE_STOP_PURE_3D`

## Recommendation: STOP_PURE_3D

The frozen evidence does not justify scaling or continuing a pure-3D architecture path; future modeling should retain the 2D baseline and first improve role semantics or use a separately preregistered fusion test.

Decision inputs: geometry signal `True`, both architectures underfit `False`, ordered noise response `True`, unique powered 3D-winning subgroups `1`, oracle gain `0.020820 eV`. Exactly one preregistered branch was selected.

`STOP_PURE_3D` does not mean that the target lacks geometric dependence. Geometry diagnostics are positive, but only three of six runs satisfy the frozen underfit rule, so `SCALE_3D` fails. The two apparent 3D wins both occur in the same target-Q4 subgroup, so the requirement for two distinct adequately powered subgroups is not met and `FUSE_2D_3D` also fails. Oracle-min uses test truth and is strictly non-deployable.

The role-candidate perturbation has median magnitude `0.07105 eV` for Merged and `0.05932 eV` for DAU—respectively 81.1% and 67.0% of their IID MAEs. Continuing a pure-3D path before improving role semantics would therefore amplify an unstable input definition.

The next experiment, if pursued, must be a separately preregistered validation-selected 2D/3D fusion study with no test-weight fitting; it is not authorized by this gate.
