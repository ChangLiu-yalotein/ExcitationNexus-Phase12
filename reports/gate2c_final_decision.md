# Gate 2-C final decision

Final status: **GATE2C_DONE_UQ_APPLICABILITY_AUDIT**.

Decision labels:

- `UQ_EMPIRICALLY_CALIBRATED_OOD`
- `BOTH_COLD_UQ_UNSUPPORTED`
- `AD_SCORE_NOT_VALIDATED`

Interpretation:

- IID and the adequately powered OOD identity protocols achieved the preregistered empirical coverage tolerance. OOD exchangeability remains unverified, so no strict distribution-free claim is made.
- Acceptor-cold point prediction is still the principal OOD failure mode, but its conservative identity-max interval did not under-cover; the cost is a wide 90% interval.
- Both-cold remains unsupported because crossed-cluster exchangeability and statistical power are inadequate, despite acceptable marginal empirical coverage.
- The validation-locked similarity AD rule was not stable across all protocols and is not validated as a deployment filter.
