# Gate 2-C UQ and applicability-domain preregistration

Status: **LOCKED BEFORE CALIBRATION AND TEST COVERAGE**.

The prior missing-calibration-label blocker is resolved by the explicitly authorized
`AUTHORIZED_MINIMAL_VALIDATION_LABEL_EXTRACTION`. The extraction performed one Arrow
read of only `molecule_id` and the frozen primary target for the union of each
protocol's validation IDs. It did not request test, buffer, quarantine, or final673
rows. The row-level artifact is local and Git-ignored.

The primary point predictor is the already frozen XGBoost-C0 model. No point model is
trained or changed. The nominal levels are 80%, 90%, and 95%. Finite-sample conformal
quantiles use rank `ceil((n+1)*coverage)`; a rank greater than the number of calibration
scores is reported as `UNATTAINABLE_FINITE_SAMPLE`, never replaced with the maximum.

The interval methods are record absolute-residual split conformal, structure-group
maximum-residual conformal, and protocol-specific held-out-identity maximum-residual
conformal. Both-cold remains an empirical donor-by-acceptor crossed-cluster analysis
without an exact conformal guarantee. Primary bootstrap units follow the frozen Gate
2-B identity definitions; 10,000 replicates and seed 20260720 are fixed.

The target-free AD score is full-molecule similarity for IID, pair-cold, and
scaffold-cold; donor similarity for donor-cold; acceptor similarity for
acceptor-cold; and `min(donor, acceptor)` for both-cold. Thresholds for retaining
100%, 90%, 80%, 70%, and 50% are fit on validation only. The high-error threshold is
the validation absolute-residual 90th percentile. Test curves are diagnostic only and
cannot alter these thresholds or methods.

The complete machine-readable contract and fixed decision rules are in
`configs/gate2c_uq_applicability_audit_v1.json`, whose SHA-256 is
`9d7ad57d63b0b64e8817c2c2ac48d9c4c094cf84b2616c1cd3265da0c0e2775b`.
