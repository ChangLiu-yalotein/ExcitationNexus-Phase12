# Gate 2-C applicability-domain audit

All cutoffs were frozen on validation. Test risk–coverage curves are diagnostic and did not select a threshold.

## acceptor_cold

- Validation-locked 80% cutoff retained 0.680 of test records.
- Record MAE: 0.097760 → 0.090314 eV.
- Identity-macro MAE: 0.097320 → 0.089060 eV.
- AD score vs absolute error Spearman: -0.1287366344757182
- Fixed preregistered usefulness rule: True

## both_cold

- Validation-locked 80% cutoff retained 0.000 of test records.
- Record MAE: 0.084399 → nan eV.
- Identity-macro MAE: 0.084378 → nan eV.
- AD score vs absolute error Spearman: -0.006071939960380951
- Fixed preregistered usefulness rule: False

## donor_cold

- Validation-locked 80% cutoff retained 0.797 of test records.
- Record MAE: 0.084649 → 0.085858 eV.
- Identity-macro MAE: 0.087609 → 0.089357 eV.
- AD score vs absolute error Spearman: 0.05862330388457706
- Fixed preregistered usefulness rule: False

## full_scaffold_cold

- Validation-locked 80% cutoff retained 0.822 of test records.
- Record MAE: 0.081984 → 0.077507 eV.
- Identity-macro MAE: 0.082087 → 0.077820 eV.
- AD score vs absolute error Spearman: -0.03231556140250891
- Fixed preregistered usefulness rule: False

## iid

- Validation-locked 80% cutoff retained 0.808 of test records.
- Record MAE: 0.085299 → 0.086190 eV.
- Identity-macro MAE: 0.084181 → 0.085124 eV.
- AD score vs absolute error Spearman: 0.05450985028121425
- Fixed preregistered usefulness rule: False

## pair_cold

- Validation-locked 80% cutoff retained 0.795 of test records.
- Record MAE: 0.085867 → 0.085279 eV.
- Identity-macro MAE: 0.085296 → 0.084520 eV.
- AD score vs absolute error Spearman: 0.01658830980223621
- Fixed preregistered usefulness rule: False

## Decision

`AD_SCORE_NOT_VALIDATED`


The preregistered evaluator retained aggregate AD-score/error correlations, fixed-cutoff risk, high-error detection, AURC, and diagnostic risk–coverage curves. It did not retain the AD-score versus binary interval-miscoverage correlation. Because the once-only test lock is consumed and no per-sample Gate 2-C residual artifact was published, that secondary correlation is `NOT_COMPUTED_FAIL_CLOSED`; test artifacts were not reopened.
