# Gate 2-E1 post-freeze integrity correction

Final status: `BLOCKED_MULTITASK_PIPELINE_INTEGRITY`.

The preregistration required primary-quantile balance based on structure-group weights. The implemented inner-unit target used an unweighted record mean. IID is unaffected because each atomic unit is one structure group. Acceptor-cold is affected: the frozen checkpoint set has 1529 records, the correct weighted reconstruction has 1561, and 146 records change assignment.

This was detected only after official validation had been consumed. Therefore v1 cannot be repaired by retraining: such a rerun would be informed by observed validation results. The 18 models and reported metrics remain preserved as diagnostic artifacts, but `MULTITASK_SIGNAL_INCONCLUSIVE` and the masked decision are invalidated as admission conclusions. No rerun, test access, or tag rewrite occurred. A future experiment requires a new preregistration and an unobserved evaluation boundary.
