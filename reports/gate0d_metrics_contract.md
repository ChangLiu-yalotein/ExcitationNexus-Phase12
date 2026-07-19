# Gate 0-D metrics contract

Implemented metrics are record MAE/RMSE/R², structure-group-macro MAE/RMSE/R², Spearman, Kendall, valid count, donor/acceptor group MAE, worst-decile error, and worst-group MAE.

Errors are calculated per record first. Structure-group macro MAE is `mean_g(mean_i(|error_i|))`; macro RMSE is `sqrt(mean_g(mean_i(error_i²)))`. Targets and predictions are never averaged before error calculation, preserving the 219 role-inconsistent duplicate groups.

The hand-check case with errors `[1,1]` in one group and `[4]` in another gives record MAE 2.0 and group-macro MAE 2.5. Constant-target R² returns `NaN` with reason `CONSTANT_TARGET`, never infinity or an exception.
