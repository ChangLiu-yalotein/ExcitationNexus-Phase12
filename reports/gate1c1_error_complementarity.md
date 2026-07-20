# Gate 1-C1 Error Complementarity

|model|M3-only correct|M3 unique advantage|oracle MAE|oracle gain|
|---|---|---|---|---|
|m3_merged_ensemble|0.109|0.363|0.068234|0.015947|
|m3_dau_shared_ensemble|0.114|0.346|0.068440|0.015742|

All values use the existing 2,319-record frozen test artifact. Oracle-min is non-deployable and no fusion weight was fitted. Adequately powered preregistered winning subgroups: `1`.

Absolute-error Spearman correlations are `0.6990` for XGBoost versus M3-Merged, `0.6968` for XGBoost versus M3-DAU, and `0.8522` between the two 3D ensembles. Thus the 3D errors are not independent enough to turn the oracle diagnostic into a deployable fusion claim.

## Morgan similarity

The preregistered low-similarity bins were deterministically merged for power. All 2,319 records remain covered.

|Nearest-train Morgan layer|Records / groups|XGBoost|Merged|DAU|Interpretation|
|---|---:|---:|---:|---:|---|
|below 0.8|155 / 147|0.082155|0.081219|0.085875|Merged-minus-XGBoost CI `[-0.00998,+0.00815]`; no stable advantage|
|0.8 to 1.0|2164 / 2048|0.084327|0.088127|0.088738|both 3D models significantly worse|

Only target Q4 is an adequately powered winning subgroup: XGBoost/Merged/DAU MAEs are `0.096351/0.085331/0.086099 eV`; both 3D-minus-XGBoost CIs are below zero. These are two model wins in one subgroup, not two independent winning subgroups. All other preregistered chemical-space results are preserved in `logs/gate1c1_evidence.json`.
