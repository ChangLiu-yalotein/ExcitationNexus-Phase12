# Gate 1-C1 Learning Dynamics

|run|best epoch|epochs|best val MAE|underfit|overfit|
|---|---|---|---|---|---|
|m3_merged_seed42|28|36|0.091612|False|False|
|m3_merged_seed123|36|36|0.092044|True|False|
|m3_merged_seed456|36|36|0.092142|True|False|
|m3_dau_shared_seed42|23|33|0.093424|False|False|
|m3_dau_shared_seed123|32|36|0.092172|False|False|
|m3_dau_shared_seed456|36|36|0.092749|True|False|

The preregistered underfit rule requires a late best epoch and improving final-five validation slope. No additional epoch is authorized by this diagnosis.
