# Gate 0-C split balance (post-freeze)

Diagnostics were computed only after manifest hash freeze; assignments were not modified.

| split_name                 | partition             |   records |   structure_groups |   effective_weight |   donors |   acceptors |   pairs |   scaffolds |
|:---------------------------|:----------------------|----------:|-------------------:|-------------------:|---------:|------------:|--------:|------------:|
| iid_group_seed42_v1        | historical_quarantine |         1 |                  1 |                  1 |        1 |           1 |       1 |           1 |
| iid_group_seed42_v1        | test                  |      2319 |               2195 |               2195 |      151 |         342 |    2270 |        1962 |
| iid_group_seed42_v1        | train                 |     10387 |              10248 |              10248 |      154 |         351 |   10328 |        7517 |
| iid_group_seed42_v1        | val                   |      2309 |               2195 |               2195 |      154 |         343 |    2263 |        1975 |
| donor_cold_v1              | historical_quarantine |         1 |                  1 |                  1 |        1 |           1 |       1 |           1 |
| donor_cold_v1              | test                  |      2251 |               2224 |               2224 |       15 |         350 |    2244 |        1776 |
| donor_cold_v1              | train                 |     10530 |              10218 |              10218 |      124 |         352 |   10396 |        7254 |
| donor_cold_v1              | val                   |      2234 |               2196 |               2196 |       15 |         349 |    2220 |        1676 |
| acceptor_cold_v1           | historical_quarantine |         1 |                  1 |                  1 |        1 |           1 |       1 |           1 |
| acceptor_cold_v1           | test                  |      2237 |               2208 |               2208 |      154 |          36 |    2222 |        1716 |
| acceptor_cold_v1           | train                 |     10543 |              10225 |              10225 |      154 |         284 |   10416 |        7512 |
| acceptor_cold_v1           | val                   |      2235 |               2205 |               2205 |      151 |          32 |    2222 |        1369 |
| pair_cold_v1               | historical_quarantine |         1 |                  1 |                  1 |        1 |           1 |       1 |           1 |
| pair_cold_v1               | test                  |      2309 |               2191 |               2191 |      151 |         342 |    2260 |        1957 |
| pair_cold_v1               | train                 |     10387 |              10257 |              10257 |      154 |         352 |   10325 |        7547 |
| pair_cold_v1               | val                   |      2319 |               2190 |               2190 |      151 |         342 |    2275 |        1961 |
| both_cold_external_test_v1 | buffer                |      3291 |               3234 |               3234 |       82 |         262 |    3268 |        2506 |
| both_cold_external_test_v1 | historical_quarantine |         1 |                  1 |                  1 |        1 |           1 |       1 |           1 |
| both_cold_external_test_v1 | test                  |       587 |                587 |                587 |       15 |          40 |     587 |         421 |
| both_cold_external_test_v1 | train                 |      9345 |               9195 |               9195 |      139 |         312 |    9288 |        6645 |
| both_cold_external_test_v1 | val                   |      1792 |               1622 |               1622 |      138 |         288 |    1717 |        1470 |
| full_scaffold_cold_v1      | historical_quarantine |         1 |                  1 |                  1 |        1 |           1 |       1 |           1 |
| full_scaffold_cold_v1      | test                  |      2250 |               2197 |               2197 |      138 |         267 |    2239 |         672 |
| full_scaffold_cold_v1      | train                 |     10511 |              10246 |              10246 |      154 |         352 |   10383 |        8601 |
| full_scaffold_cold_v1      | val                   |      2254 |               2195 |               2195 |      143 |         292 |    2238 |         731 |

Target summaries are retained in `logs/gate0c_postfreeze_diagnostics.json` and did not influence v1.
