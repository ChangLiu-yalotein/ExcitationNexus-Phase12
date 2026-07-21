# Gate 2-E1 pipeline and fairness

All arms use the same 532 C0 inputs. Shared trunk parameters: 1,457,920; primary head: 33,025. Total parameters are S0 1,490,945, M11 1,854,220, and M15 1,986,320. Primary-path names and shapes are identical. Only auxiliary heads add parameters.

The frozen inner splits have zero unit leakage. Official validation was inaccessible during inner selection and full-train refit. All 18 model hashes and normalizations were frozen before the one-time validation unlock. Test, source Parquet, buffer, quarantine, and final673 were not accessed.
