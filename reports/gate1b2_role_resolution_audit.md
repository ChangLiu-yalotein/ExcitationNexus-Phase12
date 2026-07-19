# Gate 1-B2 role resolution audit

All 15,016 records were audited without target access. Original classes reproduce 14,263 pure D/A, 366 D+A+unknown, and 387 empty-donor+unknown records.

Empty-donor resolution status: `{"RESOLVED_UNIQUE_OR_SYMMETRY_EQUIVALENT": 198, "UNRESOLVED_AMBIGUOUS": 189}`. Partition/status counts: `{"historical_quarantine": {"NOT_APPLICABLE_ORIGINAL_EXPLICIT": 1, "RESOLVED_UNIQUE_OR_SYMMETRY_EQUIVALENT": 0, "UNRESOLVED_AMBIGUOUS": 0}, "test": {"NOT_APPLICABLE_ORIGINAL_EXPLICIT": 2253, "RESOLVED_UNIQUE_OR_SYMMETRY_EQUIVALENT": 32, "UNRESOLVED_AMBIGUOUS": 34}, "train": {"NOT_APPLICABLE_ORIGINAL_EXPLICIT": 10121, "RESOLVED_UNIQUE_OR_SYMMETRY_EQUIVALENT": 140, "UNRESOLVED_AMBIGUOUS": 126}, "val": {"NOT_APPLICABLE_ORIGINAL_EXPLICIT": 2254, "RESOLVED_UNIQUE_OR_SYMMETRY_EQUIVALENT": 26, "UNRESOLVED_AMBIGUOUS": 29}}`.

Original explicit roles remain the primary analysis. A resolved role set, when uniquely supported, is sensitivity-only. Ambiguous/inconsistent/insufficient records remain explicit unknown and are never deleted or folded into donor. D81_A28 retains its conflict flag.
