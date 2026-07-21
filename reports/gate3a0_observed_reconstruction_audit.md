# Gate 3-A0 observed reconstruction audit

The production assembler uses one record-independent rule: remove the historical `[A]` serialization marker, attach through its first/preceding neighbor, remove one explicit anchor hydrogen when present, and create one single D–A bond. It never uses target values, predictions, molecule-ID product lookup, or record-specific connection mappings.

- Coverage: 100.000000% (15,016/15,016)
- Sanitized: 99.920085%
- Exact canonical matches: 5,512
- Graph-isomorphic matches: 9,492
- Canonical or graph-isomorphic: 99.920085%
- Invalid valence/kekulization: 12
- Attachment ambiguity: 0.000000%
- Pure explicit-role origin checks: 14,263; mismatches: 0
- Unknown/empty-donor original-role rows preserved as unverifiable: 753
- Reversed-input deterministic hash match: True

Decision: **ASSEMBLY_ENGINE_ADMITTED**. Graph-isomorphic matches are permitted by the frozen threshold; the assembler does not invent stereochemistry. All failures remain in a local Git-ignored audit.
