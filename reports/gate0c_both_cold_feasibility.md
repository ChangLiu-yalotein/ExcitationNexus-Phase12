# Gate 0-C both-cold feasibility

Status: **DONE** under the preregistered external-style protocol.

- train: 9,345 records / 9,195 effective structure groups
- seen-component grouped validation: 1,792 / 1,622
- both-cold test: 587 / 587
- explicit buffer: 3,291 / 3,234
- historical quarantine: 1 / 1
- retained train+val+test fraction: 78.0767%
- buffer fraction: 21.9166%
- test donor structures: 15
- test acceptor structures: 40
- test donor overlap with train+val: 0
- test acceptor overlap with train+val: 0

All preregistered minimum powers pass: test records >=500, test effective structures >=450, donors >=15, acceptors >=30, train >=7,000, and validation >=1,000. The test is 3.91% of all 15,016 records, below the preferred 5%-10% band; this preference followed zero leakage, minimum power, retained-record maximization, and buffer minimization in the locked lexicographic objective and is reported as a limitation, not repaired post-freeze.

Nearest-train Morgan medians for the test are full molecule 0.7129, donor 0.3333, and acceptor 0.7805. The donor side is substantially more OOD than the acceptor side. Similarities are post-freeze diagnostics only.
