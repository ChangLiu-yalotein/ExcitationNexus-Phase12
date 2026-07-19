# Gate 1-B2 model fairness

M3-Merged and M3-DAU-Shared use one identical `{'hidden_dim': 48, 'num_rbf': 16, 'layers': 2, 'cutoff_angstrom': 5.0, 'max_neighbors': 32}` backbone family and the same optimizer/loss. Parameter counts are 36,689 and 36,461; relative difference is 0.621%, within 5%. DAU has one shared parameter set but invokes it on donor, acceptor, and unknown subgraphs separately; empty roles use zero vectors and presence=0. Runtime/throughput, rather than parameter count alone, records its compute overhead.
