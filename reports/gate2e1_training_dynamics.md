# Gate 2-E1 training dynamics

- iid/S0: best epochs [9, 4, 7]; inner metrics [0.085603, 0.086046, 0.086313]; inner/refit wall 25.7/9.0 s; peak 45.8 MiB.
- iid/M11: best epochs [6, 3, 12]; inner metrics [0.085212, 0.086229, 0.085981]; inner/refit wall 87.1/27.3 s; peak 52.8 MiB.
- iid/M15: best epochs [14, 8, 9]; inner metrics [0.085079, 0.085752, 0.084906]; inner/refit wall 121.4/42.3 s; peak 55.4 MiB.
- acceptor_cold/S0: best epochs [11, 6, 9]; inner metrics [0.092614, 0.093151, 0.092415]; inner/refit wall 26.7/10.4 s; peak 45.8 MiB.
- acceptor_cold/M11: best epochs [8, 6, 9]; inner metrics [0.091746, 0.091908, 0.092238]; inner/refit wall 94.2/31.0 s; peak 52.8 MiB.
- acceptor_cold/M15: best epochs [15, 6, 11]; inner metrics [0.093728, 0.093181, 0.09251]; inner/refit wall 126.3/54.6 s; peak 55.4 MiB.

CUDA emitted a CuBLAS determinism warning because `CUBLAS_WORKSPACE_CONFIG` was not set. The same runtime was retained for every arm and seed; no run was repeated or selected post hoc.
