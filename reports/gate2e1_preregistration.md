# Gate 2-E1 preregistration

Status: `LOCKED_BEFORE_INNER_SPLIT_MODEL_INITIALIZATION_OR_TRAINING`.

This validation-only admission compares S0, M11, and M15 with identical frozen C0-512 inputs, shared-trunk architecture, primary head, initialization logic, optimizer, and protocol-local data. Only auxiliary heads and their frozen losses differ.

IID guards overall performance; acceptor-cold is the primary endpoint. Official validation cannot select epochs. A single protocol-level inner split is frozen first and reused by all arms and seeds. Each inner best epoch is then refit from the same seed on full official train before any official validation label or prediction is read.

The fixed loss weights are primary 1.0, 11 secondaries totaling 0.5, and four masked targets totaling 0.25. Dynamic weighting, task removal, PM6/DFT/TDDFT inputs, MoLFormer, 3D, memory, retrieval, test artifacts, source Parquet, and final673 are prohibited.

The admission thresholds, 10,000-replicate paired cluster bootstrap, and gradient-conflict diagnostics are frozen in the configuration. Gradient results are explanatory only and cannot change weights or tasks.
