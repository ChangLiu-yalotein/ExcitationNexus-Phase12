# Gate 3-A1 preregistration

Status: **FROZEN BEFORE FEATURE BUILD, MODEL FIT, AND CANDIDATE SCORING**.

The sole scorer is the Gate 1-B1 XGBoost-C0 contract: 20 RDKit descriptors plus a 512-bit radius-2 non-chiral full-molecule Morgan fingerprint. One deterministic deployment model and 20 fixed structure-group bootstrap models are frozen before a single candidate-scoring call. Bootstrap spread is ranking stability only, not calibrated uncertainty. The 4+4+4+4 selection, 0.80 extreme inclusion threshold, identity caps, diversity rule, and hash tie-breaks are locked.

The 36,523 objects are seen-component/unseen-pair computations. They are not donor-OOD, acceptor-OOD, new-scaffold discovery, catalytic activity, or experimental evidence. `PROSPECTIVE_COMPUTATIONAL_SCREENING_ONLY` and `BLOCKED_NO_EXPERIMENTAL_VALIDATION_PATH` remain mandatory.
