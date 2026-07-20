# Gate 2-D1 role-aware 2D representation preregistration

Status: **LOCKED BEFORE FEATURE CONSTRUCTION AND MODEL FITTING**.

The experiment has exactly three arms: frozen C0-512 reference (532 columns),
full-molecule C0-Wide-1536 capacity control (1,556 columns), and RA2D-1536
(20 full descriptors plus 512 full, 512 donor, and 512 acceptor Morgan bits; 1,556
columns). All Morgan fingerprints use radius 2 and `useChirality=False`. No additional
arm, scalar, interaction, learned embedding, memory, or 3D input may be added.

Models use the frozen Gate 1-B1 XGBoost contract and are trained independently for each
protocol using only that protocol's train partition. Evaluation is validation-only.
Arm A predictions are reused; only Arms B and C are fitted, for at most 12 new models.

The primary endpoint is acceptor-cold acceptor-identity-macro MAE and the primary
contrast is C minus B. Admission requires a point difference no greater than -0.0020
eV and an acceptor-cluster bootstrap CI wholly below zero, while the IID
structure-group bootstrap upper bound must be no greater than +0.0020 eV. Bootstrap
uses 10,000 replicates and seed 20260720.

Protocol-local train-label extraction was separately authorized and completed once.
No test artifact, source Parquet, or final673 access is allowed after that extraction.
