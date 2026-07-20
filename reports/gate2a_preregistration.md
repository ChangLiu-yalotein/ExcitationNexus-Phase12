# Gate 2-A preregistration

Status: **LOCKED BEFORE TRAINING**.

Five frozen protocols (`donor_cold`, `acceptor_cold`, `pair_cold`, `both_cold`, and `full_scaffold_cold`) each receive four fixed baselines: train-weighted median, Ridge-C0, XGBoost-C0, and XGBoost-C1.5-safe. This produces 20 frozen baseline assets. XGBoost runs once per feature tier and protocol with seed 42 as an engineering label; the no-subsampling configuration does not support a stochastic multi-seed claim.

C0 uses the Gate 1-B1 frozen 532-column input: 20 RDKit descriptors plus Morgan radius 2, 512 bits, and chirality disabled. C1.5-safe adds only PM6 HOMO, LUMO, and gap. The separate OOD diagnostic uses Morgan radius 2, 2048 bits, and chirality enabled for full/donor/acceptor views; it is target-free and forbidden as model input.

All preprocessing is fitted on train only with structure-group weights. Validation group-macro MAE is primary. Before any OOD test target access, all 20 model assets, ten preprocessors, validation predictions/metrics, environment, manifests, features, and diagnostic hashes must be frozen.

The five OOD test ID sets are deduplicated into one union and read from the Parquet target column in one Arrow operation. Evaluation is one-shot and a second call fails closed. Within-protocol model comparisons use paired structure-group bootstrap. IID-to-OOD degradation uses independent structure-group bootstrap because protocol test sets differ; it is descriptive rather than a paired causal comparison.

C0 remains primary. C1.5-safe may be called beneficial within one protocol only if the preregistered paired group-bootstrap CI for C1.5-safe minus C0 lies entirely below zero; it never replaces the global C0 primary baseline. Buffer and historical quarantine never form Datasets or receive predictions. `final673` remains sealed.

## V2 implementation-only correction

The first V1 execution failed before similarity output, model fitting, or test access because RDKit could not kekulize attachment fragment `*1ccc[se]c1`. V2 reuses the already frozen Gate 0-C diagnostic parser: normal sanitization first, then `sanitize=False` with all sanitization operations except kekulization. The fingerprint definition, split, features, models, bootstrap, and test policy are unchanged. The zero-progress failure remains in `logs/gate2a_training.log`; it is not a scientific rerun.

## V3 protocol-scoped target-join correction

V2 completed and froze the target-free 2048-bit similarity asset, then stopped before the first model fit because a union of all protocols' train/val labels necessarily includes molecules that are test in another protocol. V3 preserves the strict firewall by joining labels only to each protocol's own train/val rows. The later test-once evaluator still performs one deduplicated union Arrow read after all 20 assets are frozen. No scientific setting changes, and the V2 failure is preserved in `logs/gate2a_training_v2.log`.
