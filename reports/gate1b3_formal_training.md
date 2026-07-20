# Gate 1-B3 formal training protocol

> Finalized 2026-07-20: all six formal checkpoints, the one-time IID test, and the preregistered 198-record role-candidate sensitivity are complete. Final status is `GATE1B3_DONE`; see `gate1b3_final_summary.md`. No retraining or checkpoint replacement occurred during finalization.


The primary task is **J_eh_screened_eV_eps3p5 proxy**, not experimental binding energy. Primary training uses the original explicit donor/acceptor/unknown roles. The two frozen Gate 1-B2 architectures use identical optimization, batching, input, loss, and validation-selection contracts.

A target-blind, train/validation-only common calibration used 2,048/512 records and seed 20260720. Both architectures reached their best validation structure-group-macro MAE at calibration epoch 12. The preregistered formula therefore fixes a common formal maximum of 36 epochs, minimum 15 epochs, patience 10, and min-delta 0.0001 eV. Test targets remain locked.

Formal training consists of exactly seeds 42, 123, and 456 for each architecture. Wave 1 runs M3-Merged; Wave 2 runs M3-DAU-Shared. Results cannot trigger extra seeds, architecture changes, or silent reruns.
