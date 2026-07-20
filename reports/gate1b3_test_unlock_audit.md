# Gate 1-B3 test-unlock audit

Status: **READY_FOR_TEST_UNLOCK_NOT_YET_UNLOCKED**.

All six formal runs completed under the frozen protocol. M3-Merged completed 36 epochs for all seeds. M3-DAU-Shared seeds 123 and 456 completed 36 epochs; seed 42 stopped at epoch 33 under the preregistered minimum-epoch/patience rule, ten epochs after its best checkpoint at epoch 23. This is a compliant completion, not a failed or shortened run.

All six best checkpoints load on CPU, match their recorded hashes and formal-config hash, and have a corresponding 2,309-record validation prediction file with identical molecule-ID order. No NaN, OOM, Xid, traceback, or temperature stop was observed. The model and environment registries are frozen.

No Gate 1-B3 test prediction artifact exists. Test target and final673 access remain false. The next operation requires a separate explicit unlock record, after which each frozen checkpoint may be evaluated on the 2,319 IID test records exactly once.
