# Gate 1-A3 preregistration

Status: **FROZEN BEFORE CHECKPOINT INFERENCE OR TRAINING**.

Exactly one fixed 80-epoch B2-1 run will be executed for each of seeds 123 and 456 on separate physical GPUs. The historical shared-directory collision is retained as provenance: seed123 uses its epoch-80 final checkpoint; seed456 uses the surviving best-validation checkpoint. No result-driven restart or tuning is allowed.

Historical seed123 test MAE: `0.081370525062084198` eV.  Historical seed456 test MAE: `0.080265983939170837` eV.  Historical exact three-seed mean/sample std: `0.079409038027127579 ± 0.0025025339119785065` eV.

The 7,316-record B2-1 protocol, batch-macro validation limitation, and 1,097-SID cheap comparison boundary remain frozen. Final673 and new15016 are outside scope.
