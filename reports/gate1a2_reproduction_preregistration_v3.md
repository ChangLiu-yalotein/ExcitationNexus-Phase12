# Gate 1-A2 B2-1 seed42 reproduction preregistration v3

Status: **FROZEN BEFORE FORMAL CHECKPOINT INFERENCE OR TRAINING**.

V3 supersedes v2 only to correct the parameter-count definition. A constructed historical model has **1,065,570 trainable/model parameters**. The checkpoint `state_dict` contains **1,075,318 tensor elements** because it also includes non-parameter buffers. V2 incorrectly called the latter number a parameter count. V1 and v2 remain immutable audit records.

The implemented architecture and all experimental choices remain those frozen in v2: shared EquiformerV3 graph-level scalar outputs, separate 1→64 projections for the donor and acceptor calls through the shared projection layer, late concatenation/fusion, original 5,120/1,098/1,098 ASE databases, seed 42, unchanged optimizer and at most one formal training run. Test data do not select checkpoints. Final673, new15016, other seeds and other models remain outside scope.

The historical result retains two provenance-linked prediction artifacts: the original run-level MAE `0.07659060508012772 eV` and the later SID paired-comparison vector MAE `0.07656088892531875 eV`. Neither is discarded or chosen post hoc.
