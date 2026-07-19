# Gate 1-A2 B2-1 seed42 reproduction preregistration v2

Status: **FROZEN BEFORE FORMAL CHECKPOINT INFERENCE OR TRAINING**.

V2 supersedes v1 for the formal run because post-lock source inspection established that the implemented tower output is not an intermediate atom embedding. The frozen model runs the same shared EquiformerV3 tower on donor and acceptor graphs, extracts each tower's graph-level scalar `energy`, projects that scalar from 1 to 64 dimensions, concatenates the two projected vectors, and applies the fusion MLP and energy head. This is the code actually represented by the historical checkpoint. V1 remains immutable as an audit trail and is not overwritten.

All other frozen facts remain unchanged:

- original B2-1 databases: 5,120 train / 1,098 validation / 1,098 test;
- Layer G PM6-complete manifest: 5,118 / 1,098 / 1,097;
- strict reproduction uses the original 7,316 ASE DB records;
- cheap pairing uses only the common 1,097 test SIDs;
- seed 42, one formal training run, unchanged historical optimizer/configuration;
- original run-level test MAE 0.07659060508012772 eV;
- later SID paired-comparison vector MAE 0.07656088892531875 eV;
- the two historical vectors remain distinct frozen artifacts;
- no final673, new15016, other seed, hyperparameter search, B2-0, or B2-2a access.

The historical validation implementation averages batch means equally and therefore overweights the last two-record batch. The historical checkpoint directory also collided across concurrently launched seeds. These are provenance limitations, not reasons to silently repair the strict reproduction protocol.
