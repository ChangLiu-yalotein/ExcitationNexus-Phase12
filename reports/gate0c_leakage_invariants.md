# Gate 0-C leakage invariants

All six available split manifests passed independent verification.

- 15,016 rows covered; `molecule_id` is unique in every manifest.
- Every `structure_group_id_v1` is confined to one partition.
- Retained structure-group weights sum to one.
- The one `HISTORICAL_MODEL_SELECTION_QUARANTINE` row never enters train/val/test/buffer.
- All 17 `HISTORICAL_TRAIN_OVERLAP` rows are train-only.
- Donor-cold donor identities, acceptor-cold acceptor identities, pair-cold pairs, and scaffold-cold full scaffolds have zero pairwise train/val/test overlap.
- Pair-cold val/test donors and acceptors all have at least five train calculation records.
- Both-cold test donor and acceptor sets have zero overlap with train+val; its buffer is explicit.
- No manifest contains Tier 3 targets, final673 membership, labels, or target-derived statistics.
- Gate 0-B sealed aggregate new15016/final673 overlap remains ID=0 and structure=0; no final673 per-sample artifact was created.
- A second generation matched all six assignment hashes; generation from shuffled input rows also matched all six.

The pair-cold split is scientifically an unseen-pair/seen-components protocol, but it is not strong component OOD: 14,706 of 14,861 structure pairs are singletons. Its full-molecule nearest-train Morgan median is 0.8818, close to grouped IID test at 0.8830.
