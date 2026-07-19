# Gate 0-D summary

Final status: **GATE0D_DONE**.

- CPU: 33/33 tests passed; six 15,016-row joins and frozen counts passed.
- Normalization: six train-only, group-weighted registries; non-train target mutation invariance passed.
- Raw graphs: 72 deterministic PM6/DFT samples parsed across all split train/val/test partitions.
- Tiny model: plumbing-only distance/RBF role-aware model; CPU forward/backward and invariance passed.
- GPU: physical GPU 0, two FP32 batches of four, finite loss/gradients, nonzero shared and primary gradients, parameter update passed.
- Peak GPU allocated/reserved: 32.7/46.0 MiB; wall time 1.64 s.
- No epoch, DDP, checkpoint, model comparison, final673 access, or frozen-asset modification.

Gate 1-A may begin only under a new explicit instruction.

Code snapshot: `432cebe0fdd1088ec798fc130968fab0f37e3dd635deaa3833f29b104649b392` across 34 files. Gate 0-D did not run B2-1 or the 0.0702 champion.
