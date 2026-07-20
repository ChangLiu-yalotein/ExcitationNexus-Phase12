# Gate 2-D2 model security and tokenizer audit

Status: **MODEL_ASSET_AND_TOKENIZER_ADMITTED**.

- Repository: ibm-research/MoLFormer-XL-both-10pct at immutable revision a14249e5ad9e3e7c3b1bb604393e914cfcebd2c8.
- License: Apache-2.0; safetensors is available and is the only admitted weight format.
- Runtime custom code was read before execution. No network, subprocess, shell, dynamic eval/exec, or runtime file-write behavior was found. The conversion utility is excluded because it uses torch.load/torch.save.
- Repository config has deterministic_eval=false; runtime must override it to true.
- Pooling: attention-mask-aware mean of final hidden state.
- Max length=512, truncation disabled. Upstream pretraining dropped >202-token molecules, so those inputs are a documented domain-mismatch limitation.

| input | unique | max tokens | >512 | unknown | reconstruction failures |
|---|---:|---:|---:|---:|---:|
| acceptor | 352 | 372 | 0 | 0 | 0 |
| donor | 154 | 208 | 0 | 0 | 0 |
| full | 14639 | 399 | 0 | 0 | 0 |

Wildcards are preserved; no string is changed. No target, test artifact, main Parquet, or final673 asset was accessed.
