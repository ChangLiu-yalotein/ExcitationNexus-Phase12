# Gate 2-D2 v2 embedding integrity

The immutable safetensors SHA-256 matched the preregistered value. The true frozen-tokenizer maxima are 417/208/378 for full/donor/acceptor inputs; this target-free correction was locked before any molecular forward. The long-sequence gate passed without truncation.

| Input | Identities | Max tokens | >202 | Raw exact aliases | Cosine-distance Spearman | Median relative distance error | Empirical rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| acceptor | 352 | 378 | 45 | 6 | 0.989965 | 0.032240 | 256 |
| donor | 154 | 208 | 5 | 1 | 0.992972 | 0.030611 | 152 |
| full | 14639 | 417 | 6532 | 358 | 0.988994 | 0.019418 | 512 |

- Repeated forward maximum absolute difference: 0.0.
- Single-versus-padded-batch maximum absolute difference: 2.3543834686279297e-06.
- Full extraction batch-size sensitivity: 1.7881393432617188e-06.
- Trainable encoder parameters: 0; optimizer parameters: 0.

Every exact raw-embedding collision was traced to an identical frozen-tokenizer ID sequence. No distinct token sequence produced an identical embedding. These aliases are a representation limitation, not an extraction failure. Inputs above 202 tokens remain `OUTSIDE_PRETRAINING_LENGTH_SUPPORT` despite successful forward execution.
