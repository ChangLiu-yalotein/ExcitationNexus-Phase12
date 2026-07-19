# ML environment inventory

- Interpreter: `/home/changliu/miniconda3/envs/ML/bin/python`
- Python 3.10.19
- PyTorch 2.8.0+cu128; `torch.version.cuda=12.8`
- NVIDIA driver 580.95.05; driver-reported CUDA 13.0
- `torch.cuda.is_available() = True`; visible devices = 7
- PyTorch Geometric 2.7.0
- e3nn 0.5.9
- RDKit 2025.09.6
- pandas 2.3.3; pyarrow 24.0.0
- scikit-learn 1.7.2; scipy 1.15.3
- xgboost 3.2.0; NumPy 2.2.6

The environment satisfies the dependency-level entry condition for a CPU/single-GPU smoke test. No training import, data-loader, forward/backward, or CUDA kernel smoke was executed in Gate 0-A.

`equiformer_v3_model/` exists but is not a Git worktree, so a commit cannot be reported. Snapshot evidence: 1,032 Python files; aggregate SHA-256 `34962f9a277ff49f3e261a2f2735a74ea6f5b747639d612bd2eddf8c82702e5a`.
