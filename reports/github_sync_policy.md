# GitHub synchronization policy

GitHub is the versioned home for Phase 12 code, tests, configs, manifests, small audit evidence, hashes, and reports. Raw quantum-chemistry data, primary tables, sealed assets, checkpoints, and large runtime outputs remain on the server and Google Drive.

Publication occurs only at a formal Gate DONE boundary. The checkpoint workflow uses a repository lock, explicit staging whitelist, frozen-hash validation, secret and 20 MiB checks, sealed/raw filename checks, pytest, and remote divergence detection. It never force-pushes, merges, rebases, rewrites a published tag, or deletes local work after failure.

Command:

```bash
scripts/git_checkpoint.sh <gate-id> <short-outcome>
```

Each successful checkpoint creates `gate(<gate-id>): <short-outcome>` and an annotated `<gate-id>-done-YYYYMMDD` tag, then pushes the current branch and that explicit tag.
