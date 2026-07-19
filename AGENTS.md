# Repository execution rules

- Do not commit or push a Gate before its formal DONE state.
- After DONE, run frozen-hash, secret, large-file, sealed-set, and pytest checks before publication.
- Publish with `scripts/git_checkpoint.sh <gate-id> <short-outcome>`; do not use unbounded `git add .`.
- Never commit raw PM6/DFT/TDDFT data, primary tables, checkpoints, optimizer state, credentials, private keys, tokens, or sealed-set per-sample assets.
- Never force-push, rewrite a published Gate tag, or automatically merge/rebase a remote conflict.
- A failed push must preserve all local research files and be recorded for manual resolution.
- The GitHub repository stores code, configs, manifests, small evidence, hashes, and reports. Large research assets remain on the server and Google Drive.
