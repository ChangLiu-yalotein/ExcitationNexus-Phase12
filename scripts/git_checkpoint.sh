#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <gate-id> <short-outcome>" >&2
  exit 2
fi
GATE_ID="$1"
OUTCOME="$2"
[[ "$GATE_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid gate id" >&2; exit 2; }
[[ "$OUTCOME" =~ ^[A-Za-z0-9._[:space:]-]+$ ]] || { echo "invalid outcome" >&2; exit 2; }

exec 9>"$ROOT/.git/git-checkpoint.lock"
flock -n 9 || { echo "another checkpoint is active" >&2; exit 3; }

[[ -d .git ]] || { echo "not a Git repository" >&2; exit 4; }
branch="$(git branch --show-current)"
[[ -n "$branch" ]] || { echo "detached HEAD is forbidden" >&2; exit 4; }

expected_table="e7587b1546039f099a4dbd0d352e98885bb2ebdbdcfa18884dd4355eed815a83"
actual_table="$(sha256sum /home/changliu/ExcitationNexus_Data_v2/tables/molecule_values_v3.parquet | cut -d' ' -f1)"
[[ "$actual_table" == "$expected_table" ]] || { echo "frozen table hash mismatch" >&2; exit 5; }
sha256sum -c data_registry/gate0c_split_sha256.txt >/dev/null

mapfile -t publish_candidates < <(git ls-files --cached --others --exclude-standard)
for candidate in "${publish_candidates[@]}"; do
  if [[ -f "$candidate" ]] && (( $(stat -c %s "$candidate") > 20 * 1024 * 1024 )); then
    echo "unapproved file over 20 MiB: $candidate" >&2
    exit 6
  fi
done
if printf '%s\n' "${publish_candidates[@]}" | grep -Eqi '(^|/)(\.env($|\.)|.*(access[_-]?token|auth[_-]?token|api[_-]?token|credential|private_key).*|id_(rsa|ed25519).*|.*\.(pem|key)$)'; then
  echo "sensitive filename detected" >&2; exit 6
fi
if printf '%s\n' "${publish_candidates[@]}" | grep -Eqi '(final673|final[_-]?blind|sealed[_-]?set|/(raw_compact|tables|checkpoints?)/|\.(pt|pth|ckpt|onnx|safetensors)$)'; then
  echo "raw/sealed/checkpoint filename detected" >&2; exit 6
fi
if rg -l --hidden -g '!.git/**' -g '!__pycache__/**' -g '!.pytest_cache/**' \
  -e 'ghp_[A-Za-z0-9]{20,}' -e 'github_pat_[A-Za-z0-9_]+' \
  -e '-----BEGIN (OPENSSH |RSA |EC )?PRIVATE KEY-----' -e 'AKIA[0-9A-Z]{16}' . | grep -q .; then
  echo "secret content pattern detected" >&2; exit 6
fi

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
/home/changliu/miniconda3/envs/ML/bin/python -m pytest -q tests

git add -- .gitignore AGENTS.md README.md pyproject.toml \
  src tests scripts configs data_registry manifests reports logs \
  PROJECT_STATE.md TODO.md DECISIONS.md RUN_REGISTRY.csv \
  runs/gate1a1_cheap_reproduction/*.json \
  runs/gate1a1_cheap_reproduction/*.csv \
  runs/gate1a2_b21_seed42/published/*.json \
  runs/gate1a2_b21_seed42/published/*.csv \
  runs/gate1a3_b21_multiseed/published/*.json \
  runs/gate1a3_b21_multiseed/published/*.csv \
  runs/gate1b1_new_iid_cheap_baselines/test_unlock_v1.json \
  runs/gate1b1_new_iid_cheap_baselines/published/*.json \
  runs/gate1b1_new_iid_cheap_baselines/published/*.csv \
  runs/gate1b2_3d_admission/published/*.json \
  runs/gate2a_ood_baselines/published/*.json

if git diff --cached --quiet; then
  echo "no changes; checkpoint skipped"
  exit 0
fi

if git remote get-url origin >/dev/null 2>&1; then
  git fetch --no-tags origin "$branch"
  if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    read -r local_only remote_only < <(git rev-list --left-right --count "HEAD...origin/$branch")
    [[ "$remote_only" == "0" ]] || { echo "remote is ahead or diverged; stop" >&2; exit 7; }
  fi
fi

git commit -m "gate(${GATE_ID}): ${OUTCOME}"
if [[ "${GATE_ID,,}" == "gate1a1" ]]; then
  tag="gate1a1-cheap-reproduction-$(date -u +%Y%m%d)"
elif [[ "${GATE_ID,,}" == "gate1a2" ]]; then
  tag="gate1a2-b21-seed42-reproduction-$(date -u +%Y%m%d)"
elif [[ "${GATE_ID,,}" == "gate1a3" ]]; then
  tag="gate1a3-b21-multiseed-reproduction-$(date -u +%Y%m%d)"
elif [[ "${GATE_ID,,}" == "gate1b1" ]]; then
  tag="gate1b1-new-iid-cheap-baselines-$(date -u +%Y%m%d)"
elif [[ "${GATE_ID,,}" == "gate1b2" ]]; then
  tag="gate1b2-3d-baseline-admission-$(date -u +%Y%m%d)"
elif [[ "${GATE_ID,,}" == "gate1b3" ]]; then
  tag="gate1b3-iid-3d-baselines-$(date -u +%Y%m%d)"
elif [[ "${GATE_ID,,}" == "gate1c1" ]]; then
  tag="gate1c1-error-mechanism-diagnosis-$(date -u +%Y%m%d)"
elif [[ "${GATE_ID,,}" == "gate2a" ]]; then
  tag="gate2a-ood-cheap-baselines-$(date -u +%Y%m%d)"
elif [[ "${GATE_ID,,}" == "gate2b" ]]; then
  tag="gate2b-hierarchical-ood-audit-$(date -u +%Y%m%d)"
elif [[ "${GATE_ID,,}" == "gate2c" ]]; then
  tag="gate2c-ood-uq-applicability-$(date -u +%Y%m%d)"
elif [[ "${GATE_ID,,}" == "gate2d1" ]]; then
  tag="gate2d1-role-aware-2d-admission-$(date -u +%Y%m%d)"
elif [[ "${GATE_ID,,}" == "gate2e0" ]]; then
  tag="gate2e0-multitask-target-audit-$(date -u +%Y%m%d)"
elif [[ "${GATE_ID,,}" == "gate2e1" ]]; then
  tag="gate2e1-physics-multitask-admission-$(date -u +%Y%m%d)"
else
  tag="$(printf '%s' "$GATE_ID" | tr '[:upper:]' '[:lower:]')-done-$(date -u +%Y%m%d)"
fi
if git rev-parse "$tag" >/dev/null 2>&1; then
  echo "published tag already exists: $tag" >&2; exit 8
fi
git tag -a "$tag" -m "${GATE_ID} DONE: ${OUTCOME}"
git push -u origin "$branch"
git push origin "refs/tags/$tag"
echo "commit=$(git rev-parse HEAD)"
echo "tag=$tag"
echo "tests=passed"
git diff-tree --no-commit-id --name-only -r HEAD
