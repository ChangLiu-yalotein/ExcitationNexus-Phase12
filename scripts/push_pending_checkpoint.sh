#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ "$(git rev-parse --show-toplevel)" == "$ROOT" ]] || { echo "not the Phase 12 worktree" >&2; exit 2; }
exec 9>"$ROOT/.git/push-pending.lock"
flock -n 9 || { echo "another push is active" >&2; exit 3; }
[[ "$(git branch --show-current)" == "main" ]] || { echo "only main may be pushed" >&2; exit 4; }
git diff --quiet && git diff --cached --quiet || { echo "worktree must be clean" >&2; exit 5; }
[[ -z "$(git ls-files --others --exclude-standard)" ]] || { echo "untracked files remain" >&2; exit 5; }

git fetch --no-tags origin main
local_head="$(git rev-parse HEAD)"
remote_head="$(git rev-parse origin/main)"
git merge-base --is-ancestor "$remote_head" "$local_head" || { echo "origin/main is not an ancestor of local HEAD" >&2; exit 6; }
read -r local_only remote_only < <(git rev-list --left-right --count "HEAD...origin/main")
[[ "$remote_only" == "0" ]] || { echo "remote is ahead or diverged" >&2; exit 6; }

git push -u origin main

declare -a pushed_tags=()
while read -r tag object_type; do
  [[ -n "$tag" ]] || continue
  [[ "$object_type" == "tag" ]] || { echo "skip lightweight tag: $tag" >&2; continue; }
  if [[ -z "$(git ls-remote --tags origin "refs/tags/$tag")" ]]; then
    git push origin "refs/tags/$tag"
    pushed_tags+=("$tag")
  fi
done < <(git for-each-ref --points-at "$local_head" --format='%(refname:strip=2) %(objecttype)' refs/tags)

verified_main="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
[[ "$verified_main" == "$local_head" ]] || { echo "remote main verification failed" >&2; exit 7; }
for tag in "${pushed_tags[@]}"; do
  local_object="$(git rev-parse "$tag")"
  local_commit="$(git rev-parse "$tag^{}")"
  remote_object="$(git ls-remote --tags origin "refs/tags/$tag" | awk '{print $1}')"
  remote_commit="$(git ls-remote --tags origin "refs/tags/$tag^{}" | awk '{print $1}')"
  [[ "$remote_object" == "$local_object" && "$remote_commit" == "$local_commit" ]] || {
    echo "remote tag verification failed: $tag" >&2; exit 7;
  }
done

echo "push verified: main=$local_head"
if ((${#pushed_tags[@]})); then printf 'tags=%s\n' "${pushed_tags[*]}"; else echo "tags=none-pending"; fi
