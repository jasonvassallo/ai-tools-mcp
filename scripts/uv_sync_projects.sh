#!/bin/bash
set -u

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed or not in PATH." >&2
  exit 1
fi

EXCLUDE_REGEX="${EXCLUDE_REGEX:-wss|taco}"
MAXDEPTH="${MAXDEPTH:-5}"
SYNC_ALL_EXTRAS="${SYNC_ALL_EXTRAS:-1}"

ROOTS=("$@")
if [ "${#ROOTS[@]}" -eq 0 ]; then
  ROOTS=(
    "$HOME/Documents/Code"
    "$HOME/Library/Mobile Documents/com~apple~CloudDocs/Documents/Code"
    "$HOME/Downloads"
  )
fi

projects=()
for root in "${ROOTS[@]}"; do
  [ -d "$root" ] || continue
  while IFS= read -r dir; do
    projects+=("$dir")
  done < <(
    find "$root" -maxdepth "$MAXDEPTH" -type f \
      \( -name pyproject.toml -o -name requirements.txt \) \
      -not -path '*/.venv/*' \
      -not -path '*/node_modules/*' \
      -not -path '*/.git/*' \
      -not -path '*/dist/*' \
      -not -path '*/build/*' \
      2>/dev/null | sed 's#/[^/]*$##'
  )
done

if [ "${#projects[@]}" -eq 0 ]; then
  echo "No projects found."
  exit 0
fi

uniq_projects=()
while IFS= read -r line; do
  uniq_projects+=("$line")
done < <(printf '%s\n' "${projects[@]}" | awk 'NF' | sort -u)

synced=0
installed=0
skipped=0
failed=0

echo "Discovered ${#uniq_projects[@]} candidate projects."
echo "Exclude regex (case-insensitive): $EXCLUDE_REGEX"
echo

for project in "${uniq_projects[@]}"; do
  project_lc="$(printf '%s' "$project" | tr '[:upper:]' '[:lower:]')"
  if [[ "$project_lc" =~ $EXCLUDE_REGEX ]]; then
    echo "[skip] $project"
    skipped=$((skipped + 1))
    continue
  fi

  if [ -f "$project/pyproject.toml" ]; then
    echo "[uv sync] $project"
    sync_args=(--project "$project")
    if [ "$SYNC_ALL_EXTRAS" = "1" ]; then
      sync_args+=(--all-extras)
    fi

    if uv sync "${sync_args[@]}"; then
      synced=$((synced + 1))
    else
      echo "  -> uv sync failed, retrying dependencies-only: $project" >&2
      if uv sync "${sync_args[@]}" --no-install-project; then
        synced=$((synced + 1))
      else
        echo "  -> FAILED uv sync: $project" >&2
        failed=$((failed + 1))
      fi
    fi
    continue
  fi

  if [ -f "$project/requirements.txt" ]; then
    echo "[req install] $project"
    if [ ! -x "$project/.venv/bin/python" ]; then
      uv venv "$project/.venv" || true
    fi
    if uv pip install --python "$project/.venv/bin/python" -r "$project/requirements.txt"; then
      installed=$((installed + 1))
    else
      echo "  -> FAILED requirements install: $project" >&2
      failed=$((failed + 1))
    fi
    continue
  fi
done

echo
echo "Summary:"
echo "  uv synced: $synced"
echo "  requirements installed: $installed"
echo "  skipped (excluded): $skipped"
echo "  failed: $failed"

if [ "$failed" -gt 0 ]; then
  exit 2
fi
