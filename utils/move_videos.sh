#!/usr/bin/env bash
# flatten_to_parent.sh
# Move all video files from subfolders into the given parent folder.
# - Dry-run by default; use --apply to move.
# - Preserves existing parent files; if a name collision occurs,
#   appends _<subdir> or _<subdir>_<n> to the basename.

set -euo pipefail
IFS=$'\n\t'

PARENT="${1:-}"
[[ -z "$PARENT" ]] && { echo "Usage: $0 <PARENT_DIR> [--apply] [--extensions \"mp4 webm mkv mov avi\"]"; exit 1; }
shift || true

APPLY=0
EXTS=("mp4" "webm" "mkv" "mov" "avi")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1 ;;
    --extensions) shift; IFS=' ' read -r -a EXTS <<< "${1:-}" ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
  shift || true
done

[[ -d "$PARENT" ]] || { echo "Not a directory: $PARENT"; exit 1; }

# Build extension predicate for find
ext_pred=()
for e in "${EXTS[@]}"; do ext_pred+=(-iname "*.${e}" -o); done
unset 'ext_pred[${#ext_pred[@]}-1]'

echo "Parent: $PARENT"
echo "Extensions: ${EXTS[*]}"
echo "Mode: $([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)"

moves=0
skips=0

safe_dest_path() {
  local src="$1"
  local subdir_rel="$2"   # relative subdir from parent (no leading ./)
  local base ext name candidate
  base="$(basename "$src")"
  ext="${base##*.}"
  name="${base%.*}"
  # sanitize subdir part for suffix
  local suffix="$(echo "$subdir_rel" | tr '/ ' '__' )"
  candidate="$PARENT/${name}_${suffix}.${ext}"
  if [[ ! -e "$candidate" ]]; then
    echo "$candidate"; return
  fi
  # add numeric suffix if still colliding
  local n=1
  while : ; do
    candidate="$PARENT/${name}_${suffix}_${n}.${ext}"
    [[ ! -e "$candidate" ]] && { echo "$candidate"; return; }
    n=$((n+1))
  done
}

# Find files deeper than the parent (i.e., in subfolders)
while IFS= read -r -d '' f; do
  # If already in parent root, skip
  [[ "$(dirname "$f")" == "$PARENT" ]] && { ((skips++)) || true; continue; }

  rel="${f#"$PARENT/"}"
  subdir_rel="$(dirname "$rel")"

  dest="$PARENT/$(basename "$f")"
  if [[ -e "$dest" ]]; then
    dest="$(safe_dest_path "$f" "$subdir_rel")"
  fi

  echo "[MOVE] $rel -> ${dest#"$PARENT/"}"
  if [[ $APPLY -eq 1 ]]; then
    mv -n -- "$f" "$dest"
  fi
  ((moves++)) || true
done < <(find "$PARENT" -mindepth 2 -type f \( "${ext_pred[@]}" \) -print0)

echo
echo "Summary:"
echo "  Planned/Performed moves : $moves"
echo "  Skipped (already at root): $skips"
if [[ $APPLY -eq 0 ]]; then
  echo
  echo "DRY-RUN. Re-run with --apply to perform the moves."
fi