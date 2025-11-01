#!/usr/bin/env bash
# organize_by_categories.sh
# Create the same category structure in TARGET (_0003) as in SOURCE (_0002),
# using only the true category folders to avoid false duplicates.

set -euo pipefail
IFS=$'\n\t'

SOURCE=${1:?Usage: $0 <SOURCE_DIR:_0002> <TARGET_DIR:_0003> [--apply]}
TARGET=${2:?Usage: $0 <SOURCE_DIR:_0002> <TARGET_DIR:_0003> [--apply]}
shift 2 || true

APPLY=0
EXTS=("mp4" "webm" "mkv" "mov" "avi")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1 ;;
    --extensions) shift; IFS=' ' read -r -a EXTS <<< "${1:-}" ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
  shift || true
done

# Only these top-level category dirs from SOURCE will be indexed:
CATEGORIES=(
  "action_binding"
  "consistent_attribute_binding"
  "dynamic_attribute_binding"
  "generative_numeracy"
  "motion_binding"
  "object_interactions"
  "spatial_relationships"
)

for d in "$SOURCE" "$TARGET"; do
  [[ -d "$d" ]] || { echo "Directory not found: $d" >&2; exit 1; }
done

# Build find predicate for extensions
ext_pred=()
for e in "${EXTS[@]}"; do ext_pred+=(-iname "*.${e}" -o); done
unset 'ext_pred[${#ext_pred[@]}-1]'

declare -A MAP   # basename -> category (relative path under SOURCE)
declare -A DUP   # basename -> 1 if duplicate across categories

echo "Indexing SOURCE categories only..."
for cat in "${CATEGORIES[@]}"; do
  src_cat="$SOURCE/$cat"
  [[ -d "$src_cat" ]] || { echo "WARN: missing category in SOURCE: $cat (skipping)"; continue; }
  while IFS= read -r -d '' f; do
    base="$(basename "$f")"
    if [[ -n "${MAP[$base]:-}" && "${MAP[$base]}" != "$cat" ]]; then
      DUP["$base"]=1
    else
      MAP["$base"]="$cat"
    fi
  done < <(find "$src_cat" -type f \( "${ext_pred[@]}" \) -print0)
done

echo "Indexed ${#MAP[@]} basenames from categories."
echo "Found ${#DUP[@]} duplicates across categories (these will be skipped)."

moved=0
skipped_nomatch=0
skipped_dup=0
already_ok=0

echo "Planning moves into TARGET by category..."
# Only consider files currently at TARGET top-level (the flat/broken layout)
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  if [[ -n "${DUP[$base]:-}" ]]; then
    ((skipped_dup++)) || true
    echo "[SKIP-AMBIG] $base  (appears in multiple categories in SOURCE)"
    continue
  fi
  cat="${MAP[$base]:-}"
  if [[ -z "$cat" ]]; then
    ((skipped_nomatch++)) || true
    echo "[NO-MATCH ] $base"
    continue
  fi
  dest_dir="$TARGET/$cat"
  dest_path="$dest_dir/$base"
  if [[ "$f" == "$dest_path" ]]; then
    ((already_ok++)) || true
    continue
  fi
  echo "[MOVE     ] $base -> $cat/"
  if [[ $APPLY -eq 1 ]]; then
    mkdir -p "$dest_dir"
    mv -n -- "$f" "$dest_path"
  fi
  ((moved++)) || true
done < <(find "$TARGET" -maxdepth 1 -type f \( "${ext_pred[@]}" \) -print0)

echo
echo "Summary:"
echo "  Planned/Performed moves : $moved"
echo "  Already in place        : $already_ok"
echo "  No match in categories  : $skipped_nomatch"
echo "  Skipped ambiguous names : $skipped_dup"
if [[ $APPLY -eq 0 ]]; then
  echo
  echo "DRY-RUN: re-run with --apply to perform the moves."
fi