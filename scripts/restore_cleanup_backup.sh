#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

latest_backup="$(ls -dt _backup/cleanup-* 2>/dev/null | head -n1 || true)"
if [[ -z "$latest_backup" ]]; then
  echo "No cleanup backup found under _backup/cleanup-*"
  exit 1
fi

echo "Restoring from: $latest_backup"

# Copy backed up files back to project root.
# Existing files with same path will be overwritten.
find "$latest_backup" -type f | while read -r src; do
  rel="${src#${latest_backup}/}"
  dst="$ROOT_DIR/$rel"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "Restored: $rel"
done

echo "Restore completed."
