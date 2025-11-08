#!/bin/bash

OUTPUT_FILE="categories/typst.toml"
WORD_DIR="words"

mkdir -p "$(dirname "$OUTPUT_FILE")"

(
  echo "[wordlist]"
  echo "list = ["

  for file in "$WORD_DIR"/*.toml; do
    if [ -e "$file" ]; then
      filename=$(basename "$file")
      echo "  \"$filename\","
    fi
  done | sort | sed '$s/,$//'

  echo "]"
) > "$OUTPUT_FILE"

echo "Updated $OUTPUT_FILE"
