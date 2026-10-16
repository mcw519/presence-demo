#!/usr/bin/env bash
# Render docs/img/*.mmd to PNG with mermaid-cli.
#   MMDC=/path/to/mmdc scripts/render_diagrams.sh
# Install once:  npm i @mermaid-js/mermaid-cli   (needs a Chrome/Chromium)
set -euo pipefail
cd "$(dirname "$0")/../docs/img"
MMDC=${MMDC:-mmdc}
for f in *.mmd; do
  "$MMDC" ${MMDC_ARGS:-} -i "$f" -o "${f%.mmd}.png" -s 2 -b white >/dev/null
  echo "rendered ${f%.mmd}.png"
done
