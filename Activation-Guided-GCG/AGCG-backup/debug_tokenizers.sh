#!/usr/bin/env bash

# Collects diagnostic info for tokenizers install issues and writes to tokenizers_debug.log
# Usage: bash debug_tokenizers.sh

set -euo pipefail
LOGFILE="tokenizers_debug.log"

python - <<'PY' > "$LOGFILE" 2>&1
import sys, platform
print("=== PYTHON / PLATFORM ===")
print("python:", sys.version.replace("\n"," "))
print("platform:", platform.system(), platform.machine())
PY

{
  echo -e "\n=== PIP TOKENIZERS VERSIONS (first 40) ==="
  pip index versions tokenizers | head -n 40

  echo -e "\n=== TRY WHEEL DOWNLOAD (tokenizers>=0.13,<0.14) ==="
  pip download "tokenizers>=0.13,<0.14" --only-binary=:all: -d /tmp/tokenizers-wheel -v
} >> "$LOGFILE" 2>&1 || true

{
  echo -e "\n=== TOOLCHAIN CHECK ==="
  echo -n "cargo: "; (cargo --version || echo "missing")
  echo -n "rustc: "; (rustc --version || echo "missing")
  echo -n "pkg-config openssl: "; (pkg-config --modversion openssl || echo "missing")
} >> "$LOGFILE" 2>&1

{
  echo -e "\n=== VERBOSE TOKENIZERS INSTALL (no deps) ==="
  pip install "tokenizers>=0.13,<0.14" -v --no-deps
} >> "$LOGFILE" 2>&1 || true

echo "Diagnostics written to $LOGFILE"
