#!/usr/bin/env bash
set -euo pipefail

# ===== pretty header =====
echo "============================================================"
echo "Testing package implementation (default to Hidden Shift Benchmark)"
echo

cd ..

# Use the directory of this script as the working base (like %~dp0)
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Args:
# - 0 args -> defaults
# - 2 args -> use both
# - anything else -> error
if [ "$#" -eq 0 ]; then
  folder="hidden_shift"
  bmname="hs_benchmark"
elif [ "$#" -eq 2 ]; then
  folder="$1"
  bmname="$2"
else
  echo "Error: Both arguments required or none" >&2
  exit 1
fi

# Which Python to use (fallback to python3, then python)
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="python"
fi

echo "folder=$folder"
echo "bmname=$bmname"

# ----- run in BM directory -----
echo "... run in BM directory ..."
pushd "$folder" >/dev/null
"$PYTHON" "${bmname}.py" -a cudaq
popd >/dev/null

# pause (only if interactive)
if [ -t 0 ]; then
  read -r -n 1 -s -p "Press any key to continue..." _; echo
fi

# ----- run at top level -----
echo "... run at top level ..."
"$PYTHON" "${folder}/${bmname}.py"  -a cudaq

# pause (only if interactive)
if [ -t 0 ]; then
  read -r -n 1 -s -p "Press any key to continue..." _; echo
fi

# ----- run at top level as a module -----
echo "... run at top level as a module"
"$PYTHON" -m "${folder}.${bmname}"  -a cudaq
