#!/usr/bin/env bash
set -euo pipefail

echo "Testing package implementation with Hamlib Benchmark"
echo

cd ..

# Run from the script's own directory (like %~dp0)
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Choose Python (prefer python3, fall back to python)
PYTHON="${PYTHON:-python3} -u"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON="python"

# ----- run in BM directory -----
echo "... run in BM directory ..."
pushd hamlib >/dev/null
"$PYTHON" hamlib_simulation_benchmark.py -a cudaq -nop -nod
popd >/dev/null

# pause (only if interactive)
if [ -t 0 ]; then
  read -r -n 1 -s -p "Press any key to continue..." _; echo
fi

# ----- run at top level -----
echo "... run at top level ..."
"$PYTHON" hamlib/hamlib_simulation_benchmark.py -a cudaq -nop -nod

# pause (only if interactive)
if [ -t 0 ]; then
  read -r -n 1 -s -p "Press any key to continue..." _; echo
fi

# ----- run at top level as a module -----
echo "... run at top level as a module"
"$PYTHON" -m hamlib.hamlib_simulation_benchmark -a cudaq -nop -nod

# pause (only if interactive)
if [ -t 0 ]; then
  read -r -n 1 -s -p "Press any key to continue..." _; echo
fi

# ----- run at top level as a module (observable test) -----
echo "... run at top level as a module (observable test)"
"$PYTHON" -m hamlib.hamlib_simulation_benchmark -a cudaq -nop -nod -obs 
