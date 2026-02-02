#!/usr/bin/env bash
set -euo pipefail

# ===== pretty header =====
echo "============================================================"
echo "Testing package implementation with Hamlib Benchmark (with obs test)"
echo

cd ..

# Run from the script's own directory (like %~dp0)
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Args:
# - 0 args -> defaults (hamlib, -a qiskit)
# - args starting with "-" -> hamlib + all args passed to Python
#
# Examples:
#   ./package_test_2.sh
#   ./package_test_2.sh -a cudaq
#   ./package_test_2.sh -a qiskit --num_qubits 6

folder="hamlib"
bmname="hamlib_simulation_benchmark"

if [ "$#" -eq 0 ]; then
  extra_args="-a qiskit"
else
  extra_args="$*"
fi

# Choose Python (prefer python3, fall back to python)
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="python"
fi

echo "folder=$folder"
echo "bmname=$bmname"
echo "extra_args=$extra_args -nop -nod"

# ----- run in BM directory -----
echo "... run in BM directory ..."
pushd "$folder" >/dev/null
"$PYTHON" "${bmname}.py" $extra_args -nop -nod
popd >/dev/null

# pause (only if interactive)
if [ -t 0 ]; then
  read -r -n 1 -s -p "Press any key to continue..." _; echo
fi

# ----- run at top level -----
echo "... run at top level ..."
"$PYTHON" "${folder}/${bmname}.py" $extra_args -nop -nod

# pause (only if interactive)
if [ -t 0 ]; then
  read -r -n 1 -s -p "Press any key to continue..." _; echo
fi

# ----- run at top level as a module -----
echo "... run at top level as a module"
"$PYTHON" -m "${folder}.${bmname}" $extra_args -nop -nod

# pause (only if interactive)
if [ -t 0 ]; then
  read -r -n 1 -s -p "Press any key to continue..." _; echo
fi

# ----- run at top level as a module (observable test) -----
echo "... run at top level as a module (observable test)"
"$PYTHON" -m "${folder}.${bmname}" $extra_args -nop -nod -obs
