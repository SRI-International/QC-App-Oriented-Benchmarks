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
# - 0 args -> defaults (hidden_shift, no extra args)
# - 2+ args -> first two are folder/bmname, rest passed to Python
#
# Examples:
#   ./package_test_1.sh
#   ./package_test_1.sh hydrogen_lattice hydrogen_lattice_benchmark
#   ./package_test_1.sh hydrogen_lattice hydrogen_lattice_benchmark -a qiskit
#   ./package_test_1.sh hamlib hamlib_simulation_benchmark -a cudaq --num_qubits 4

if [ "$#" -eq 0 ]; then
  folder="hidden_shift"
  bmname="hs_benchmark"
  extra_args=""
elif [ "$#" -ge 2 ]; then
  folder="$1"
  bmname="$2"
  shift 2
  extra_args="$*"
else
  echo "Error: Need 0 args (defaults) or 2+ args (folder bmname [python_args...])" >&2
  exit 1
fi

# Which Python to use (fallback to python3, then python)
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="python"
fi

echo "folder=$folder"
echo "bmname=$bmname"
echo "extra_args=$extra_args"

# ----- run in BM directory -----
echo "... run in BM directory ..."
pushd "$folder" >/dev/null
"$PYTHON" "${bmname}.py" $extra_args
popd >/dev/null

# pause (only if interactive)
if [ -t 0 ]; then
  read -r -n 1 -s -p "Press any key to continue..." _; echo
fi

# ----- run at top level -----
echo "... run at top level ..."
"$PYTHON" "${folder}/${bmname}.py" $extra_args

# pause (only if interactive)
if [ -t 0 ]; then
  read -r -n 1 -s -p "Press any key to continue..." _; echo
fi

# ----- run at top level as a module -----
echo "... run at top level as a module"
"$PYTHON" -m "${folder}.${bmname}" $extra_args
