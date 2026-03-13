#!/usr/bin/env bash
set -euo pipefail

# Run package_test_1 for CUDA-Q supported benchmarks.
# Uses package_test_2 for hamlib (includes observable test).
# Comment out any lines below to skip specific benchmarks.

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
RUNTEST_1="$SCRIPT_DIR/package_test_1.sh"
RUNTEST_2="$SCRIPT_DIR/package_test_2.sh"

echo "============================================================"
echo "Running CUDA-Q benchmarks"
echo "============================================================"

"$RUNTEST_1" bernstein_vazirani bv_benchmark -a cudaq "$@"
"$RUNTEST_1" hidden_shift hs_benchmark -a cudaq "$@"
"$RUNTEST_1" phase_estimation pe_benchmark -a cudaq "$@"
"$RUNTEST_1" quantum_fourier_transform qft_benchmark -a cudaq "$@"

# Hamlib uses package_test_2 (includes observable test)
"$RUNTEST_2" -a cudaq "$@"

echo "============================================================"
echo "All CUDA-Q benchmarks complete."
echo "============================================================"
