#!/usr/bin/env bash
set -euo pipefail

# Run package_test_1 for all 16 benchmarks.
# Uses package_test_2 for hamlib (includes observable test).
# Comment out any lines below to skip specific benchmarks.

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
RUNTEST_1="$SCRIPT_DIR/package_test_1.sh"
RUNTEST_2="$SCRIPT_DIR/package_test_2.sh"

echo "============================================================"
echo "Running all benchmarks via package_test_1 (and _2 for hamlib)"
echo "============================================================"

# Group 1 - Simple deterministic
"$RUNTEST_1" bernstein_vazirani bv_benchmark "$@"
"$RUNTEST_1" deutsch_jozsa dj_benchmark "$@"
"$RUNTEST_1" grovers grovers_benchmark "$@"
"$RUNTEST_1" hidden_shift hs_benchmark "$@"

# Group 2 - Transform / estimation
"$RUNTEST_1" amplitude_estimation ae_benchmark "$@"
"$RUNTEST_1" phase_estimation pe_benchmark "$@"
"$RUNTEST_1" quantum_fourier_transform qft_benchmark "$@"

# Group 3 - Complex (qiskit subdir)
"$RUNTEST_1" hydrogen_lattice hydrogen_lattice_benchmark "$@"
"$RUNTEST_1" maxcut maxcut_benchmark "$@"
"$RUNTEST_1" image_recognition image_recognition_benchmark "$@"
"$RUNTEST_1" hhl hhl_benchmark "$@"
"$RUNTEST_1" shors shors_benchmark "$@"
"$RUNTEST_1" vqe vqe_benchmark "$@"

# Group 4 - Simulation
"$RUNTEST_1" hamiltonian_simulation hamiltonian_simulation_benchmark "$@"
"$RUNTEST_1" monte_carlo mc_benchmark "$@"

# Hamlib uses package_test_2 (includes observable test)
"$RUNTEST_2" "$@"

echo "============================================================"
echo "All benchmarks complete."
echo "============================================================"
