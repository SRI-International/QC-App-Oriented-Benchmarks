"""
run_all.py — Run a standard set of QED-C benchmarks.

Usage:
    python -m qedcbench.run_all                                  # defaults: simulator, 2-8 qubits
    python -m qedcbench.run_all -b qasm_simulator
    python -m qedcbench.run_all -b ibm_sherbrooke -s 100 -max 6
    python -m qedcbench.run_all --list                           # show available benchmarks

Arguments match individual benchmark scripts:
    -a / --api             Quantum SDK (qiskit, cudaq)           [qiskit]
    -b / --backend_id      Backend name                          [qasm_simulator]
    -min / --min_qubits    Minimum circuit width                 [2]
    -max / --max_qubits    Maximum circuit width                 [8]
    -c / --max_circuits    Circuits per qubit group              [3]
    -s / --num_shots       Shots per circuit                     [100]
    -m / --method          Algorithm variant                     [1]
    --list                 Show available benchmarks and exit
"""

import argparse
import time
import sys

from qedclib import metrics

# Default benchmark set — spans complexity levels 1-3, all have uniform run() interface
DEFAULT_BENCHMARKS = [
    "hidden_shift",
    "bernstein_vazirani",
    "quantum_fourier_transform",
    "phase_estimation",
    "amplitude_estimation",
]

# All benchmarks with uniform run() interface
ALL_UNIFORM_BENCHMARKS = [
    "hidden_shift",
    "bernstein_vazirani",
    "quantum_fourier_transform",
    "grovers",
    "phase_estimation",
    "amplitude_estimation",
    "monte_carlo",
    "hamiltonian_simulation",
]


def list_benchmarks():
    """Print available benchmarks, highlighting the default set."""
    print("\nAvailable benchmarks with uniform run() interface:\n")
    for name in ALL_UNIFORM_BENCHMARKS:
        marker = " *" if name in DEFAULT_BENCHMARKS else "  "
        print(f"  {marker} {name}")
    print(f"\n  * = included in default set ({len(DEFAULT_BENCHMARKS)} benchmarks)")
    print(f"\nUse --benchmarks to specify a custom list, e.g.:")
    print(f"  python -m qedcbench.run_all --benchmarks hidden_shift,grovers,monte_carlo\n")


def import_benchmark(name):
    """Import a benchmark module by name."""
    if name == "hidden_shift":
        from qedcbench.hidden_shift import hs_benchmark
        return hs_benchmark
    elif name == "bernstein_vazirani":
        from qedcbench.bernstein_vazirani import bv_benchmark
        return bv_benchmark
    elif name == "quantum_fourier_transform":
        from qedcbench.quantum_fourier_transform import qft_benchmark
        return qft_benchmark
    elif name == "phase_estimation":
        from qedcbench.phase_estimation import pe_benchmark
        return pe_benchmark
    elif name == "amplitude_estimation":
        from qedcbench.amplitude_estimation import ae_benchmark
        return ae_benchmark
    elif name == "grovers":
        from qedcbench.grovers import grovers_benchmark
        return grovers_benchmark
    elif name == "monte_carlo":
        from qedcbench.monte_carlo import mc_benchmark
        return mc_benchmark
    elif name == "hamiltonian_simulation":
        from qedcbench.hamiltonian_simulation import hamiltonian_simulation_benchmark
        return hamiltonian_simulation_benchmark
    else:
        print(f"ERROR: Unknown benchmark '{name}'")
        return None


def run_benchmarks(benchmarks, run_args):
    """Run a list of benchmarks with shared arguments."""
    total = len(benchmarks)
    results = {}

    print(f"\n{'='*60}")
    print(f"QED-C Benchmark Suite — {total} benchmarks")
    print(f"Backend: {run_args.get('backend_id', 'qasm_simulator')}")
    print(f"Qubits: {run_args['min_qubits']}-{run_args['max_qubits']}, "
          f"circuits: {run_args['max_circuits']}, shots: {run_args['num_shots']}")
    print(f"{'='*60}\n")

    suite_start = time.time()

    for i, name in enumerate(benchmarks, 1):
        print(f"\n--- [{i}/{total}] {name} ---\n")

        bm = import_benchmark(name)
        if bm is None:
            results[name] = "SKIPPED (import failed)"
            continue

        t0 = time.time()
        try:
            bm.run(**run_args)
            elapsed = time.time() - t0
            results[name] = f"OK ({elapsed:.1f}s)"
            print(f"\n  completed in {elapsed:.1f}s")

            # Store metrics to data file (normally done inside plot_metrics)
            backend_id = run_args.get("backend_id", "qasm_simulator")
            api = run_args.get("api", "qiskit")
            app_title = f"Benchmark Results - {bm.benchmark_name} - {api.capitalize()}"
            metrics.store_app_metrics(backend_id, metrics.circuit_metrics,
                                     metrics.group_metrics, app_title,
                                     start_time=metrics.start_time, end_time=metrics.end_time)

        except Exception as e:
            elapsed = time.time() - t0
            results[name] = f"FAILED: {e}"
            print(f"\n  FAILED after {elapsed:.1f}s: {e}")

    suite_elapsed = time.time() - suite_start

    # Summary
    print(f"\n{'='*60}")
    print(f"Suite complete in {suite_elapsed:.1f}s")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name:.<40} {status}")
    print()

    # Show combined volumetric plot at the end
    backend_id = run_args.get("backend_id", "qasm_simulator")
    metrics.plot_all_app_metrics(backend_id)


# === Main ===

parser = argparse.ArgumentParser(
    description="Run QED-C Application-Oriented Benchmarks",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("-a", "--api", default="qiskit", help="Quantum SDK (default: qiskit)")
parser.add_argument("-b", "--backend_id", default="qasm_simulator", help="Backend (default: qasm_simulator)")
parser.add_argument("-min", "--min_qubits", type=int, default=2, help="Min qubits (default: 2)")
parser.add_argument("-max", "--max_qubits", type=int, default=8, help="Max qubits (default: 8)")
parser.add_argument("-c", "--max_circuits", type=int, default=3, help="Circuits per group (default: 3)")
parser.add_argument("-s", "--num_shots", type=int, default=100, help="Shots per circuit (default: 100)")
parser.add_argument("-m", "--method", type=int, default=1, help="Algorithm variant (default: 1)")
parser.add_argument("--benchmarks", default=None,
                    help="Comma-separated list of benchmarks to run (default: standard set)")
parser.add_argument("--list", action="store_true", help="Show available benchmarks and exit")

args = parser.parse_args()

if args.list:
    list_benchmarks()
    sys.exit(0)

# Determine which benchmarks to run
if args.benchmarks:
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
else:
    benchmarks = DEFAULT_BENCHMARKS

# Build the kwargs dict passed to each benchmark's run()
run_args = {
    "min_qubits": args.min_qubits,
    "max_qubits": args.max_qubits,
    "max_circuits": args.max_circuits,
    "num_shots": args.num_shots,
    "method": args.method,
    "api": args.api,
    "backend_id": args.backend_id,
    "draw_circuits": False,
    "plot_results": False,
}

run_benchmarks(benchmarks, run_args)
