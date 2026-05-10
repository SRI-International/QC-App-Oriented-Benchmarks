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
import os
import time
import sys

from qedclib import metrics

# Default benchmark sets by API
DEFAULT_BENCHMARKS_QISKIT = [
    "hidden_shift",
    "bernstein_vazirani",
    "quantum_fourier_transform",
    "phase_estimation",
    "amplitude_estimation",
]

DEFAULT_BENCHMARKS_CUDAQ = [
    "hidden_shift",
    "bernstein_vazirani",
    "quantum_fourier_transform",
    "phase_estimation",
]

# All benchmarks with uniform run() interface, by API
ALL_UNIFORM_BENCHMARKS_QISKIT = [
    "hidden_shift",
    "bernstein_vazirani",
    "quantum_fourier_transform",
    "grovers",
    "phase_estimation",
    "amplitude_estimation",
    "monte_carlo",
    "hamiltonian_simulation",
]

ALL_UNIFORM_BENCHMARKS_CUDAQ = [
    "hidden_shift",
    "bernstein_vazirani",
    "quantum_fourier_transform",
    "phase_estimation",
]


def list_benchmarks(api="qiskit"):
    """Print available benchmarks, highlighting the default set."""
    all_bms = ALL_UNIFORM_BENCHMARKS_CUDAQ if api == "cudaq" else ALL_UNIFORM_BENCHMARKS_QISKIT
    default_bms = DEFAULT_BENCHMARKS_CUDAQ if api == "cudaq" else DEFAULT_BENCHMARKS_QISKIT

    print(f"\nAvailable benchmarks for {api}:\n")
    for name in all_bms:
        marker = " *" if name in default_bms else "  "
        print(f"  {marker} {name}")
    print(f"\n  * = included in default set ({len(default_bms)} benchmarks)")
    print(f"\nUse --benchmarks to specify a custom list, e.g.:")
    print(f"  python -m qedcbench.run_all --benchmarks hidden_shift,grovers\n")


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


def configure_backend(backend_id, run_args):
    """Set up run_args for the given backend_id.

    Resolves provider credentials and adds the params that run_circuits()
    needs to call set_execution_target() correctly.
    """

    # IBM backends: "ibm" for least-busy, or specific like "ibm_sherbrooke"
    if backend_id.startswith("ibm"):
        ibm_instance = os.environ.get("IBM_INSTANCE", "")
        if not ibm_instance:
            print("  WARNING: IBM_INSTANCE not set — will use saved default credentials.")
            print("  Set IBM_INSTANCE to your CRN or service name to target a specific plan.")

        if backend_id == "ibm":
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(channel="ibm_cloud", instance=ibm_instance)
            backend = service.least_busy(simulator=False, operational=True, min_num_qubits=100)
            backend_id = backend.name
            print(f"  Least-busy IBM backend: {backend_id}")

        run_args["backend_id"] = backend_id
        run_args["hub"] = ""
        run_args["group"] = ""
        run_args["project"] = ibm_instance
        run_args["exec_options"] = {"use_ibm_quantum_platform": False, "use_sessions": False}
        return

    # IonQ backends: "ionq" for simulator, "ionq_qpu" for real hardware
    if backend_id.startswith("ionq"):
        from qiskit_ionq import IonQProvider
        provider = IonQProvider()
        ionq_backend = "ionq_simulator" if backend_id == "ionq" else backend_id
        run_args["provider_backend"] = provider.get_backend(ionq_backend)
        return

    # IQM backends: "iqm" for Garnet via Resonance
    if backend_id.startswith("iqm"):
        from iqm.qiskit_iqm import IQMProvider
        iqm_server_url = "https://resonance.meetiqm.com"
        quantum_computer = "garnet"
        iqm_token = os.environ.get("IQM_API_TOKEN")

        provider = IQMProvider(iqm_server_url, quantum_computer=quantum_computer, token=iqm_token)
        run_args["backend_id"] = quantum_computer
        run_args["provider_backend"] = provider.get_backend()
        return

    # Local simulators: no special setup needed


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
    try:
        backend_id = run_args.get("backend_id", "qasm_simulator")
        metrics.plot_all_app_metrics(backend_id)
    except KeyboardInterrupt:
        print("... KeyboardInterrupt: plot display cancelled.")
        pass


# === Main ===

parser = argparse.ArgumentParser(
    description="Run QED-C Application-Oriented Benchmarks",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("-a", "--api", default="qiskit", help="Quantum SDK (default: qiskit)")
parser.add_argument("-b", "--backend_id", default=None, help="Backend (default: qasm_simulator for qiskit, nvidia for cudaq)")
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
    list_benchmarks(args.api)
    sys.exit(0)

# Set default backend based on API
if args.backend_id is None:
    args.backend_id = "nvidia" if args.api == "cudaq" else "qasm_simulator"

# Determine which benchmarks to run
if args.benchmarks:
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
else:
    benchmarks = DEFAULT_BENCHMARKS_CUDAQ if args.api == "cudaq" else DEFAULT_BENCHMARKS_QISKIT

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

# Configure hardware backend (IBM, IonQ, IQM) — adds provider params to run_args
configure_backend(args.backend_id, run_args)

run_benchmarks(benchmarks, run_args)
