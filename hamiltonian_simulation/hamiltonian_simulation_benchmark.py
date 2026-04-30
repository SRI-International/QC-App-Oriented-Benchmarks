'''
Hamiltonian Simulation Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.

Three key functions, each independently callable:
  - get_circuits(): Create benchmark circuits (std + app args)
  - run_circuits(): Execute circuits and collect metrics (exec args)
  - plot_results(): Draw circuits and plot metrics (plot args)
  - run(): Convenience that calls all three
'''

import inspect
import json
import os
import time
import numpy as np

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from _common.qedc_init import qedc_get_kernel, qedc_is_leader
from _common import metrics

benchmark_name = "Hamiltonian Simulation"

np.random.seed(0)
verbose = False

# Import precalculated data to compare against
filename = os.path.join(os.path.dirname(__file__), "_common", "precalculated_data.json")
with open(filename, 'r') as file:
    data = file.read()
precalculated_data = json.loads(data)

# we only generate simulation data for up to 14 qubits
MAX_QUBITS_CLASSICAL = 14


############### Get Circuits

def get_circuits(
    # Standard args (common across benchmarks)
    min_qubits=2, max_qubits=8, skip_qubits=1,
    max_circuits=3, num_shots=100, method=1,
    # App-specific args
    hamiltonian="heisenberg", use_XX_YY_ZZ_gates=False,
    random_pauli_flag=False, init_state=None,
    K=None, t=None,
    api=None,
):
    """Create Hamiltonian Simulation benchmark circuits.

    Standard args (common to all benchmarks):
        min_qubits: smallest circuit width (default 2)
        max_qubits: largest circuit width (default 8)
        skip_qubits: increment between widths (default 1)
        max_circuits: max circuits per qubit group (default 3)
        num_shots: measurement shots, stored in metrics (default 100)
        method: 1=trotterized, 2=exact classical, 3=mirror circuit (default 1)

    App-specific args:
        hamiltonian: "heisenberg" or "tfim" (default "heisenberg")
        use_XX_YY_ZZ_gates: use unoptimized XX, YY, ZZ gates (default False)
        random_pauli_flag: use random Pauli operations (default False)
        init_state: initial state "checkerboard" or "ghz"; None = auto (default None)
        K: number of Trotter steps; None = use precalculated (default None)
        t: time of simulation; None = use precalculated (default None)
        api: programming API; None = use qedc_set_api() value (default None)

    Returns (all_qcs, circuit_metrics) — nested circuit dict and creation metrics.
    """

    # Load the API-specific circuit kernel for this benchmark
    kernel = qedc_get_kernel("hamiltonian_simulation_kernel", api=api)

    max_qubits = max(2, max_qubits)
    min_qubits = min(max(2, min_qubits), max_qubits)
    skip_qubits = max(1, skip_qubits)

    if not (hamiltonian == "heisenberg" or hamiltonian == "tfim"):
        print(f"ERROR: invalid Hamiltonian name: {hamiltonian}")
        return {}, {}

    # set the initial state if not given
    if init_state is None:
        init_state = "ghz" if hamiltonian == "tfim" else "checkerboard"

    if use_XX_YY_ZZ_gates:
        print("... using unoptimized XX YY ZZ gates")

    print(f"... using init_state = {init_state}")
    print(f"... using random_pauli_flag = {random_pauli_flag}")

    metrics.init_metrics()

    # Build circuits at each qubit width
    all_qcs = {}
    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):

        # methods 1 and 2 use pre-calculated data, limited to MAX_QUBITS_CLASSICAL
        if method == 1 or method == 2:
            if num_qubits > MAX_QUBITS_CLASSICAL:
                print(f"ERROR: cannot execute method 1 or 2 above {MAX_QUBITS_CLASSICAL} qubits")
                break

        np.random.seed(0)
        num_circuits = max(1, max_circuits)

        print(f"************\nCreating [{num_circuits}] circuits with num_qubits = {num_qubits}")
        all_qcs[str(num_qubits)] = {}

        # Parameters of simulation from precalculated data
        w = precalculated_data['w']
        k = precalculated_data['k']
        t_val = precalculated_data['t']
        hx = precalculated_data['hx'][:num_qubits]
        hz = precalculated_data['hz'][:num_qubits]

        # Create each circuit and store in the dict
        for circuit_id in range(num_circuits):
            ts = time.time()

            qc = kernel.HamiltonianSimulation(num_qubits, K=k, t=t_val,
                    hamiltonian=hamiltonian,
                    w=w, hx=hx, hz=hz,
                    use_XX_YY_ZZ_gates=use_XX_YY_ZZ_gates,
                    method=method, random_pauli_flag=random_pauli_flag)

            metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time() - ts)

            all_qcs[str(num_qubits)][str(circuit_id)] = qc

    return all_qcs, metrics.circuit_metrics


############### Result Analysis (used by run_circuits)

def analyze_and_print_result(qc, result, num_qubits, num_shots,
            type=None, hamiltonian=None, method=None, random_pauli_flag=False, init_state=None):
    """Compare measured results against precalculated distribution and compute fidelity."""
    counts = result.get_counts(qc)
    if verbose:
        print_top_measurements(f"For type {type} measured counts = ", counts, 100)

    hamiltonian = hamiltonian.strip().lower()

    if method == 1 and hamiltonian == "heisenberg":
        correct_dist = precalculated_data[f"Heisenberg - Qubits{num_qubits}"]
    elif method == 2 and hamiltonian == "heisenberg":
        correct_dist = precalculated_data[f"Exact Heisenberg - Qubits{num_qubits}"]
    elif method == 1 and hamiltonian == "tfim":
        correct_dist = precalculated_data[f"TFIM - Qubits{num_qubits}"]
    elif method == 2 and hamiltonian == "tfim":
        correct_dist = precalculated_data[f"Exact TFIM - Qubits{num_qubits}"]
    elif method == 3:
        correct_dist = key_from_initial_state(num_qubits, num_shots, init_state, random_pauli_flag)
    else:
        raise ValueError("Method is not 1 or 2 or 3, or hamiltonian is not tfim or heisenberg.")

    if verbose:
        print_top_measurements(f"Correct dist = ", correct_dist, 100)

    fidelity = metrics.polarization_fidelity(counts, correct_dist)
    return counts, fidelity

def key_from_initial_state(num_qubits, num_shots, init_state, random_pauli_flag):
    """Create expected distribution for mirror circuit method."""
    def generate_pattern(starting_bit):
        return ''.join([str((i + starting_bit) % 2) for i in range(num_qubits)])

    correct_dist = {}
    if init_state == "checkerboard":
        if random_pauli_flag:
            starting_bit = 0 if num_qubits % 2 != 0 else 1
        else:
            starting_bit = 1 if num_qubits % 2 != 0 else 0
        correct_dist[generate_pattern(starting_bit)] = num_shots
    elif init_state == "ghz":
        correct_dist = {'0' * num_qubits: num_shots/2, '1' * num_qubits: num_shots/2}
    return correct_dist

def print_top_measurements(label, counts, top_n):
    """Print the top N measurements from a counts dictionary."""
    if label is not None: print(label)
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    total_measurements = len(sorted_counts)
    if top_n >= total_measurements:
        top_counts = sorted_counts
        more_counts = []
    else:
        top_counts = sorted_counts[:top_n]
        more_counts = sorted_counts[top_n:]

    print("{", end=" ")
    for i, (measurement, count) in enumerate(top_counts):
        print(f"'{measurement}': {round(count,6)}", end="")
        if i < len(top_counts) - 1:
            print(",", end=" ")
    if more_counts:
        print(f", ... and {len(more_counts)} more.")
    else:
        print(" }")


############### Run Circuits

def run_circuits(all_qcs,
    num_shots=100, method=1, max_batch_size=None,
    hamiltonian="heisenberg", random_pauli_flag=False, init_state=None,
    backend_id=None, provider_backend=None,
    hub="ibm-q", group="open", project="main",
    exec_options=None, context=None, api=None,
):
    """Execute benchmark circuits and collect metrics.

    Args:
        all_qcs: circuit dict from get_circuits()
        num_shots: measurement shots per circuit (default 100)
        method: algorithm method, for result analysis (default 1)
        max_batch_size: max circuits per batch; None = no limit (default None)
        hamiltonian: hamiltonian name, for result analysis (default "heisenberg")
        random_pauli_flag: for result analysis (default False)
        init_state: for result analysis (default None)
        backend_id: backend identifier (default None = qasm_simulator)
        provider_backend: provider backend instance (default None)
        hub, group, project: IBMQ credentials (defaults "ibm-q"/"open"/"main")
        exec_options: additional execution options dict (default None)
        context: context identifier for metrics (default None)
        api: programming API if not already initialized (default None)
    """
    qedc_get_kernel("hamiltonian_simulation_kernel", api=api)
    import execute as ex

    if context is None:
        context = f"{benchmark_name} Benchmark"

    # set init_state default if not provided (needed for analyze)
    if init_state is None:
        init_state = "ghz" if hamiltonian == "tfim" else "checkerboard"

    # Result handler: called for each circuit after execution completes
    def execution_handler(qc, result, num_qubits, circuit_id, num_shots):
        num_qubits = int(num_qubits)
        counts, expectation_a = analyze_and_print_result(qc, result, num_qubits, num_shots,
                type=circuit_id, hamiltonian=hamiltonian, method=method,
                random_pauli_flag=random_pauli_flag, init_state=init_state)
        metrics.store_metric(num_qubits, circuit_id, 'fidelity', expectation_a)

    # Set up execution target and submit all circuits as a batch
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)

    ex.compute_all_circuit_metrics(all_qcs)
    ex.submit_circuits(all_qcs, num_shots=num_shots, max_batch_size=max_batch_size)
    metrics.finalize_all_groups()


############### Plot Results

def plot_results(
    method=1, num_shots=100, max_circuits=3,
    hamiltonian="heisenberg", use_XX_YY_ZZ_gates=False,
    random_pauli_flag=False,
    api=None, draw_circuits=True, plot_results=True,
):
    """Draw sample circuit and plot benchmark metrics.

    Args:
        method: algorithm method, for plot title (default 1)
        num_shots: shots, for plot subtitle (default 100)
        max_circuits: circuit reps, for plot subtitle (default 3)
        hamiltonian: hamiltonian name, for kernel_draw and plot title (default "heisenberg")
        use_XX_YY_ZZ_gates: for kernel_draw (default False)
        random_pauli_flag: for kernel_draw (default False)
        api: programming API name for plot title (default None)
        draw_circuits: draw a sample circuit diagram (default True)
        plot_results: generate metrics plots (default True)
    """
    kernel = qedc_get_kernel("hamiltonian_simulation_kernel", api=api)

    if qedc_is_leader():
        if draw_circuits:
            kernel.kernel_draw(hamiltonian, use_XX_YY_ZZ_gates, method, random_pauli_flag)

        if plot_results:
            options = {"ham": hamiltonian, "method": method, "shots": num_shots, "reps": max_circuits}
            if use_XX_YY_ZZ_gates: options.update({"xyz": use_XX_YY_ZZ_gates})
            metrics.plot_metrics(
                f"Benchmark Results - {benchmark_name} - {api if api is not None else 'Qiskit'}",
                options=options)


############### Run (convenience)

def run(**kwargs):
    """Create circuits, execute, and plot. Accepts any arg from
    get_circuits(), run_circuits(), or plot_results()."""

    def _for(func):
        return {k: kwargs[k] for k in kwargs if k in inspect.signature(func).parameters}

    get_circuits_only = kwargs.pop('get_circuits', False)

    print(f"{benchmark_name} Benchmark Program - Qiskit")

    # Step 1: Create the benchmark circuits
    all_qcs, circuit_metrics = get_circuits(**_for(get_circuits))
    if not all_qcs: return

    # Step 2: If user just wants circuits, return them now
    if get_circuits_only:
        print(f"************\nReturning circuits and circuit information")
        return all_qcs, circuit_metrics

    # Step 3: Execute circuits on the target backend
    run_circuits(all_qcs, **_for(run_circuits))

    # Step 4: Draw sample circuit and plot metrics
    plot_results(**_for(plot_results))


#######################
# MAIN

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Hamiltonian Simulation Benchmark")
    parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    parser.add_argument("--target", "-t", default=None, help="Target Backend", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits (min = max = N)", type=int)
    parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)
    parser.add_argument("--hamiltonian", "-ham", default="heisenberg", help="Name of Hamiltonian", type=str)
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument("--use_XX_YY_ZZ_gates", action="store_true", help="Use explicit XX, YY, ZZ gates")
    parser.add_argument("--max_batch_size", "-mbs", default=None, help="Max circuits per execution batch", type=int)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--random_pauli_flag", "-ranp", action="store_true", help="random pauli flag")
    parser.add_argument("--init_state", "-init", default=None, help="initial state")
    parser.add_argument("--noplot", "-nop", action="store_true", help="Do not plot results")
    parser.add_argument("--nodraw", "-nod", action="store_true", help="Do not draw circuit diagram")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    verbose = args.verbose
    if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits

    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots, hamiltonian=args.hamiltonian,
        method=args.method, random_pauli_flag=args.random_pauli_flag,
        use_XX_YY_ZZ_gates=args.use_XX_YY_ZZ_gates,
        init_state=args.init_state, backend_id=args.backend_id,
        exec_options={"noise_model": None} if args.nonoise else {},
        api=args.api, max_batch_size=args.max_batch_size,
        draw_circuits=not args.nodraw, plot_results=not args.noplot)
