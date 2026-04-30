'''
Amplitude Estimation Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.

Three key functions, each independently callable:
  - get_circuits(): Create benchmark circuits (std + app args)
  - run_circuits(): Execute circuits and collect metrics (exec args)
  - plot_results(): Draw circuits and plot metrics (plot args)
  - run(): Convenience that calls all three
'''

import inspect
import time
import numpy as np

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from _common.qedc_init import qedc_get_kernel, qedc_is_leader
from _common import metrics

benchmark_name = "Amplitude Estimation"

np.random.seed(0)
verbose = False

# Circuit size grows significantly with num_qubits
MAX_QUBITS = 8


############### Get Circuits

def get_circuits(
    # Standard args (common across benchmarks)
    min_qubits=3, max_qubits=8, skip_qubits=1,
    max_circuits=3, num_shots=100,
    # App-specific args
    num_state_qubits=1,
    api=None,
):
    """Create Amplitude Estimation benchmark circuits.

    Standard args (common to all benchmarks):
        min_qubits: smallest circuit width (default 3)
        max_qubits: largest circuit width (default 8, clamped to 8)
        skip_qubits: increment between widths (default 1)
        max_circuits: max circuits per qubit group (default 3)
        num_shots: measurement shots, stored in metrics (default 100)

    App-specific args:
        num_state_qubits: number of state qubits (default 1)
        api: programming API; None = use qedc_set_api() value (default None)

    Returns (all_qcs, circuit_metrics) — nested circuit dict and creation metrics.
    """

    # Load the API-specific circuit kernel for this benchmark
    kernel = qedc_get_kernel("ae_kernel", api=api)

    if max_qubits > MAX_QUBITS:
        print(f"INFO: {benchmark_name} benchmark is limited to a maximum of {MAX_QUBITS} qubits.")
        max_qubits = MAX_QUBITS

    num_state_qubits = max(1, num_state_qubits)
    if max_qubits < num_state_qubits + 2:
        print(f"ERROR: AE Benchmark needs at least {num_state_qubits + 2} qubits to run")
        return {}, {}
    min_qubits = max(max(3, min_qubits), num_state_qubits + 2)
    skip_qubits = max(1, skip_qubits)

    metrics.init_metrics()

    # Build circuits at each qubit width
    all_qcs = {}
    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):

        np.random.seed(0)
        num_counting_qubits = num_qubits - num_state_qubits - 1
        num_circuits = min(2 ** (num_counting_qubits), max_circuits)

        print(f"************\nCreating [{num_circuits}] circuits with num_qubits = {num_qubits}")
        all_qcs[str(num_qubits)] = {}

        # Select random amplitude values as circuit inputs
        if 2**(num_counting_qubits) <= max_circuits:
            s_range = list(range(num_circuits))
        else:
            s_range = np.random.choice(2**(num_counting_qubits), num_circuits, False)

        # Create each circuit and store in the dict
        for s_int in s_range:
            circuit_id = s_int

            ts = time.time()
            a_ = a_from_s_int(s_int, num_counting_qubits)
            qc = kernel.AmplitudeEstimation(num_state_qubits, num_counting_qubits, a_)
            metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time() - ts)

            # collapse the 3 sub-circuit levels used in this benchmark (for qiskit)
            qc2 = qc.decompose().decompose().decompose()

            all_qcs[str(num_qubits)][str(circuit_id)] = qc2

    return all_qcs, metrics.circuit_metrics


############### Result Analysis (used by run_circuits)

def analyze_and_print_result(qc, result, num_qubits, num_shots, s_int=None, num_state_qubits=1):
    """Compare measured results against expected amplitude distribution and compute fidelity."""
    num_counting_qubits = num_qubits - num_state_qubits - 1
    counts = result.get_counts(qc)

    a = a_from_s_int(s_int, num_counting_qubits)
    correct_dist = a_to_bitstring(a, num_counting_qubits)
    thermal_dist = metrics.uniform_dist(num_counting_qubits)

    if verbose:
        app_counts = bitstring_to_a(counts, num_counting_qubits)
        app_correct_dist = {a: 1.0}
        print(f"For amplitude {a}, expected: {correct_dist} measured: {counts}")
        print(f"For amplitude {a}, app expected: {app_correct_dist} measured: {app_counts}")

    fidelity = metrics.polarization_fidelity(counts, correct_dist, thermal_dist)
    if verbose: print(f"  ... fidelity: {fidelity}")
    return counts, fidelity

def a_to_bitstring(a, num_counting_qubits):
    m = num_counting_qubits
    num1 = round(np.arcsin(np.sqrt(a)) / np.pi * 2**m)
    num2 = round((np.pi - np.arcsin(np.sqrt(a))) / np.pi * 2**m)
    if num1 != num2 and num2 < 2**m and num1 < 2**m:
        return {format(num1, "0"+str(m)+"b"): 0.5, format(num2, "0"+str(m)+"b"): 0.5}
    else:
        return {format(num1, "0"+str(m)+"b"): 1}

def bitstring_to_a(counts, num_counting_qubits):
    est_counts = {}
    m = num_counting_qubits
    precision = int(num_counting_qubits / (np.log2(10))) + 2
    for key, r in counts.items():
        num = int(key, 2) / (2**m)
        a_est = round((np.sin(np.pi * num))**2, precision)
        if a_est not in est_counts:
            est_counts[a_est] = 0
        est_counts[a_est] += r
    return est_counts

def a_from_s_int(s_int, num_counting_qubits):
    theta = s_int * np.pi / (2**num_counting_qubits)
    precision = int(num_counting_qubits / (np.log2(10))) + 2
    return round(np.sin(theta)**2, precision)


############### Run Circuits

def run_circuits(all_qcs,
    num_shots=100, max_batch_size=None,
    num_state_qubits=1,
    backend_id=None, provider_backend=None,
    hub="ibm-q", group="open", project="main",
    exec_options=None, context=None, api=None,
):
    """Execute benchmark circuits and collect metrics.

    Args:
        all_qcs: circuit dict from get_circuits()
        num_shots: measurement shots per circuit (default 100)
        max_batch_size: max circuits per batch; None = no limit (default None)
        num_state_qubits: for result analysis (default 1)
        backend_id: backend identifier (default None = qasm_simulator)
        provider_backend: provider backend instance (default None)
        hub, group, project: IBMQ credentials (defaults "ibm-q"/"open"/"main")
        exec_options: additional execution options dict (default None)
        context: context identifier for metrics (default None)
        api: programming API if not already initialized (default None)
    """
    qedc_get_kernel("ae_kernel", api=api)
    import execute as ex

    if context is None:
        context = f"{benchmark_name} Benchmark"

    # Result handler: called for each circuit after execution completes
    def execution_handler(qc, result, num_qubits, circuit_id, num_shots):
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, num_shots,
                s_int=int(circuit_id), num_state_qubits=num_state_qubits)
        metrics.store_metric(num_qubits, circuit_id, 'fidelity', fidelity)

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
    num_shots=100, max_circuits=3,
    api=None, draw_circuits=True, plot_results=True,
):
    """Draw sample circuit and plot benchmark metrics.

    Args:
        num_shots: shots, for plot subtitle (default 100)
        max_circuits: circuit reps, for plot subtitle (default 3)
        api: programming API name for plot title (default None)
        draw_circuits: draw a sample circuit diagram (default True)
        plot_results: generate metrics plots (default True)
    """
    kernel = qedc_get_kernel("ae_kernel", api=api)

    if qedc_is_leader():
        if draw_circuits:
            kernel.kernel_draw()

        if plot_results:
            options = {"shots": num_shots, "reps": max_circuits}
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

    # Step 1: Create the benchmark circuits
    all_qcs, circuit_metrics = get_circuits(**_for(get_circuits))

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
    parser = argparse.ArgumentParser(description="Amplitude Estimation Benchmark")
    parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits", type=int)
    parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)
    parser.add_argument("--num_state_qubits", "-nsq", default=1, help="Number of State Qubits", type=int)
    parser.add_argument("--max_batch_size", "-mbs", default=None, help="Max circuits per execution batch", type=int)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--noplot", "-nop", action="store_true", help="Do not plot results")
    parser.add_argument("--nodraw", "-nod", action="store_true", help="Do not draw circuit diagram")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    verbose = args.verbose
    if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits

    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots, num_state_qubits=args.num_state_qubits,
        backend_id=args.backend_id,
        exec_options={"noise_model": None} if args.nonoise else {},
        api=args.api, max_batch_size=args.max_batch_size,
        draw_circuits=not args.nodraw, plot_results=not args.noplot)
