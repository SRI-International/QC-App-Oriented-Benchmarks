'''
Deutsch-Jozsa Benchmark Program
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

from qedclib.qedc_init import qedc_get_kernel, qedc_is_leader
from qedclib import metrics

benchmark_name = "Deutsch-Jozsa"

np.random.seed(0)
verbose = False


############### Get Circuits

def get_circuits(
    # Standard args (common across benchmarks)
    min_qubits=3, max_qubits=8, skip_qubits=1,
    max_circuits=3, num_shots=100,
    api=None,
):
    """Create Deutsch-Jozsa benchmark circuits.

    Standard args (common to all benchmarks):
        min_qubits: smallest circuit width (default 3)
        max_qubits: largest circuit width (default 8)
        skip_qubits: increment between widths (default 1)
        max_circuits: max circuits per qubit group (default 3)
        num_shots: measurement shots, stored in metrics (default 100)
        api: programming API; None = use qedc_set_api() value (default None)

    Note: DJ creates exactly 2 circuits per width (constant and balanced).

    Returns (all_qcs, circuit_metrics) — nested circuit dict and creation metrics.
    """

    # Load the API-specific circuit kernel for this benchmark (e.g. qiskit or cudaq)
    kernel = qedc_get_kernel("dj_kernel", api=api)

    max_qubits = max(3, max_qubits)
    min_qubits = min(max(3, min_qubits), max_qubits)
    skip_qubits = max(1, skip_qubits)

    metrics.init_metrics()

    # Build circuits at each qubit width (2 circuits: constant and balanced)
    all_qcs = {}
    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):

        num_circuits = min(2, max_circuits)

        print(f"************\nCreating [{num_circuits}] circuits with num_qubits = {num_qubits}")
        all_qcs[str(num_qubits)] = {}

        # Create one constant (type=0) and one balanced (type=1) circuit
        for type in range(num_circuits):

            circuit_id = type

            ts = time.time()
            qc = kernel.DeutschJozsa(num_qubits, type)
            metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time()-ts)

            # collapse the sub-circuit levels used in this benchmark (for qiskit)
            qc2 = qc.decompose()

            all_qcs[str(num_qubits)][str(circuit_id)] = qc2

    return all_qcs, metrics.circuit_metrics


############### Result Analysis (used by run_circuits)

def analyze_and_print_result(qc, result, num_qubits, num_shots, type=None):
    """Compare measured results against expected output and compute fidelity."""
    input_size = num_qubits - 1
    counts = result.get_counts(qc)
    if verbose: print(f"For type {type} measured: {counts}")

    # constant (type=0) should measure all zeros; balanced (type=1) should measure all ones
    if type == 0: key = '0'*input_size
    else: key = '1'*input_size

    correct_dist = {key: 1.0}
    fidelity = metrics.polarization_fidelity(counts, correct_dist)
    return counts, fidelity


############### Run Circuits

def run_circuits(all_qcs,
    num_shots=100, max_batch_size=None,
    backend_id=None, provider_backend=None,
    hub="ibm-q", group="open", project="main",
    exec_options=None, context=None, api=None,
):
    """Execute benchmark circuits and collect metrics.

    Args:
        all_qcs: circuit dict from get_circuits()
        num_shots: measurement shots per circuit (default 100)
        max_batch_size: max circuits per batch; None = no limit (default None)
        backend_id: backend identifier (default None = qasm_simulator)
        provider_backend: provider backend instance (default None)
        hub, group, project: IBMQ credentials (defaults "ibm-q"/"open"/"main")
        exec_options: additional execution options dict (default None)
        context: context identifier for metrics (default None)
        api: programming API if not already initialized (default None)
    """
    qedc_get_kernel("dj_kernel", api=api)
    import execute as ex
    ex.verbose = verbose

    if context is None:
        context = f"{benchmark_name} Benchmark"

    # Result handler: called for each circuit after execution completes
    def execution_handler(qc, result, num_qubits, circuit_id, num_shots):
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, num_shots,
                type=int(circuit_id))
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
    kernel = qedc_get_kernel("dj_kernel", api=api)

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
    parser = argparse.ArgumentParser(description="Deutsch-Jozsa Benchmark")
    parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits", type=int)
    parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)
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
        num_shots=args.num_shots, backend_id=args.backend_id,
        exec_options={"noise_model": None} if args.nonoise else {},
        api=args.api, max_batch_size=args.max_batch_size,
        draw_circuits=not args.nodraw, plot_results=not args.noplot)
