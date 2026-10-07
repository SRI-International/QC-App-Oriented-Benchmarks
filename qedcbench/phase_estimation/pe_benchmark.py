'''
Phase Estimation Benchmark Program
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

import qedclib
from qedclib import get_kernel, is_leader, metrics

benchmark_name = "Phase Estimation"

np.random.seed(0)
verbose = False


############### Get Circuits

def get_circuits(
    # Standard args (common across benchmarks)
    min_qubits=3, max_qubits=8, skip_qubits=1,
    max_circuits=3,
    # App-specific args
    use_midcircuit_measurement=False, init_phase=None,
    api=None,
):
    """Create Phase Estimation benchmark circuits.

    Standard args (common to all benchmarks):
        min_qubits: smallest circuit width (default 3)
        max_qubits: largest circuit width (default 8)
        skip_qubits: increment between widths (default 1)
        max_circuits: max circuits per qubit group (default 3)

    App-specific args:
        use_midcircuit_measurement: use dynamic circuits for inverse QFT (default False)
        init_phase: fixed phase value for all circuits; None = random (default None)
        api: programming API; None = use set_api() value (default None)

    Returns (all_qcs, circuit_metrics) — nested circuit dict and creation metrics.
    """

    # Load the API-specific circuit kernel for this benchmark (e.g. qiskit or cudaq)
    kernel = get_kernel("pe_kernel", api=api)

    num_state_qubits = 1  # fixed, not exposed to users

    # validate parameters
    if max_qubits < num_state_qubits + 2:
        print(f"ERROR: PE Benchmark needs at least {num_state_qubits + 2} qubits to run")
        return {}, {}
    min_qubits = max(max(3, min_qubits), num_state_qubits + 2)
    skip_qubits = max(1, skip_qubits)

    # Build circuits at each qubit width, with max_circuits random phase values per width
    all_qcs = {}
    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):

        np.random.seed(0)

        # as circuit width grows, the number of counting qubits increases
        num_counting_qubits = num_qubits - num_state_qubits
        num_circuits = min(2 ** (num_counting_qubits), max_circuits)

        print(f"************\nCreating [{num_circuits}] circuits with num_qubits = {num_qubits}")
        all_qcs[str(num_qubits)] = {}

        # Select random theta values scaled to [0, 1)
        if 2**(num_counting_qubits) <= max_circuits:
            theta_choices = list(range(num_circuits))
        else:
            theta_choices = np.random.randint(1, 2**(num_counting_qubits), num_circuits + 10)
            theta_choices = list(set(theta_choices))[0:num_circuits]

        theta_range = [i/(2**(num_counting_qubits)) for i in theta_choices]

        # Create each circuit with a different phase value and store in the dict
        for theta in theta_range:
            theta = float(theta)
            if init_phase:
                theta = init_phase

            circuit_id = theta

            ts = time.time()
            qc = kernel.PhaseEstimation(num_qubits, theta, use_midcircuit_measurement)
            metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time() - ts)

            all_qcs[str(num_qubits)][str(circuit_id)] = qc

    return all_qcs, metrics.circuit_metrics


############### Result Analysis (used by run_circuits)

def analyze_and_print_result(qc, result, num_qubits, num_shots, theta=None):
    """Compare measured results against expected phase distribution and compute fidelity."""
    num_counting_qubits = num_qubits - 1
    counts = result.get_counts(qc)

    correct_dist = theta_to_bitstring(theta, num_counting_qubits)

    if num_counting_qubits < 15:
        thermal_dist = metrics.uniform_dist(num_counting_qubits)
    else:
        thermal_dist = None

    if verbose:
        app_counts = bitstring_to_theta(counts, num_counting_qubits)
        app_correct_dist = {theta: 1.0}
        print(f"For theta {theta}, expected: {correct_dist} measured: {counts}")
        print(f"For theta {theta}, app expected: {app_correct_dist} measured: {app_counts}")

    fidelity = metrics.polarization_fidelity(counts, correct_dist, thermal_dist)
    if verbose: print(f"  ... fidelity: {fidelity}")
    return counts, fidelity

def theta_to_bitstring(theta, num_counting_qubits):
    """Convert theta to expected bitstring distribution."""
    return {format(int(theta * (2**num_counting_qubits)), "0"+str(num_counting_qubits)+"b"): 1.0}

def bitstring_to_theta(counts, num_counting_qubits):
    """Convert bitstring counts to theta representation (for debugging)."""
    theta_counts = {}
    for key, r in counts.items():
        theta = int(key, 2) / (2**num_counting_qubits)
        if theta not in theta_counts:
            theta_counts[theta] = 0
        theta_counts[theta] += r
    return theta_counts


############### Run Circuits

def run_circuits(all_qcs,
    num_shots=100, method=None, max_batch_size=None,
    backend_id=None, provider_backend=None,
    hub="ibm-q", group="open", project="main",
    exec_options=None, context=None, api=None,
    parallel=False,
):
    """Execute benchmark circuits and collect metrics.

    Args:
        all_qcs: circuit dict from get_circuits()
        num_shots: measurement shots per circuit (default 100)
        method: algorithm method, for plot options (default None)
        max_batch_size: max circuits per batch; None = no limit (default None)
        backend_id: backend identifier (default None = qasm_simulator)
        provider_backend: provider backend instance (default None)
        hub, group, project: IBMQ credentials (defaults "ibm-q"/"open"/"main")
        exec_options: additional execution options dict (default None)
        context: context identifier for metrics (default None)
        api: programming API if not already initialized (default None)
        parallel: enable parallel circuit execution (default False)
    """
    get_kernel("pe_kernel", api=api)
    ex = qedclib.execute
    ex.verbose = verbose

    if context is None:
        context = f"{benchmark_name} Benchmark"

    # Result handler: called for each circuit after execution completes
    def execution_handler(qc, result, num_qubits, circuit_id, num_shots):
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, num_shots,
                theta=float(circuit_id))
        metrics.store_metric(num_qubits, circuit_id, 'fidelity', fidelity)

    # Set up execution target and submit all circuits as a batch
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)

    ex.compute_all_circuit_metrics(all_qcs)
    ex.parallel_execution = parallel
    ex.submit_circuits(all_qcs, num_shots=num_shots, max_batch_size=max_batch_size)
    metrics.finalize_all_groups()

    metrics.save_app_metrics(benchmark_name)


############### Plot Results

def plot_results(
    method=None, num_shots=100, max_circuits=3,
    api=None, draw_circuits=True, plot_results=True,
):
    """Draw sample circuit and plot benchmark metrics.

    Args:
        method: algorithm method, for plot options (default None)
        num_shots: shots, for plot subtitle (default 100)
        max_circuits: circuit reps, for plot subtitle (default 3)
        api: programming API name for plot title (default None)
        draw_circuits: draw a sample circuit diagram (default True)
        plot_results: generate metrics plots (default True)
    """
    kernel = get_kernel("pe_kernel", api=api)

    if is_leader():
        if draw_circuits:
            kernel.kernel_draw()

        if plot_results:
            options = {"method": method, "shots": num_shots, "reps": max_circuits}
            metrics.plot_metrics(
                f"Benchmark Results - {benchmark_name} - {api if api is not None else 'Qiskit'}",
                options=options)


############### Run (convenience)

def run(**kwargs):
    """Create circuits, execute, and plot. Accepts any arg from
    get_circuits(), run_circuits(), or plot_results()."""

    # If max_batch_size set, use batched create-execute loop to limit memory
    if kwargs.get('max_batch_size') is not None:
        from qedclib.batched import batched_run
        return batched_run(get_circuits, run_circuits, plot_results, **kwargs)

    # Partition incoming arguments to the function that accepts them
    def _for(func):
        return {k: kwargs[k] for k in kwargs if k in inspect.signature(func).parameters}

    # Step 1: Create the benchmark circuits
    metrics.init_metrics()
    all_qcs, circuit_metrics = get_circuits(**_for(get_circuits))

    # Step 2: Execute circuits on the target backend
    run_circuits(all_qcs, **_for(run_circuits))
    metrics.end_metrics()

    # Step 3: Draw sample circuit and plot metrics
    plot_results(**_for(plot_results))


#######################
# MAIN

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Phase Estimation Benchmark")
    parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    parser.add_argument("--target", "-t", default=None, help="Target Backend", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits", type=int)
    parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument("--init_phase", "-p", default=0.0, help="Input Phase Value", type=float)
    parser.add_argument("--max_batch_size", "-mbs", default=None, help="Max circuits per execution batch", type=int)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--use_midcircuit_measurement", "-mid", action="store_true", help="Use dynamic circuit")
    parser.add_argument("--noplot", "-nop", action="store_true", help="Do not plot results")
    parser.add_argument("--nodraw", "-nod", action="store_true", help="Do not draw circuit diagram")
    parser.add_argument("--parallel", "-pm", action="store_true", help="Enable parallel circuit execution")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    verbose = args.verbose
    if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits

    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots, method=args.method,
        use_midcircuit_measurement=args.use_midcircuit_measurement,
        init_phase=args.init_phase, backend_id=args.backend_id,
        exec_options={"noise_model": None} if args.nonoise else None,
        api=args.api, max_batch_size=args.max_batch_size,
        draw_circuits=not args.nodraw, plot_results=not args.noplot,
        parallel=args.parallel)
