'''
Monte Carlo Sampling Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.

Three key functions, each independently callable:
  - get_circuits(): Create benchmark circuits (std + app args)
  - run_circuits(): Execute circuits and collect metrics (exec args)
  - plot_results(): Draw circuits and plot metrics (plot args)
  - run(): Convenience that calls all three
'''

import functools
import inspect
import time
import numpy as np

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from qedclib import get_kernel, is_leader
from qedclib import metrics

# Add local _common to path for mc_utils
sys.path.insert(0, str(Path(__file__).parent / "_common"))
import mc_utils

benchmark_name = "Monte Carlo Sampling"

np.random.seed(0)

# default function is f(x) = x^2
f_of_X = functools.partial(mc_utils.power_f, power=2)

# default distribution is gaussian distribution
p_distribution = mc_utils.gaussian_dist

verbose = False

MIN_QUBITS = 4
MIN_STATE_QUBITS = 1
MIN_QUBITS_M1 = 5
MIN_STATE_QUBITS_M1 = 2
MAX_QUBITS = 10


############### Get Circuits

def get_circuits(
    # Standard args (common across benchmarks)
    min_qubits=MIN_QUBITS, max_qubits=10, skip_qubits=1,
    max_circuits=1, method=2,
    # App-specific args
    epsilon=0.05, degree=2, num_state_qubits=MIN_STATE_QUBITS,
    api=None,
):
    """Create Monte Carlo Sampling benchmark circuits.

    Standard args (common to all benchmarks):
        min_qubits: smallest circuit width (default 4)
        max_qubits: largest circuit width (default 10, clamped to 10)
        skip_qubits: increment between widths (default 1)
        max_circuits: max circuits per qubit group (default 1)
        method: 1=gaussian distribution, 2=uniform (default 2)

    App-specific args:
        epsilon: approximation error parameter (default 0.05)
        degree: polynomial degree for f(x) (default 2)
        num_state_qubits: number of state qubits (default 1)
        api: programming API; None = use set_api() value (default None)

    Returns (all_qcs, circuit_metrics) — nested circuit dict and creation metrics.
    """

    # Load the API-specific circuit kernel for this benchmark
    kernel = get_kernel("mc_kernel", api=api)

    if max_qubits > MAX_QUBITS:
        print(f"INFO: {benchmark_name} benchmark is limited to a maximum of {MAX_QUBITS} qubits.")
        max_qubits = MAX_QUBITS

    if method == 2:
        if max_qubits < MIN_QUBITS:
            print(f"INFO: {benchmark_name} benchmark method ({method}) requires a minimum of {MIN_QUBITS} qubits.")
            return {}, {}
        if min_qubits < MIN_QUBITS:
            min_qubits = MIN_QUBITS
    elif method == 1:
        if max_qubits < MIN_QUBITS_M1:
            print(f"INFO: {benchmark_name} benchmark method ({method}) requires a minimum of {MIN_QUBITS_M1} qubits.")
            return {}, {}
        if min_qubits < MIN_QUBITS_M1:
            min_qubits = MIN_QUBITS_M1

    if (method == 1) and (num_state_qubits == MIN_STATE_QUBITS):
        num_state_qubits = MIN_STATE_QUBITS_M1

    skip_qubits = max(1, skip_qubits)

    # Store c_star globally for use by run_circuits result handler
    global _c_star, _num_state_qubits, _method
    _c_star = (2*epsilon)**(1/(degree+1))
    _num_state_qubits = num_state_qubits
    _method = method

    # Build circuits at each qubit width
    all_qcs = {}
    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):

        np.random.seed(0)
        input_size = num_qubits - 1
        num_counting_qubits = num_qubits - num_state_qubits - 1
        num_circuits = min(2 ** (input_size), max_circuits)

        print(f"************\nCreating [{num_circuits}] circuits with num_qubits = {num_qubits}")
        all_qcs[str(num_qubits)] = {}

        # Select random mu values as circuit inputs
        if 2**(input_size) <= max_circuits:
            mu_range = [i/2**(input_size) for i in range(num_circuits)]
        else:
            mu_range = [i/2**(input_size) for i in np.random.choice(2**(input_size), num_circuits, False)]

        # Create each circuit and store in the dict
        for mu in mu_range:
            target_dist = p_distribution(num_state_qubits, mu)
            f_to_estimate = functools.partial(f_of_X, num_state_qubits=num_state_qubits)

            ts = time.time()
            qc = kernel.MonteCarloSampling(target_dist, f_to_estimate, num_state_qubits,
                                           num_counting_qubits, epsilon, degree, method=method)
            metrics.store_metric(num_qubits, mu, 'create_time', time.time() - ts)

            # collapse the sub-circuit levels used in this benchmark (for qiskit)
            qc2 = qc.decompose().decompose().decompose().decompose()

            all_qcs[str(num_qubits)][str(mu)] = qc2

    return all_qcs, metrics.circuit_metrics


############### Result Analysis (used by run_circuits)

def analyze_and_print_result(qc, result, num_counting_qubits, mu, num_shots,
                             method, num_state_qubits, c_star):
    """Compare measured results against expected MC distribution and compute fidelity."""
    target_dist = p_distribution(num_state_qubits, mu)
    f = functools.partial(f_of_X, num_state_qubits=num_state_qubits)
    if method == 1:
        exact = mc_utils.estimated_value(target_dist, f)
    elif method == 2:
        exact = 0.5

    counts = result.get_counts(qc)
    correct_dist = a_to_bitstring(exact, num_counting_qubits)
    thermal_dist = metrics.uniform_dist(num_counting_qubits)

    if verbose:
        app_counts = expectation_from_bits(counts, num_counting_qubits, num_shots, method, c_star)
        app_correct_dist = mc_utils.mc_dist(num_counting_qubits, exact, c_star, method)
        print(f"For expected value {exact}, expected: {correct_dist} measured: {counts}")
        print(f"For expected value {exact}, app expected: {app_correct_dist} measured: {app_counts}")

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

def expectation_from_bits(bits, num_qubits, num_shots, method, c_star):
    amplitudes = {}
    for b, r in bits.items():
        precision = int(num_qubits / (np.log2(10))) + 2
        a_meas = pow(np.sin(np.pi*int(b, 2)/pow(2, num_qubits)), 2)
        if method == 1:
            a = ((a_meas - 0.5)/c_star) + 0.5
        if method == 2:
            a = a_meas
        a = round(a, precision)
        if a not in amplitudes:
            amplitudes[a] = 0
        amplitudes[a] += r
    return amplitudes


############### Run Circuits

def run_circuits(all_qcs,
    num_shots=100, method=2, max_batch_size=None,
    num_state_qubits=MIN_STATE_QUBITS, epsilon=0.05, degree=2,
    backend_id=None, provider_backend=None,
    hub="ibm-q", group="open", project="main",
    exec_options=None, context=None, api=None,
):
    """Execute benchmark circuits and collect metrics.

    Args:
        all_qcs: circuit dict from get_circuits()
        num_shots: measurement shots per circuit (default 100)
        method: algorithm method, for result analysis (default 2)
        max_batch_size: max circuits per batch; None = no limit (default None)
        num_state_qubits: for result analysis (default 1)
        epsilon: for c_star computation (default 0.05)
        degree: for c_star computation (default 2)
        backend_id: backend identifier (default None = qasm_simulator)
        provider_backend: provider backend instance (default None)
        hub, group, project: IBMQ credentials (defaults "ibm-q"/"open"/"main")
        exec_options: additional execution options dict (default None)
        context: context identifier for metrics (default None)
        api: programming API if not already initialized (default None)
    """
    get_kernel("mc_kernel", api=api)
    import execute as ex
    ex.verbose = verbose

    if context is None:
        context = f"{benchmark_name} ({method}) Benchmark"

    c_star = (2*epsilon)**(1/(degree+1))

    # Result handler: called for each circuit after execution completes
    def execution_handler(qc, result, num_qubits, mu, num_shots):
        num_counting_qubits = int(num_qubits) - num_state_qubits - 1
        counts, fidelity = analyze_and_print_result(
            qc, result, num_counting_qubits, float(mu), num_shots,
            method=method, num_state_qubits=num_state_qubits, c_star=c_star)
        metrics.store_metric(num_qubits, mu, 'fidelity', fidelity)

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
    method=2, num_shots=100, max_circuits=1,
    api=None, draw_circuits=True, plot_results=True,
):
    """Draw sample circuit and plot benchmark metrics.

    Args:
        method: algorithm method, for plot title (default 2)
        num_shots: shots, for plot subtitle (default 100)
        max_circuits: circuit reps, for plot subtitle (default 1)
        api: programming API name for plot title (default None)
        draw_circuits: draw a sample circuit diagram (default True)
        plot_results: generate metrics plots (default True)
    """
    kernel = get_kernel("mc_kernel", api=api)

    if is_leader():
        if draw_circuits:
            kernel.kernel_draw()

        if plot_results:
            options = {"method": method, "shots": num_shots, "reps": max_circuits}
            metrics.plot_metrics(
                f"Benchmark Results - {benchmark_name} ({method}) - {api if api is not None else 'Qiskit'}",
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
    parser = argparse.ArgumentParser(description="Monte Carlo Sampling Benchmark")
    parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits", type=int)
    parser.add_argument("--min_qubits", "-min", default=4, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=1, help="Maximum circuit repetitions", type=int)
    parser.add_argument("--method", "-m", default=2, help="Algorithm Method", type=int)
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
        num_shots=args.num_shots, method=args.method,
        num_state_qubits=args.num_state_qubits,
        backend_id=args.backend_id,
        exec_options={"noise_model": None} if args.nonoise else {},
        api=args.api, max_batch_size=args.max_batch_size,
        draw_circuits=not args.nodraw, plot_results=not args.noplot)
