'''
Monte Carlo Sampling Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import functools
import time
import numpy as np

# Add benchmark home dir to path, so the benchmark can be run from anywhere
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# The QED-C initialization module (import before adding local _common to path)
from _common.qedc_init import qedc_benchmarks_init
from _common import metrics

# Add local _common to path for mc_utils (after global _common imports)
sys.path.insert(0, str(Path(__file__).parent / "_common"))
import mc_utils


# Benchmark Name
benchmark_name = "Monte Carlo Sampling"

np.random.seed(0)

# default function is f(x) = x^2
f_of_X = functools.partial(mc_utils.power_f, power=2)

# default distribution is gaussian distribution
p_distribution = mc_utils.gaussian_dist

verbose = False


############### Analysis

def analyze_and_print_result(qc, result, num_counting_qubits, mu, num_shots, method, num_state_qubits, c_star):
    """Analyze and print measured results."""

    # generate exact value for the expectation value given our function and dist
    target_dist = p_distribution(num_state_qubits, mu)
    f = functools.partial(f_of_X, num_state_qubits=num_state_qubits)
    if method == 1:
        exact = mc_utils.estimated_value(target_dist, f)
    elif method == 2:
        exact = 0.5  # hard coded exact value from uniform dist and square function

    counts = result.get_counts(qc)

    # calculate the expected output histogram
    correct_dist = a_to_bitstring(exact, num_counting_qubits)

    # generate thermal_dist
    thermal_dist = metrics.uniform_dist(num_counting_qubits)

    # convert counts to app form for visibility
    app_counts = expectation_from_bits(counts, num_counting_qubits, num_shots, method, c_star)
    app_correct_dist = mc_utils.mc_dist(num_counting_qubits, exact, c_star, method)
    app_thermal_dist = expectation_from_bits(thermal_dist, num_counting_qubits, num_shots, method, c_star)

    if verbose:
        print(f"For expected value {exact}, expected: {correct_dist} measured: {counts}")
        print(f"   ... For expected value {exact} thermal_dist: {thermal_dist}")
        print(f"For expected value {exact}, app expected: {app_correct_dist} measured: {app_counts}")
        print(f"   ... For expected value {exact} app_thermal_dist: {app_thermal_dist}")

    # use polarization fidelity with rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist, thermal_dist)

    hf_fidelity = metrics.hellinger_fidelity_with_expected(counts, correct_dist)

    # the max in the counts is what the algorithm would report as the correct answer
    a, _ = mc_utils.value_and_max_prob_from_dist(counts)

    if verbose: print(f"For expected value {exact} measured: {a}")
    if verbose: print(f"Solution counts: {counts}")
    if verbose: print(f"  ... fidelity: {fidelity}  hf_fidelity: {hf_fidelity}")

    return counts, fidelity

def a_to_bitstring(a, num_counting_qubits):
    m = num_counting_qubits
    num1 = round(np.arcsin(np.sqrt(a)) / np.pi * 2**m)
    num2 = round((np.pi - np.arcsin(np.sqrt(a))) / np.pi * 2**m)
    if num1 != num2 and num2 < 2**m and num1 < 2**m:
        counts = {format(num1, "0"+str(m)+"b"): 0.5, format(num2, "0"+str(m)+"b"): 0.5}
    else:
        counts = {format(num1, "0"+str(m)+"b"): 1}
    return counts

def expectation_from_bits(bits, num_qubits, num_shots, method, c_star):
    amplitudes = {}
    for b in bits.keys():
        precision = int(num_qubits / (np.log2(10))) + 2
        r = bits[b]
        a_meas = pow(np.sin(np.pi*int(b, 2)/pow(2, num_qubits)), 2)
        if method == 1:
            a = ((a_meas - 0.5)/c_star) + 0.5
        if method == 2:
            a = a_meas
        a = round(a, precision)
        if a not in amplitudes.keys():
            amplitudes[a] = 0
        amplitudes[a] += r
    return amplitudes


################ Benchmark Loop

MIN_QUBITS = 4
MIN_STATE_QUBITS = 1
MIN_QUBITS_M1 = 5
MIN_STATE_QUBITS_M1 = 2
MAX_QUBITS = 10

def run(min_qubits=MIN_QUBITS, max_qubits=10, skip_qubits=1, max_circuits=1, num_shots=100,
        epsilon=0.05, degree=2, num_state_qubits=MIN_STATE_QUBITS, method=2,
        backend_id=None, provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None, api=None, get_circuits=False):

    # Configure the QED-C Benchmark package for use with the given API
    qedc_benchmarks_init(api, "monte_carlo", ["mc_kernel"])
    import mc_kernel as kernel
    import execute as ex

    ##########

    print(f"{benchmark_name} ({method}) Benchmark Program - {api if api else 'Qiskit'}")

    # Clamp the maximum number of qubits
    if max_qubits > MAX_QUBITS:
        print(f"INFO: {benchmark_name} benchmark is limited to a maximum of {MAX_QUBITS} qubits.")
        max_qubits = MAX_QUBITS

    if method == 2:
        if max_qubits < MIN_QUBITS:
            print(f"INFO: {benchmark_name} benchmark method ({method}) requires a minimum of {MIN_QUBITS} qubits.")
            return
        if min_qubits < MIN_QUBITS:
            min_qubits = MIN_QUBITS
    elif method == 1:
        if max_qubits < MIN_QUBITS_M1:
            print(f"INFO: {benchmark_name} benchmark method ({method}) requires a minimum of {MIN_QUBITS_M1} qubits.")
            return
        if min_qubits < MIN_QUBITS_M1:
            min_qubits = MIN_QUBITS_M1

    if (method == 1) and (num_state_qubits == MIN_STATE_QUBITS):
        num_state_qubits = MIN_STATE_QUBITS_M1

    skip_qubits = max(1, skip_qubits)

    # create context identifier
    if context is None: context = f"{benchmark_name} ({method}) Benchmark"

    # special argument handling
    ex.verbose = verbose

    ##########

    # Initialize metrics module
    metrics.init_metrics()

    c_star = (2*epsilon)**(1/(degree+1))

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, mu, num_shots):
        num_counting_qubits = int(num_qubits) - num_state_qubits - 1
        counts, fidelity = analyze_and_print_result(
            qc, result, num_counting_qubits, float(mu), num_shots,
            method=method, num_state_qubits=num_state_qubits, c_star=c_star
        )
        metrics.store_metric(num_qubits, mu, 'fidelity', fidelity)

    # Initialize execution module
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)

    ##########

    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):

        np.random.seed(0)
        input_size = num_qubits - 1
        num_counting_qubits = num_qubits - num_state_qubits - 1
        num_circuits = min(2 ** (input_size), max_circuits)

        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")

        # determine range of circuits to loop over
        if 2**(input_size) <= max_circuits:
            mu_range = [i/2**(input_size) for i in range(num_circuits)]
        else:
            mu_range = [i/2**(input_size) for i in np.random.choice(2**(input_size), num_circuits, False)]

        for mu in mu_range:
            target_dist = p_distribution(num_state_qubits, mu)
            f_to_estimate = functools.partial(f_of_X, num_state_qubits=num_state_qubits)

            ts = time.time()
            qc = kernel.MonteCarloSampling(target_dist, f_to_estimate, num_state_qubits,
                                           num_counting_qubits, epsilon, degree, method=method)
            metrics.store_metric(num_qubits, mu, 'create_time', time.time() - ts)

            # collapse the sub-circuit levels
            qc2 = qc.decompose().decompose().decompose().decompose()

            ex.submit_circuit(qc2, num_qubits, mu, num_shots)

        ex.throttle_execution(metrics.finalize_group)

    ex.finalize_execution(metrics.finalize_group)

    ##########

    kernel.kernel_draw()

    options = {"method": method, "shots": num_shots, "reps": max_circuits}
    metrics.plot_metrics(f"Benchmark Results - {benchmark_name} ({method}) - {api if api is not None else 'Qiskit'}", options=options)


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
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    verbose = args.verbose

    if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits

    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        method=args.method,
        num_state_qubits=args.num_state_qubits,
        backend_id=args.backend_id,
        exec_options={"noise_model": None} if args.nonoise else {},
        api=args.api
        )
