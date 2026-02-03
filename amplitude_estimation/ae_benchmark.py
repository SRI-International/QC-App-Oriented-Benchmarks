'''
Amplitude Estimation Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import time
import numpy as np

# Add benchmark home dir to path, so the benchmark can be run from anywhere
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# The QED-C initialization module
from _common.qedc_init import qedc_benchmarks_init
from _common import metrics


# Benchmark Name
benchmark_name = "Amplitude Estimation"

np.random.seed(0)

verbose = False


############### Analysis

# Analyze and print measured results
# Expected result is always the secret_int (which encodes alpha), so fidelity calc is simple
def analyze_and_print_result(qc, result, num_qubits, num_shots, s_int=None, num_state_qubits=1):

    num_counting_qubits = num_qubits - num_state_qubits - 1

    counts = result.get_counts(qc)

    # calculate expected output histogram
    a = a_from_s_int(s_int, num_counting_qubits)
    correct_dist = a_to_bitstring(a, num_counting_qubits)

    # generate thermal_dist for polarization calculation
    thermal_dist = metrics.uniform_dist(num_counting_qubits)

    # convert counts, expectation, and thermal_dist to app form for visibility
    app_counts = bitstring_to_a(counts, num_counting_qubits)
    app_correct_dist = {a: 1.0}
    app_thermal_dist = bitstring_to_a(thermal_dist, num_counting_qubits)

    if verbose:
        print(f"For amplitude {a}, expected: {correct_dist} measured: {counts}")
        print(f"   ... For amplitude {a} thermal_dist: {thermal_dist}")
        print(f"For amplitude {a}, app expected: {app_correct_dist} measured: {app_counts}")
        print(f"   ... For amplitude {a} app_thermal_dist: {app_thermal_dist}")

    # use polarization fidelity with rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist, thermal_dist)

    hf_fidelity = metrics.hellinger_fidelity_with_expected(counts, correct_dist)

    if verbose: print(f"  ... fidelity: {fidelity}  hf_fidelity: {hf_fidelity}")

    return counts, fidelity

def a_to_bitstring(a, num_counting_qubits):
    m = num_counting_qubits
    num1 = round(np.arcsin(np.sqrt(a)) / np.pi * 2**m)
    num2 = round( (np.pi - np.arcsin(np.sqrt(a))) / np.pi * 2**m)
    if num1 != num2 and num2 < 2**m and num1 < 2**m:
        counts = {format(num1, "0"+str(m)+"b"): 0.5, format(num2, "0"+str(m)+"b"): 0.5}
    else:
        counts = {format(num1, "0"+str(m)+"b"): 1}
    return counts

def bitstring_to_a(counts, num_counting_qubits):
    est_counts = {}
    m = num_counting_qubits
    precision = int(num_counting_qubits / (np.log2(10))) + 2
    for key in counts.keys():
        r = counts[key]
        num = int(key,2) / (2**m)
        a_est = round((np.sin(np.pi * num) )** 2, precision)
        if a_est not in est_counts.keys():
            est_counts[a_est] = 0
        est_counts[a_est] += r
    return est_counts

def a_from_s_int(s_int, num_counting_qubits):
    theta = s_int * np.pi / (2**num_counting_qubits)
    precision = int(num_counting_qubits / (np.log2(10))) + 2
    a = round(np.sin(theta)**2, precision)
    return a


################ Benchmark Loop

# Because circuit size grows significantly with num_qubits
# limit the max_qubits here ...
MAX_QUBITS=8

# Execute program with default parameters
def run(min_qubits=3, max_qubits=8, skip_qubits=1, max_circuits=3, num_shots=100,
        num_state_qubits=1, # default, not exposed to users
        backend_id=None, provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None, api=None, get_circuits=False,
        draw_circuits=True, plot_results=True):

    # Configure the QED-C Benchmark package for use with the given API
    qedc_benchmarks_init(api, "amplitude_estimation", ["ae_kernel"])
    import ae_kernel as kernel
    import execute as ex

    ##########

    print(f"{benchmark_name} Benchmark Program - {api if api else 'Qiskit'}")

    # Clamp the maximum number of qubits
    if max_qubits > MAX_QUBITS:
        print(f"INFO: {benchmark_name} benchmark is limited to a maximum of {MAX_QUBITS} qubits.")
        max_qubits = MAX_QUBITS

    # validate parameters (smallest circuit is 3 qubits)
    num_state_qubits = max(1, num_state_qubits)
    if max_qubits < num_state_qubits + 2:
        print(f"ERROR: AE Benchmark needs at least {num_state_qubits + 2} qubits to run")
        return
    min_qubits = max(max(3, min_qubits), num_state_qubits + 2)
    skip_qubits = max(1, skip_qubits)

    # create context identifier
    if context is None: context = f"{benchmark_name} Benchmark"

    # special argument handling
    ex.verbose = verbose

    ##########

    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, circuit_id, num_shots):
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, num_shots,
                s_int=int(circuit_id), num_state_qubits=num_state_qubits)
        metrics.store_metric(num_qubits, circuit_id, 'fidelity', fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)

    ##########

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):

        # reset random seed
        np.random.seed(0)

        # as circuit width grows, the number of counting qubits is increased
        num_counting_qubits = num_qubits - num_state_qubits - 1

        # determine number of circuits to execute for this group
        num_circuits = min(2 ** (num_counting_qubits), max_circuits)

        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        if verbose:
            print(f"              with num_state_qubits = {num_state_qubits}  num_counting_qubits = {num_counting_qubits}")

        # determine range of secret strings to loop over
        if 2**(num_counting_qubits) <= max_circuits:
            s_range = list(range(num_circuits))
        else:
            s_range = np.random.choice(2**(num_counting_qubits), num_circuits, False)

        # loop over limited # of secret strings for this
        for s_int in s_range:

            # create circuit_id for use with metrics and execution framework
            circuit_id = s_int

            # create the circuit for given qubit size and secret string, store time metric
            ts = time.time()

            a_ = a_from_s_int(s_int, num_counting_qubits)

            qc = kernel.AmplitudeEstimation(num_state_qubits, num_counting_qubits, a_)
            metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time() - ts)

            # collapse the 3 sub-circuit levels used in this benchmark (for qiskit)
            qc2 = qc.decompose().decompose().decompose()

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc2, num_qubits, circuit_id, num_shots)

        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)

    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    ##########

    if draw_circuits:
        # draw a sample circuit
        kernel.kernel_draw()

    if plot_results:
        # Plot metrics for all circuit sizes
        options = {"shots": num_shots, "reps": max_circuits}
        metrics.plot_metrics(f"Benchmark Results - {benchmark_name} - {api if api is not None else 'Qiskit'}", options=options)


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
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--noplot", "-nop", action="store_true", help="Do not plot results")
    parser.add_argument("--nodraw", "-nod", action="store_true", help="Do not draw circuit diagram")
    return parser.parse_args()

# if main, execute method
if __name__ == '__main__':
    args = get_args()

    # special argument handling
    verbose = args.verbose

    if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits

    # execute benchmark program
    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        num_state_qubits=args.num_state_qubits,
        backend_id=args.backend_id,
        exec_options = {"noise_model" : None} if args.nonoise else {},
        api=args.api,
        draw_circuits=not args.nodraw, plot_results=not args.noplot
        )
