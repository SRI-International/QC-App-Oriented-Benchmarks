'''
Grovers Search Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import time
import numpy as np

from _common import qcb_mpi as mpi
from _common import metrics
from _common.qedc_init import qedc_benchmarks_init
	
# Benchmark Name
benchmark_name = "Grovers Search"

np.random.seed(0)

verbose = False


################ Analysis
  
# Analyze and print measured results
# Expected result is always the secret_int, so fidelity calc is simple
def analyze_and_print_result(qc, result, num_qubits, marked_item, num_shots):
    
    counts = result.get_counts(qc)
    if verbose: print(f"For type {marked_item} measured: {counts}")

    # we compare counts to analytical correct distribution
    correct_dist = grovers_dist(num_qubits, marked_item)
    if verbose: print(f"Marked item: {marked_item}, Correct dist: {correct_dist}")

    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)

    return counts, fidelity

def grovers_dist(num_qubits, marked_item):
    
    n_iterations = int(np.pi * np.sqrt(2 ** num_qubits) / 4)
    
    dist = {}

    for i in range(2**num_qubits):
        key = bin(i)[2:].zfill(num_qubits)
        theta = np.arcsin(1/np.sqrt(2 ** num_qubits))
        
        if i == int(marked_item):
            dist[key] = np.sin((2*n_iterations+1)*theta)**2
        else:
            dist[key] = (np.cos((2*n_iterations+1)*theta)/(np.sqrt(2 ** num_qubits - 1)))**2
    return dist



################ Benchmark Loop

# Because this circuit size grows significantly with num_qubits (due to the mcx gate)
# limit the max_qubits here ...
MAX_QUBITS=8

# Execute program with default parameters
def run(min_qubits=2, max_qubits=6, skip_qubits=1, max_circuits=3, num_shots=100,
        use_mcx_shim=False,
        backend_id=None, provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None, api=None, get_circuits=False):

    # configure the QED-C Benchmark package for use with the given API
    qedc_benchmarks_init(api, "grovers", ["grovers_kernel"])
    import grovers_kernel as kernel
    import execute as ex

    print(f"{benchmark_name} Benchmark Program - Qiskit")

    # Clamp the maximum number of qubits
    if max_qubits > MAX_QUBITS:
        print(f"INFO: {benchmark_name} benchmark is limited to a maximum of {MAX_QUBITS} qubits.")
        max_qubits = MAX_QUBITS
        
    # validate parameters (smallest circuit is 2 qubits)
    max_qubits = max(2, max_qubits)
    min_qubits = min(max(2, min_qubits), max_qubits)
    skip_qubits = max(1, skip_qubits)
    #print(f"min, max qubits = {min_qubits} {max_qubits}")
    
    # create context identifier
    if context is None: context = f"{benchmark_name} Benchmark"
    
    # set the flag to use an mcx shim if given
    if use_mcx_shim:
        print("... using MCX shim")
    
    ##########
    
    # Initialize metrics module
    metrics.init_metrics()

    # Variable to store all created circuits to return
    if get_circuits:
        all_qcs = {}

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, s_int, num_shots):

        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
        metrics.store_metric(num_qubits, s_int, 'fidelity', fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)

    ##########
    
    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):
        
        # determine number of circuits to execute for this group
        num_circuits = min(2 ** (num_qubits), max_circuits)

        if not get_circuits:
            print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        else:
            print(f"************\nCreating [{num_circuits}] circuits with num_qubits = {num_qubits}")
            # Initialize dictionary to store circuits for this qubit group. 
            all_qcs[str(num_qubits)] = {}
        
        # determine range of secret strings to loop over
        if 2**(num_qubits) <= max_circuits:
            s_range = list(range(num_circuits))
        else:
            # create selection larger than needed and remove duplicates (faster than random.choice())
            s_range = np.random.randint(1, 2**(num_qubits), num_circuits + 10)
            s_range = list(set(s_range))[0:max_circuits]
        
        # loop over limited # of secret strings for this
        for s_int in s_range:
            # create the circuit for given qubit size and secret string, store time metric
            ts = time.time()

            n_iterations = int(np.pi * np.sqrt(2 ** num_qubits) / 4)

            qc = kernel.GroversSearch(num_qubits, s_int, n_iterations, use_mcx_shim)
            metrics.store_metric(num_qubits, s_int, 'create_time', time.time() - ts)
            
            # Store each circuit if we want to return them
            if get_circuits:
                all_qcs[str(num_qubits)][str(s_int)] = qc
                # Continue to skip sumbitting the circuit for execution. 
                continue
            
            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc, num_qubits, s_int, num_shots)
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
    
    # Early return if we want the circuits and creation information
    if get_circuits:
        print(f"************\nReturning circuits and circuit information")
        return all_qcs, metrics.circuit_metrics
    
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)
    
    ##########
    
    # draw a sample circuit
    kernel.kernel_draw()

    # Plot metrics for all circuit sizes
    options = {"shots": num_shots, "reps": max_circuits}
    metrics.plot_metrics(f"Benchmark Results - {benchmark_name} - {api if api is not None else 'Qiskit'}", options=options)


#######################
# MAIN

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Bernstei-Vazirani Benchmark")
    parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    #parser.add_argument("--target", "-t", default=None, help="Target Backend", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits", type=int)
    parser.add_argument("--min_qubits", "-min", default=2, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=6, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)  
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    #parser.add_argument("--input_value", "-i", default=None, help="Fixed Input Value", type=int)
    parser.add_argument("--use_mcx_shim", action="store_true", help="Use MCX Shim")
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    return parser.parse_args()
    
# if main, execute method
if __name__ == '__main__': 
    args = get_args()
    
    # configure the QED-C Benchmark package for use with the given API
    # (done here so we can set verbose for now)
    qedc_benchmarks_init(args.api, "grovers", ["grovers_kernel"])
    import execute as ex
    
    # special argument handling
    ex.verbose = args.verbose
    verbose = args.verbose
    
    if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits
    
    # execute benchmark program
    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        #method=args.method,            # not used currently
        use_mcx_shim=args.use_mcx_shim,
        backend_id=args.backend_id,
        exec_options = {"noise_model" : None} if args.nonoise else {},
        api=args.api
        )