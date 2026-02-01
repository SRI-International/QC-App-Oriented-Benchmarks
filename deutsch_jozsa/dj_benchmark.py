'''
Deutsch-Jozsa Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

# This benchmark program runs at the top level of the named benchmark directory.
# It uses the "api" parameter to select the API to be used for kernel construction and execution.

import os, sys
import time
import numpy as np

# Add benchmark home dir to path, so the benchmark can be run without pip installing.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# The QED-C initialization module
from _common.qedc_init import qedc_benchmarks_init
from _common import metrics
from _common import qcb_mpi as mpi


# Benchmark Name
benchmark_name = "Deutsch-Jozsa"

np.random.seed(0)

verbose = False

############### Result Data Analysis

# Analyze and print measured results
# Expected result is always the type, so fidelity calc is simple
def analyze_and_print_result (qc, result, num_qubits, type, num_shots):

    # Size of input is one less than available qubits
    input_size = num_qubits - 1

    # obtain counts from the result object
    counts = result.get_counts(qc)
    if verbose: print(f"For type {type} measured: {counts}")
    
    # create the key that is expected to have all the measurements (for this circuit)
    if type == 0: key = '0'*input_size
    else: key = '1'*input_size
    
    # correct distribution is measuring the key 100% of the time
    correct_dist = {key: 1.0}

    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)

    return counts, fidelity

################ Benchmark Loop

# Execute program with default parameters
def run (min_qubits=3, max_qubits=8, skip_qubits=1, max_circuits=3, num_shots=100,
        backend_id=None, provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None, api=None, get_circuits=False):

    # Configure the QED-C Benchmark package for use with the given API
    qedc_benchmarks_init(api, "deutsch_jozsa", ["dj_kernel"])
    import dj_kernel as kernel
    import execute as ex

    mpi.init()
    
    ##########
        
    print(f"{benchmark_name} Benchmark Program - Qiskit")

    # create context identifier
    if context is None: context = f"{benchmark_name} Benchmark"
    
    # special argument handling
    ex.verbose = verbose
    
    # validate parameters (smallest circuit is 3 qubits)
    max_qubits = max(3, max_qubits)
    min_qubits = min(max(3, min_qubits), max_qubits)
    skip_qubits = max(1, skip_qubits)
    #print(f"min, max qubits = {min_qubits} {max_qubits}")
     
    ##########

    # Variable to store all created circuits to return
    if get_circuits:
        all_qcs = {}
        
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler (qc, result, num_qubits, type, num_shots):  
     
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(type), num_shots)
        metrics.store_metric(num_qubits, type, 'fidelity', fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)       
    
    ##########
    
    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):
    
        input_size = num_qubits - 1
        
        # determine number of circuits to execute for this group
        num_circuits = min(2, max_circuits)
        
        if not get_circuits:
            print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        else:
            print(f"************\nCreating [{num_circuits}] circuits with num_qubits = {num_qubits}")
            # Initialize dictionary to store circuits for this qubit group. 
            all_qcs[str(num_qubits)] = {}
        
        # loop over only 2 circuits
        for type in range( num_circuits ):
            
            # create the circuit for given qubit size and secret string, store time metric
            ts = time.time()
            mpi.barrier()
            qc = kernel.DeutschJozsa(num_qubits, type)
            metrics.store_metric(num_qubits, type, 'create_time', time.time()-ts)

            # collapse the sub-circuit levels used in this benchmark (for qiskit)
            qc2 = qc.decompose()

            # Store each circuit if we want to return them
            if get_circuits:
                all_qcs[str(num_qubits)][str(type)] = qc2
                # Continue to skip sumbitting the circuit for execution. 
                continue

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc2, num_qubits, type, num_shots)
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
    
    # Early return if we want the circuits and creation information
    if get_circuits:
        print(f"************\nReturning circuits and circuit information")
        return all_qcs, metrics.circuit_metrics
    
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    ##########
    
    if mpi.leader():
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
    parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)  
    #parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--exec_options", "-e", default=None, help="Additional execution options to be passed to the backend", type=str)
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
        #method=args.method,
        backend_id=args.backend_id,
        exec_options = {"noise_model" : None} if args.nonoise else {},
        #api=args.api
        )
   
