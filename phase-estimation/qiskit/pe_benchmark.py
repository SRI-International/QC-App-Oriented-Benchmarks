'''
Phase Estimation Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import sys
import time

import numpy as np

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit"]
import execute as ex
import metrics as metrics

#from qft_benchmark import inv_qft_gate
from pe_kernel import PhaseEstimation, kernel_draw

# Benchmark Name
benchmark_name = "Phase Estimation"

np.random.seed(0)

verbose = False


############### Analyze Result

# Analyze and print measured results
# Expected result is always theta, so fidelity calc is simple
def analyze_and_print_result(qc, result, num_counting_qubits, theta, num_shots):

    # get results as measured counts
    counts = result.get_counts(qc)  

    # calculate expected output histogram
    correct_dist = theta_to_bitstring(theta, num_counting_qubits)
    
    # generate thermal_dist and ap form of thermal_dist to be comparable to correct_dist
    if num_counting_qubits < 15:
        thermal_dist = metrics.uniform_dist(num_counting_qubits)
        app_thermal_dist = bitstring_to_theta(thermal_dist, num_counting_qubits)
    else :
        thermal_dist = None
        app_thermal_dist = None
        
    # convert counts expectation to app form for visibility
    # app form of correct distribution is measuring theta correctly 100% of the time
    app_counts = bitstring_to_theta(counts, num_counting_qubits)
    app_correct_dist = {theta: 1.0} 
    
    if verbose:
        print(f"For theta {theta}, expected: {correct_dist} measured: {counts}")
        #print(f"   ... For theta {theta} thermal_dist: {thermal_dist}")
        print(f"For theta {theta}, app expected: {app_correct_dist} measured: {app_counts}")
        #print(f"   ... For theta {theta} app_thermal_dist: {app_thermal_dist}")
        
    # use polarization fidelity with rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist, thermal_dist)
    
    # use polarization fidelity with rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist, thermal_dist)
    #fidelity = metrics.polarization_fidelity(app_counts, app_correct_dist, app_thermal_dist)
    
    hf_fidelity = metrics.hellinger_fidelity_with_expected(counts, correct_dist)
    
    if verbose: print(f"  ... fidelity: {fidelity}  hf_fidelity: {hf_fidelity}")
        
    return counts, fidelity

# Convert theta to a bitstring distribution
def theta_to_bitstring(theta, num_counting_qubits):
    counts = {format( int(theta * (2**num_counting_qubits)), "0"+str(num_counting_qubits)+"b"): 1.0}
    return counts

# Convert bitstring to theta representation, useful for debugging
def bitstring_to_theta(counts, num_counting_qubits):
    theta_counts = {}
    for item in counts.items():
        key, r = item
        theta = int(key,2) / (2**num_counting_qubits)
        if theta not in theta_counts.keys():
            theta_counts[theta] = 0
        theta_counts[theta] += r
    return theta_counts


################ Benchmark Loop

# Execute program with default parameters
def run(min_qubits=3, max_qubits=8, skip_qubits=1, max_circuits=3, num_shots=100,
        init_phase=None,
        backend_id=None, provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None):

    print(f"{benchmark_name} Benchmark Program - Qiskit")

    num_state_qubits = 1 # default, not exposed to users, cannot be changed in current implementation

    # validate parameters (smallest circuit is 3 qubits)
    num_state_qubits = max(1, num_state_qubits)
    if max_qubits < num_state_qubits + 2:
        print(f"ERROR: PE Benchmark needs at least {num_state_qubits + 2} qubits to run")
        return
    min_qubits = max(max(3, min_qubits), num_state_qubits + 2)
    skip_qubits = max(1, skip_qubits)
    #print(f"min, max, state = {min_qubits} {max_qubits} {num_state_qubits}")

    # create context identifier
    if context is None: context = f"{benchmark_name} Benchmark"
    
    ##########
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, theta, num_shots):

        # determine fidelity of result set
        num_counting_qubits = int(num_qubits) - 1
        counts, fidelity = analyze_and_print_result(qc, result, num_counting_qubits, float(theta), num_shots)
        metrics.store_metric(num_qubits, theta, 'fidelity', fidelity)

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
        num_counting_qubits = num_qubits - num_state_qubits

        # determine number of circuits to execute for this group
        num_circuits = min(2 ** (num_counting_qubits), max_circuits)
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        
        # determine range of secret strings to loop over
        if 2**(num_counting_qubits) <= max_circuits:
            theta_choices = list(range(num_circuits))
        else:
            theta_choices = np.random.randint(1, 2**(num_counting_qubits), num_circuits + 10)
            theta_choices = list(set(theta_choices))[0:num_circuits]
            
        # scale choices to 1.0
        theta_range = [i/(2**(num_counting_qubits)) for i in theta_choices]

        # loop over limited # of random theta choices
        for theta in theta_range:
            theta = float(theta) 
            
            # if initial phase passed in, use it instead of random values
            if init_phase:
                theta = init_phase
        
            # create the circuit for given qubit size and theta, store time metric
            ts = time.time()
            qc = PhaseEstimation(num_qubits, theta)
            metrics.store_metric(num_qubits, theta, 'create_time', time.time() - ts)
            
            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc, num_qubits, theta, num_shots)

        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)

    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    ##########

    # draw a sample circuit
    kernel_draw()

    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - {benchmark_name} - Qiskit")


#######################
# MAIN

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Bernstei-Vazirani Benchmark")
    #parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    #parser.add_argument("--target", "-t", default=None, help="Target Backend", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits", type=int)
    parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)  
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument("--init_phase", "-p", default=0.0, help="Input Phase Value", type=float)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    return parser.parse_args()
    
# if main, execute method
if __name__ == '__main__': 
    args = get_args()
    
    # configure the QED-C Benchmark package for use with the given API
    # (done here so we can set verbose for now)
    #PhaseEstimation, kernel_draw = qedc_benchmarks_init(args.api)
    
    # special argument handling
    ex.verbose = args.verbose
    verbose = args.verbose
    
    if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits
    
    # execute benchmark program
    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        #method=args.method,
        init_phase=args.init_phase,
        backend_id=args.backend_id,
        exec_options = {"noise_model" : None} if args.nonoise else {},
        #api=args.api
        )
   

