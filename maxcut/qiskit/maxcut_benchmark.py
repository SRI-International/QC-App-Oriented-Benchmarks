"""
MaxCut Benchmark Program - Qiskit
"""

import sys
import time
from collections import namedtuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

sys.path[1:1] = [ "_common", "_common/qiskit" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit" ]
import execute as ex
import metrics as metrics

import common       # from lanl-ansi-max-cut

np.random.seed(0)

verbose = False

# Variable for number of resets to perform after mid circuit measurements
num_resets = 1

# saved circuits for display
QC_ = None
Uf_ = None

# based on examples from https://qiskit.org/textbook/ch-applications/qaoa.html
QAOA_Parameter  = namedtuple('QAOA_Parameter', ['beta', 'gamma'])


############### Circuit Definition
  
# Create ansatz specific to this problem, defined by G = nodes, edges, and the given parameters
def create_qaoa_circ(nqubits, edges, parameters):

    qc = QuantumCircuit(nqubits)

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    for par in parameters:
        # problem unitary
        for i,j in edges:
            qc.rzz(2 * par.gamma, i, j)

        qc.barrier()
        
        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * par.beta, i)

    qc.measure_all()

    return qc
    
# Create the benchmark program 
def MaxCut (num_qubits, secret_int, edges, method = 1):
    
    # Method 1 - execute one instance of ansatz and determine fidelity
    if method == 1:
    
        # create ansatz circuit
        
        ROUNDS = 2
        theta = 2*ROUNDS*[0.7]
        
        # put parameters into the form expected by the ansatz generator
        p = len(theta)//2  # number of qaoa rounds
        beta = theta[:p]
        gamma = theta[p:]
        parameters = [QAOA_Parameter(*t) for t in zip(beta,gamma)]
        
        qc = create_qaoa_circ(num_qubits, edges, parameters)   

    # Method 2 - execute multiple instances of ansatz, primarily to measure execution time average
    # DEVNOTE: unclear if this is the desired approach yet
    elif method == 2:
        pass
    
    # Method 3 - include classical minimizer to control execution and to obtain optimal result
    # Need to build in some sophistication here to control for time/quality profile determination - TBD
    elif method == 3:
        pass
        
    # save small circuit example for display
    global QC_
    if QC_ == None or num_qubits <= 6:
        if num_qubits < 9: QC_ = qc

    # return a handle on the circuit
    return qc

 
############### Result Data Analysis

# DEVNOTE:
# The section below on Result Data Analysis needs work, it is just pulled from HHL
# Need to implement this by pre-calculating the expectation values using noiseless simulator

saved_result = None

# Analyze and print measured results
# Expected result is always the secret_int, so fidelity calc is simple

# NOTE: for the hard-coded matrix A:  [ 1, -1/3, -1/3, 1 ]
#       x - y/3 = 1 - beta
#       -x/3 + y = beta
#  ==
#       x = 9/8 - 3*beta/4
#       y = 3/8 + 3*beta/4
#
#   and beta is stored as secret_int / 10000
#   This allows us to calculate the expected distribution
#
#   NOTE: we are not actually calculating the distribution, since it would have to include the ancilla
#   For now, we just return a distribution of only the 01 and 11 counts
#   Then we compare the ratios obtained with expected ratio to determine fidelity (incorrectly)

def compute_expectation(beta, num_shots):
    x = 9/8 - (3 * beta)/4
    y = 3/8 + (3 * beta)/4
    ratio = x / y
    ratio_sq = ratio * ratio
    #print(f"  ... x,y = {x, y} ratio={ratio} ratio_sq={ratio_sq}")
    
    iy = int(num_shots / (1 + ratio_sq))
    ix = num_shots - iy
    #print(f"    ... ix,iy = {ix, iy}")

    return { '01':ix, '11': iy }

def analyze_and_print_result (qc, result, num_qubits, secret_int, num_shots):
    global saved_result
    saved_result = result
    
    # obtain counts from the result object
    counts = result.get_counts(qc)
    if verbose: print(f"For secret int {secret_int} measured: {counts}")
    
    # compute beta from secret_int, and get expected distribution
    # compute ratio of 01 to 11 measurements for both expected and obtained
    beta = secret_int / 10000
    expected_dist = compute_expectation(beta, num_shots)
    #print(f"... expected = {expected_dist}")
    
    ratio_exp = expected_dist['01'] / expected_dist['11']
    ratio_counts = counts['01'] / counts['11']
    print(f"  ... ratio_exp={ratio_exp}  ratio_counts={ratio_counts}")
    
    # (NOTE: we should use this fidelity calculation, but cannot since we don't know actual expected)
    # use our polarization fidelity rescaling
    ##fidelity = metrics.polarization_fidelity(counts, expected_dist)
    
    # instead, approximate fidelity by comparing ratios
    if ratio_exp > ratio_counts:
        fidelity = ratio_counts / ratio_exp
    else:
        fidelity = ratio_exp / ratio_counts
    if verbose: print(f"  ... fidelity = {fidelity}")
    
    return counts, fidelity


################ Benchmark Loop

# Execute program with default parameters
def run (min_qubits=3, max_qubits=6, max_circuits=3, num_shots=100,
        backend_id='qasm_simulator', method = 1, provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None):

    print("MaxCut Benchmark Program - Qiskit")

    # validate parameters (smallest circuit is 4 qubits)
    max_qubits = max(4, max_qubits)
    min_qubits = min(max(4, min_qubits), max_qubits)
    max_circuits = min(10, max_circuits)
    #print(f"min, max qubits = {min_qubits} {max_qubits}")
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler (qc, result, num_qubits, s_int, num_shots):  
     
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
        metrics.store_metric(num_qubits, s_int, 'fidelity', fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options)

    # for noiseless simulation, set noise model to be None
    # ex.set_noise_model(None)

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    # DEVNOTE: increment by 2 to match the collection of problems in 'instance' folder
    for num_qubits in range(min_qubits, max_qubits + 1, 2):
            
        # determine number of circuits to execute for this group
        #num_circuits = min(2**(num_qubits), max_circuits)
        num_circuits = max_circuits
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
          
        # loop over limited # of inputs for this
        # assume the solution files start with 3 and go up from there
        circuits_done = 0
        for i in range(3, 3 + max_circuits + 1):
            
            # create integer that represents the problem instance; use s_int as circuit id
            s_int = i
            print(f"  ... i={i} s_int={s_int}")
            
            # create filename from num_qubits and circuit_id (s_int), then load the problem file
            instance_filename = f"instance/mc_{str(num_qubits).zfill(3)}_{str(i).zfill(3)}_000.txt"
            #print(f"... instance_filename = {instance_filename}")
            nodes, edges = common.read_maxcut_instance(instance_filename)
            #print(f"nodes = {nodes}")
            #print(f"edges = {edges}")
            
            # if the file does not exist, we are done with this number of qubits
            if nodes == None:
                print(f"  ... problem {str(i).zfill(3)} not found, limiting to {circuits_done} circuit(s).")
                break;
            
            circuits_done += 1
            
            # create the circuit for given qubit size and secret string, store time metric
            ts = time.time()
            qc = MaxCut(num_qubits, s_int, edges, method)
            metrics.store_metric(num_qubits, s_int, 'create_time', time.time()-ts)

            # collapse the sub-circuit levels used in this benchmark (for qiskit)
            qc2 = qc.decompose()

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc2, num_qubits, s_int, shots=num_shots)
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
        
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    #if method == 1: print("\nQuantum Oracle 'Uf' ="); print(Uf_ if Uf_ != None else " ... too large!")

    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - MaxCut ({method}) - Qiskit")

# if main, execute method
if __name__ == '__main__': run()
   
