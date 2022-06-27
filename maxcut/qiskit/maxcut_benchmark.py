"""
MaxCut Benchmark Program - Qiskit
"""

import os
import sys
import time
from collections import namedtuple

import math
import numpy as np
from scipy.optimize import minimize

from qiskit import (Aer, ClassicalRegister,  # for computing expectation tables
                    QuantumCircuit, QuantumRegister, execute)
from qiskit.circuit import Parameter

sys.path[1:1] = [ "_common", "_common/qiskit", "maxcut/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit", "../../maxcut/_common/" ]
import common
import execute as ex
import metrics as metrics

np.random.seed(0)

verbose = False

# Indicates whether to perform the (expensive) pre compute of expectations
do_compute_expectation = True

# saved circuits for display
QC_ = None
Uf_ = None

# based on examples from https://qiskit.org/textbook/ch-applications/qaoa.html
QAOA_Parameter  = namedtuple('QAOA_Parameter', ['beta', 'gamma'])


############### Circuit Definition
  
# Create ansatz specific to this problem, defined by G = nodes, edges, and the given parameters
# Do not include the measure operation, so we can pre-compute statevector
def create_qaoa_circ(nqubits, edges, parameters):

    qc = QuantumCircuit(nqubits)

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    for par in parameters:
        #print(f"... gamma, beta = {par.gamma} {par.beta}")
        
        # problem unitary
        for i,j in edges:
            qc.rzz(2 * par.gamma, i, j)

        qc.barrier()
        
        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * par.beta, i)

    return qc
   
# Create the benchmark program circuit
# Accepts optional rounds and array of thetas (betas and gammas)
def MaxCut (num_qubits, secret_int, edges, rounds, thetas_array, parameterized):

    if parameterized:
        return MaxCut_param(num_qubits, secret_int, edges, rounds, thetas_array)

    # if no thetas_array passed in, create defaults 
    if thetas_array is None:
        thetas_array = 2*rounds*[1.0]
    
    #print(f"... incoming thetas_array={thetas_array} rounds={rounds}")
       
    # get number of qaoa rounds (p) from length of incoming array
    p = len(thetas_array)//2 
    
    # if rounds passed in is less than p, truncate array
    if rounds < p:
        p = rounds
        thetas_array = thetas_array[:2*rounds]
    
    # if more rounds requested than in thetas_array, give warning (can fill array later)
    elif rounds > p:
        rounds = p
        print(f"WARNING: rounds is greater than length of thetas_array/2; using rounds={rounds}")
    
    #print(f"... actual thetas_array={thetas_array}")
    
    # create parameters in the form expected by the ansatz generator
    # this is an array of betas followed by array of gammas, each of length = rounds
    betas = thetas_array[:p]
    gammas = thetas_array[p:]
    parameters = [QAOA_Parameter(*t) for t in zip(betas,gammas)]
           
    # and create the circuit, without measurements
    qc = create_qaoa_circ(num_qubits, edges, parameters)   

    # pre-compute and save an array of expected measurements
    if do_compute_expectation:
        compute_expectation(qc, num_qubits, secret_int)
        
    # add the measure here
    qc.measure_all()
        
    # save small circuit example for display
    global QC_
    if QC_ == None or num_qubits <= 6:
        if num_qubits < 9: QC_ = qc

    # return a handle on the circuit
    return qc


############### Circuit Definition - Parameterized version
  
# Create ansatz specific to this problem, defined by G = nodes, edges, and the given parameters
# Do not include the measure operation, so we can pre-compute statevector
def create_qaoa_circ_param(nqubits, edges, parameters):

    qc = QuantumCircuit(nqubits)

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    for par in parameters:
        #print(f"... par={par}  gamma, beta = {par.gamma} {par.beta}")
        
        # problem unitary
        for i,j in edges:
            qc.rzz(2 * par.gamma, i, j)

        qc.barrier()
        
        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * par.beta, i)

    return qc
  
_qc = None
beta_params = []
gamma_params = []
        
# Create the benchmark program circuit
# Accepts optional rounds and array of thetas (betas and gammas)
def MaxCut_param (num_qubits, secret_int, edges, rounds, thetas_array):

    global _qc, beta_params, gamma_params
            
    # if no thetas_array passed in, create defaults 
    if thetas_array is None:
        thetas_array = 2*rounds*[1.0]
    
    #print(f"... incoming thetas_array={thetas_array} rounds={rounds}")
       
    # get number of qaoa rounds (p) from length of incoming array
    p = len(thetas_array)//2 
    
    # if rounds passed in is less than p, truncate array
    if rounds < p:
        p = rounds
        thetas_array = thetas_array[:2*rounds]
    
    # if more rounds requested than in thetas_array, give warning (can fill array later)
    elif rounds > p:
        rounds = p
        print(f"WARNING: rounds is greater than length of thetas_array/2; using rounds={rounds}")
    
    #print(f"... actual thetas_array={thetas_array}")
    
    # create parameters in the form expected by the ansatz generator
    # this is an array of betas followed by array of gammas, each of length = rounds
    betas = thetas_array[:p]
    gammas = thetas_array[p:]
    
    # create the circuit the first time, add measurements
    # first circuit in iterative step is a multiple of 1000
    if secret_int % 1000 == 0 or secret_int < 1000:    # < 1000 is for method 1
    
        # create the named parameter objects used to define the circuit
        beta_params = []
        gamma_params = []
        for i, beta in enumerate(betas):
            beta_params.append(Parameter("beta" + str(i)))
        for j, gamma in enumerate(gammas):
            gamma_params.append(Parameter("gamma" + str(j)))
        #print(f"... param names = {beta_params} {gamma_params}")
        
        parameters = [QAOA_Parameter(*t) for t in zip(beta_params,gamma_params)]
    
        _qc = create_qaoa_circ_param(num_qubits, edges, parameters)
        
        # add the measure here, only after circuit is created
        _qc.measure_all()
        
        #print(f"... created circuit: \n {_qc}")
    
    params = {}
    for i, beta_param in enumerate(beta_params):
        params[beta_param] = thetas_array[i]
    for j, gamma_param in enumerate(gamma_params):
        params[gamma_param] = thetas_array[j + p]
    #print(f"... params and values = {params}")
    
    qc = _qc.bind_parameters(params)
    #print(qc)
    
    # pre-compute and save an array of expected measurements
    ##compute_expectation(qc, num_qubits, secret_int)
   
    # save small circuit example for display
    global QC_
    if QC_ == None or num_qubits <= 6:
        if num_qubits < 9: QC_ = qc

    # return a handle on the circuit
    return qc


############### Expectation Tables

# DEVNOTE: We are building these tables on-demand for now, but for larger circuits
# this will need to be pre-computed ahead of time and stored in a data file to avoid run-time delays.

# dictionary used to store pre-computed expectations, keyed by num_qubits and secret_string
# these are created at the time the circuit is created, then deleted when results are processed
expectations = {}

# Compute array of expectation values in range 0.0 to 1.0
# Use statevector_simulator to obtain exact expectation
def compute_expectation(qc, num_qubits, secret_int, backend_id='statevector_simulator'):
    
    #ts = time.time()
    
    #execute statevector simulation
    sv_backend = Aer.get_backend(backend_id)
    sv_result = execute(qc, sv_backend).result()

    # get the probability distribution
    counts = sv_result.get_counts()

    #print(f"... statevector expectation = {counts}")
    
    # store in table until circuit execution is complete
    id = f"_{num_qubits}_{secret_int}"
    expectations[id] = counts

    #print(f"  ... time to execute statevector simulator: {time.time() - ts}")
    
# Return expected measurement array scaled to number of shots executed
def get_expectation(num_qubits, secret_int, num_shots):

    # find expectation counts for the given circuit 
    id = f"_{num_qubits}_{secret_int}"
    if id in expectations:
        counts = expectations[id]
        
        # scale to number of shots
        for k, v in counts.items():
            counts[k] = round(v * num_shots)
        
        # delete from the dictionary
        del expectations[id]
        
        return counts
        
    else:
        return None
    
    
############### Result Data Analysis

expected_dist = {}

# Compare the measurement results obtained with the expected measurements to determine fidelity
def analyze_and_print_result (qc, result, num_qubits, secret_int, num_shots):
    global expected_dist
    
    # obtain counts from the result object
    counts = result.get_counts(qc)
    
    # retrieve pre-computed expectation values for the circuit that just completed
    expected_dist = get_expectation(num_qubits, secret_int, num_shots)
    
    # if the expectation is not being calculated (only need if we want to compute fidelity)
    # assume that the expectation is the same as measured counts, yielding fidelity = 1
    if expected_dist == None:
        expected_dist = counts
    
    if verbose: print(f"For width {num_qubits} problem {secret_int}\n  measured: {counts}\n  expected: {expected_dist}")

    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, expected_dist)

    if verbose: print(f"For secret int {secret_int} fidelity: {fidelity}")
    
    return counts, fidelity

# Compute the objective function on a given sample
def compute_objective(results, nodes, edges):
    counts = results.get_counts()
    
    avg = 0
    sum_count = 0
    for solution, count in counts.items():
        obj = -1*common.eval_cut(nodes, edges, solution)

        avg += obj * count
        sum_count += count

    return avg/sum_count

# Modified objective function that only considers top N largest counts when 
# calculating the average
def compute_max_objective(results, nodes, edges, N):
    counts = results.get_counts()

    top_n = sorted(counts, key=counts.get, reverse=True)[:N]
    
    avg = 0
    sum_count = 0
    for solution, count in counts.items():
        if solution in top_n:
            obj = -1*common.eval_cut(nodes, edges, solution)

            avg += obj * count
            sum_count += count
        else:
            continue

    return avg/sum_count


# CVaR objective function (Conditional Value at Risk)
def compute_cvar_objective(results, nodes, edges, alpha=0.1):
    """
    Obtains the Confidence Value at Risk or CVaR for samples measured at the end of the variational circuit.
    Reference: Barkoutsos, P. K., Nannicini, G., Robert, A., Tavernelli, I. & Woerner, S. Improving Variational Quantum Optimization using CVaR. Quantum 4, 256 (2020).

    Parameters
    ----------
    results, nodes, edges : self explanatory
    alpha : float, optional
        Confidence interval value for CVaR. The default is 0.1.

    Returns
    -------
    float
        CVaR value

    """

    strings = np.array(list(results.get_counts().keys()))
    counts = np.array(list(results.get_counts().values()))

    # for each measurement outcome |ψ>, obtain <ψ|H|ψ> (i.e. negative of the weight of the cut)
    cut_weights = np.array([-1 * common.eval_cut(nodes, edges, string) for string in strings])

    # Sort cut_weights in a non-decreasing order.
    # Sort counts and strings in the same order as cut_weights, so that i^th element of each correspond to each other
    sort_inds = np.argsort(cut_weights)
    cut_weights = cut_weights[sort_inds]
    strings = strings[sort_inds]
    counts = counts[sort_inds]
    # Cumulative sum of counts
    cumsum_counts = np.cumsum(counts)
    num_shots = cumsum_counts[-1]

    # Restrict to the first int(alpha * num_shots) number of samples in these arrays
    num_averaged = math.ceil(alpha * num_shots)
    final_index = np.digitize(num_averaged, cumsum_counts, right=True)
    counts = counts[:final_index + 1]
    cut_weights = cut_weights[:final_index + 1]
    if final_index == 0:
        counts[0] = min(counts[0], num_averaged)
    elif cumsum_counts[final_index] != num_averaged: # can only be > or =. If >, then need to modify the last entry of counts
        counts[-1] = num_averaged - cumsum_counts[final_index - 1]

    assert num_averaged == int(np.sum(counts)), "number of samples to be averaged is different from cumsum of restricted counts"
    return np.sum(counts * cut_weights) / num_averaged

    # ## Version which avoids using numpy as much as possible
    # # Has a problem that needs to be fixed for certain fringe cases
    # strings = list(results.get_counts().keys())
    # counts = list(results.get_counts().values())

    # # for each measurement outcome |ψ>, obtain <ψ|H|ψ> (i.e. negative of the weight of the cut)
    # cut_weights = [-1 * common.eval_cut(nodes, edges, string) for string in strings]

    # # sort in non-decreasing order
    # sort_inds = sorted(range(len(cut_weights)), key = cut_weights.__getitem__) #same as numpy.argsort
    # cut_weights = [cut_weights[i] for i in sort_inds]
    # strings = [strings[i] for i in sort_inds]
    # counts = [counts[i] for i in sort_inds]

    # cumsum_counts = np.cumsum(counts)
    # num_shots = cumsum_counts[-1]

    # # the samples to be averaged over for cvar are the smallest ceil(alpha*num_shots) number of measured strings
    # num_averaged = math.ceil(alpha * num_shots)
    # # Find the index of the first element of cumsum_counts that is greater than or equal to num_averaged
    # final_index = cumsum_counts.tolist().index(min(x for x in cumsum_counts if x>= num_averaged))

    # counts = counts[:final_index + 1]
    # cut_weights = cut_weights[:final_index + 1]
    # if final_index == 0:
    #     counts[0] = min(counts[0], num_averaged)
    # elif cumsum_counts[final_index] != num_averaged: # can only be > or =. If >, then need to modify the last entry of counts
    #     counts[-1] = num_averaged - cumsum_counts[final_index - 1]

    # assert num_averaged == sum(counts), "number of samples to be averaged is different from cumsum of restricted counts"

    # return sum([i * j for (i,j) in zip(counts, cut_weights)]) / num_averaged



################ Benchmark Loop

# Problem definitions only available for up to 10 qubits currently
MAX_QUBITS = 24
saved_result = None
instance_filename = None

# Execute program with default parameters
def run (min_qubits=3, max_qubits=6, max_circuits=3, num_shots=100,
        method=1, rounds=1, degree=3, thetas_array=None, N=0, alpha=None, parameterized= False, do_fidelities=True,
        max_iter=30, score_metric='fidelity', x_metric='cumulative_exec_time', y_metric='num_qubits',
        fixed_metrics={}, num_x_bins=15, y_size=None, x_size=None,
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None):
    
    global QC_
    global circuits_done
    global unique_circuit_index
    global opt_ts
    
    print("MaxCut Benchmark Program - Qiskit")

    QC_ = None
    
    # validate parameters (smallest circuit is 4 qubits)
    max_qubits = max(4, max_qubits)
    max_qubits = min(MAX_QUBITS, max_qubits)
    min_qubits = min(max(4, min_qubits), max_qubits)
    max_circuits = min(10, max_circuits)
    #print(f"min, max qubits = {min_qubits} {max_qubits}")
    
    # don't compute exectation unless fidelity is is needed
    global do_compute_expectation
    do_compute_expectation = True
    #if method == 2:
    if do_fidelities == False:
        do_compute_expectation = False
    
    rounds = max(1, rounds)
    
    # if more rounds requested than in thetas_array, give warning (DEVNOTE: pad array with 1s)
    if thetas_array != None and rounds > len(thetas_array)/2:
        rounds = len(thetas_array)/2
        print(f"WARNING: rounds is greater than length of thetas_array/2; using rounds={rounds}")
        
    # if no thetas_array passed in, create default array (required for minimizer function)
    if thetas_array == None:
        thetas_array = 2*rounds*[1.0]
    
    # given that this benchmark does every other width, set y_size default to 1.5
    if y_size == None:
        y_size = 1.5
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler (qc, result, num_qubits, s_int, num_shots):  
     
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
        metrics.store_metric(num_qubits, s_int, 'fidelity', fidelity)

    def execution_handler2 (qc, result, num_qubits, s_int, num_shots):
        global saved_result
        global instance_filename
        
        nodes, edges = common.read_maxcut_instance(instance_filename)
        opt, _ = common.read_maxcut_solution(instance_filename[:-4]+'.sol')
        
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
        metrics.store_metric(num_qubits, s_int, 'fidelity', fidelity)
        
        if alpha is not None:
            a_r = -1 * compute_cvar_objective(result, nodes, edges, alpha=alpha) / opt
            metrics.store_metric(num_qubits, s_int, 'cvar_approx_ratio', a_r)
        elif N:
            a_r = -1 * compute_max_objective(result, nodes, edges, N) / opt
            metrics.store_metric(num_qubits, s_int, 'Max_N_approx_ratio', a_r)
        else:
            a_r = -1 * compute_objective(result, nodes, edges) / opt
            metrics.store_metric(num_qubits, s_int, 'approx_ratio', a_r)

        
        saved_result = result
     
    # Initialize execution module using the execution result handler above and specified backend_id
    # for method=2 we need to set max_jobs_active to 1, so each circuit completes before continuing
    if method == 2:
        ex.max_jobs_active = 1
        ex.init_execution(execution_handler2)
    else:
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
      
        # how many of the num_circuits have we completed
        circuits_complete = 0
        
        # loop over each of num_circuits
        # assume the solution files start with 3 and go up from there
        if degree > 0: 
            degree_range = range(degree, degree + num_circuits) 
        else:
            _start = max(3, (num_qubits + degree - max_circuits))
            degree_range = range(_start, _start + max_circuits)

        for i in degree_range:
        
            # create integer that represents the problem instance; use s_int as circuit id
            s_int = i
            #print(f"  ... i={i} s_int={s_int}")
        
            # create filename from num_qubits and circuit_id (s_int), then load the problem file
            global instance_filename
            instance_filename = os.path.join(os.path.dirname(__file__),
                "..", "_common", common.INSTANCE_DIR, f"mc_{num_qubits:03d}_{i:03d}_000.txt"
            )
            # print(f"... instance_filename = {instance_filename}")
            nodes, edges = common.read_maxcut_instance(instance_filename)
            #print(f"nodes = {nodes}")
            #print(f"edges = {edges}")
        
            # if the file does not exist, we are done with this number of qubits
            if nodes == None:
                print(f"  ... problem {i:03d} not found, limiting to {circuits_complete} circuit(s).")
                break;
        
            circuits_complete += 1
        
            if method != 2:
        
                # create the circuit for given qubit size and secret string, store time metric
                ts = time.time()
                qc = MaxCut(num_qubits, s_int, edges, rounds, thetas_array, parameterized)
                metrics.store_metric(num_qubits, s_int, 'create_time', time.time()-ts)

                # collapse the sub-circuit levels used in this benchmark (for qiskit)
                qc2 = qc.decompose()

                # submit circuit for execution on target (simulator, cloud simulator, or hardware)
                ex.submit_circuit(qc2, num_qubits, s_int, shots=num_shots)
                 
            if method == 2:
                      
                # a unique circuit index used inside the inner minimizer loop as identifier         
                unique_circuit_index = 0 
                start_iters_t = time.time()
                
                def expectation(thetas_array):
                    global unique_circuit_index
                    global opt_ts
                    
                    # Every circuit needs a unique id; add unique_circuit_index instead of s_int
                    unique_id = s_int*1000 + unique_circuit_index
                    
                    # store the optimizer execution time of last cycle
                    # NOTE: the first time it is stored it is just the initialization time for optimizer
                    metrics.store_metric(num_qubits, unique_id, 'opt_exec_time', time.time()-opt_ts)
                    
                    unique_circuit_index += 1
                
                    # create the circuit for given qubit size and secret string, store time metric
                    ts = time.time()
                    qc = MaxCut(num_qubits, unique_id, edges, rounds, thetas_array, parameterized)
                    metrics.store_metric(num_qubits, unique_id, 'create_time', time.time()-ts)
                    
                    # also store the 'rounds' for each execution
                    metrics.store_metric(num_qubits, unique_id, 'rounds', rounds)

                    # collapse the sub-circuit levels used in this benchmark (for qiskit)
                    qc2 = qc.decompose()

                    # submit circuit for execution on target (simulator, cloud simulator, or hardware)
                    ex.submit_circuit(qc2, num_qubits, unique_id, shots=num_shots)
                
                    # Must wait for circuit to complete
                    #ex.throttle_execution(metrics.finalize_group)
                    ex.finalize_execution(None, report_end=False)    # don't finalize group until all circuits done
                
                    # reset timer for optimizer execution after each iteration of quantum program completes
                    opt_ts = time.time()
                    
                    if alpha is not None:
                        return compute_cvar_objective(saved_result, nodes, edges, alpha)
                    elif N:
                        return compute_max_objective(saved_result, nodes, edges, N)
                    else:
                        return compute_objective(saved_result, nodes, edges)
            
                opt_ts = time.time()
                
                # perform the complete algorithm; minimizer invokes 'expectation' function iteratively
                res = minimize(expectation, thetas_array, method='COBYLA', options = { 'maxiter': max_iter} )
                
                unique_circuit_index = 0
                unique_id = s_int*1000 + unique_circuit_index
                metrics.store_metric(num_qubits, unique_id, 'opt_exec_time', time.time()-opt_ts)
                
                #read solution from file for this instance
                opt, sol = common.read_maxcut_solution(instance_filename)
            
                num_qubits = int(num_qubits)
                #counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
                fidelity = -1 * res.fun / opt #known optimum
            
                '''
                metrics.store_metric(num_qubits, s_int, 'fidelity', fidelity)
                metrics.store_metric(num_qubits, s_int, 'rounds', p_depth)
                '''
                #print(res)
            
        # for method 2, need to aggregate the detail metrics appropriately for each group
        # Note that this assumes that all iterations of the circuit have completed by this point
        if method == 2:                  
            metrics.process_circuit_metrics_2_level(num_qubits)
            metrics.finalize_group(str(num_qubits))
            
    # Wait for some active circuits to complete; report metrics when groups complete
    ex.throttle_execution(metrics.finalize_group)
        
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)
             
    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    #if method == 1: print("\nQuantum Oracle 'Uf' ="); print(Uf_ if Uf_ != None else " ... too large!")

    # Plot metrics for all circuit sizes
    if method == 1:
        metrics.plot_metrics(f"Benchmark Results - MaxCut ({method}) - Qiskit",
                options=dict(shots=num_shots))
    elif method == 2:
        #metrics.print_all_circuit_metrics()
        
        # Generate area plot showing iterative evolution of metrics 
        metrics.plot_all_area_metrics(f"Benchmark Results - MaxCut ({method}) - Qiskit",
                score_metric=score_metric, x_metric=x_metric, y_metric=y_metric, fixed_metrics=fixed_metrics,
                num_x_bins=num_x_bins, x_size=x_size, y_size=y_size,
                options=dict(shots=num_shots, rounds=rounds, degree=degree))
                
        # Generate bar chart showing optimality gaps in final results
        metrics.plot_metrics_optgaps(f"Benchmark Results - MaxCut ({method}) - Qiskit",
                options=dict(shots=num_shots, rounds=rounds, degree=degree),
                            suffix=f'-s{num_shots}_r{rounds}_d{rounds}')

# if main, execute method
if __name__ == '__main__': run()
