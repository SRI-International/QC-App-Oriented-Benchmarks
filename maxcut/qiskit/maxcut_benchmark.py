"""
MaxCut Benchmark Program - Qiskit
"""

import datetime
import json
import logging
import math
import os
import re
import sys
import time
from collections import namedtuple

import numpy as np
from scipy.optimize import minimize

from qiskit import (Aer, ClassicalRegister,  # for computing expectation tables
                    QuantumCircuit, QuantumRegister, execute, transpile)
from qiskit.circuit import ParameterVector

sys.path[1:1] = [ "_common", "_common/qiskit", "maxcut/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit", "../../maxcut/_common/" ]
import common
import execute as ex
import metrics as metrics

# DEVNOTE: this logging feature should be moved to common level
logger = logging.getLogger(__name__)
fname, _, ext = os.path.basename(__file__).partition(".")
log_to_file = False

try:
    if log_to_file:
        logging.basicConfig(
            # filename=f"{fname}_{datetime.datetime.now().strftime('%Y_%m_%d_%S')}.log",
            filename=f"{fname}.log",
            filemode='w',
            encoding='utf-8',
            level=logging.INFO,
            format='%(asctime)s %(name)s - %(levelname)s:%(message)s'
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s %(name)s - %(levelname)s:%(message)s')
        
except Exception as e:
    print(f'Exception {e} occured while configuring logger: bypassing logger config to prevent data loss')
    pass

# Benchmark Name
benchmark_name = "MaxCut"

np.random.seed(0)

maxcut_inputs = dict() #inputs to the run method
verbose = False
print_sample_circuit = True
# Indicates whether to perform the (expensive) pre compute of expectations
do_compute_expectation = True

# saved circuits for display
QC_ = None
Uf_ = None

# based on examples from https://qiskit.org/textbook/ch-applications/qaoa.html
QAOA_Parameter  = namedtuple('QAOA_Parameter', ['beta', 'gamma'])

# Qiskit uses the little-Endian convention. Hence, measured bit-strings need to be reversed while evaluating cut sizes
reverseStep = -1

#%% MaxCut circuit creation and fidelity analaysis functions
def create_qaoa_circ(nqubits, edges, parameters):

    qc = QuantumCircuit(nqubits)

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    for par in parameters:
        #print(f"... gamma, beta = {par.gamma} {par.beta}")
        
        # problem unitary
        for i,j in edges:
            qc.rzz(- par.gamma, i, j)

        qc.barrier()
        
        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * par.beta, i)

    return qc
   

def MaxCut (num_qubits, secret_int, edges, rounds, thetas_array, parameterized, measured = True):

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
    
    logger.info(f'*** Constructing NON-parameterized circuit for {num_qubits = } {secret_int}')
    
    # create parameters in the form expected by the ansatz generator
    # this is an array of betas followed by array of gammas, each of length = rounds
    betas = thetas_array[:p]
    gammas = thetas_array[p:]
    parameters = [QAOA_Parameter(*t) for t in zip(betas,gammas)]
           
    # and create the circuit, without measurements
    qc = create_qaoa_circ(num_qubits, edges, parameters)   

    # pre-compute and save an array of expected measurements
    if do_compute_expectation:
        logger.info('Computing expectation')
        compute_expectation(qc, num_qubits, secret_int)
        
    # add the measure here
    if measured: qc.measure_all()
        
    # save small circuit example for display
    global QC_
    if QC_ == None or num_qubits <= 6:
        if num_qubits < 9: QC_ = qc

    # return a handle on the circuit
    return qc, None


############### Circuit Definition - Parameterized version
  
# Create ansatz specific to this problem, defined by G = nodes, edges, and the given parameters
# Do not include the measure operation, so we can pre-compute statevector
def create_qaoa_circ_param(nqubits, edges, betas, gammas):

    qc = QuantumCircuit(nqubits)

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    for beta, gamma in zip(betas, gammas):
        #print(f"... gamma, beta = {gammas}, {betas}")
        
        # problem unitary
        for i,j in edges:
            qc.rzz(- gamma, i, j)

        qc.barrier()
        
        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta, i)

    return qc
  
_qc = None
beta_params = []
gamma_params = []
        
# Create the benchmark program circuit
# Accepts optional rounds and array of thetas (betas and gammas)
def MaxCut_param (num_qubits, secret_int, edges, rounds, thetas_array):
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
    global _qc
    global betas
    global gammas
    
    # create the circuit the first time, add measurements
    if ex.do_transpile_for_execute:
        logger.info(f'*** Constructing parameterized circuit for {num_qubits = } {secret_int}')
        betas = ParameterVector("𝞫", p)
        gammas = ParameterVector("𝞬", p)
    
        _qc = create_qaoa_circ_param(num_qubits, edges, betas, gammas)
        
        # add the measure here, only after circuit is created
        _qc.measure_all()
    
    params = {betas: thetas_array[:p], gammas: thetas_array[p:]}   
    #logger.info(f"Binding parameters {params = }")
    logger.info(f"Create binding parameters for {thetas_array}")
    
    qc = _qc
    #print(qc)
    
    # pre-compute and save an array of expected measurements
    if do_compute_expectation:
        logger.info('Computing expectation')
        compute_expectation(qc, num_qubits, secret_int, params=params)
   
    # save small circuit example for display
    global QC_
    if QC_ == None or num_qubits <= 6:
        if num_qubits < 9: QC_ = qc

    # return a handle on the circuit
    return qc, params


############### Expectation Tables

# DEVNOTE: We are building these tables on-demand for now, but for larger circuits
# this will need to be pre-computed ahead of time and stored in a data file to avoid run-time delays.

# dictionary used to store pre-computed expectations, keyed by num_qubits and secret_string
# these are created at the time the circuit is created, then deleted when results are processed
expectations = {}

# Compute array of expectation values in range 0.0 to 1.0
# Use statevector_simulator to obtain exact expectation
def compute_expectation(qc, num_qubits, secret_int, backend_id='statevector_simulator', params=None):
    
    #ts = time.time()
    if params != None:
        qc = qc.bind_parameters(params)
    
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
def get_expectation(num_qubits, degree, num_shots):

    # find expectation counts for the given circuit 
    id = f"_{num_qubits}_{degree}"
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


#%% Computation of various metrics, such as approximation ratio, etc.
def compute_cutsizes(results, nodes, edges):
    """
    Given a result object, extract the values of meaasured cuts and the corresponding 
    counts into ndarrays. Also compute and return the corresponding cut sizes.

    Returns
    -------
    cuts : list of strings
        each element is a bitstring denoting a cut
    counts : ndarray of ints
        measured counts corresponding to cuts
    sizes : ndarray of ints
        cut sizes (i.e. number of edges crossing the cut)
    """
    cuts = list(results.get_counts().keys())
    counts = list(results.get_counts().values())
    sizes = [common.eval_cut(nodes, edges, cut, reverseStep) for cut in cuts]
    return cuts, counts, sizes

def get_size_dist(counts, sizes):
    """ For given measurement outcomes, i.e. combinations of cuts, counts and sizes, return counts corresponding to each cut size.
    """
    unique_sizes = list(set(sizes))
    unique_counts = [0] * len(unique_sizes)
    
    for i, size in enumerate(unique_sizes):
        corresp_counts = [counts[ind] for ind,s in enumerate(sizes) if s == size]
        unique_counts[i] = sum(corresp_counts)
    
    # Make sure that the scores are in ascending order
    s_and_c_list = [[a,b] for (a,b) in zip(unique_sizes, unique_counts)]
    s_and_c_list = sorted(s_and_c_list, key = lambda x : x[0])
    unique_sizes = [x[0] for x in s_and_c_list]
    unique_counts = [x[1] for x in s_and_c_list]
    cumul_counts = np.cumsum(unique_counts)
    return unique_counts, unique_sizes, cumul_counts.tolist()


# Compute the objective function on a given sample
def compute_sample_mean(counts, sizes, **kwargs):
    """
    Compute the mean of cut sizes (i.e. the weighted average of sizes weighted by counts)
    This approximates the expectation value of the state at the end of the circuit

    Parameters
    ----------
    counts : ndarray of ints
        measured counts corresponding to cuts
    sizes : ndarray of ints
        cut sizes (i.e. number of edges crossing the cut)
    **kwargs : optional arguments
        will be ignored

    Returns
    -------
    float
        

    """
    # Convert counts and sizes to ndarrays, if they are lists
    counts, sizes = np.array(counts), np.array(sizes)

    return - np.sum(counts * sizes) / np.sum(counts)

def compute_cvar(counts, sizes, alpha = 0.1, **kwargs):
    """
    Obtains the Conditional Value at Risk or CVaR for samples measured at the end of the variational circuit.
    Reference: Barkoutsos, P. K., Nannicini, G., Robert, A., Tavernelli, I. & Woerner, S. Improving Variational Quantum Optimization using CVaR. Quantum 4, 256 (2020).

    Parameters
    ----------
    counts : ndarray of ints
        measured counts corresponding to cuts
    sizes : ndarray of ints
        cut sizes (i.e. number of edges crossing the cut)
    alpha : float, optional
        Confidence interval value for CVaR. The default is 0.1.

    Returns
    -------
    float
        CVaR value

    """
    # Convert counts and sizes to ndarrays, if they are lists
    counts, sizes = np.array(counts), np.array(sizes)
    
    # Sort the negative of the cut sizes in a non-decreasing order.
    # Sort counts in the same order as sizes, so that i^th element of each correspond to each other
    sort_inds = np.argsort(-sizes)
    sizes = sizes[sort_inds]
    counts = counts[sort_inds]

    # Choose only the top num_avgd = ceil(alpha * num_shots) cuts. These will be averaged over.
    num_avgd = math.ceil(alpha * np.sum(counts))

    # Compute cvar
    cvar_sum = 0
    counts_so_far = 0
    for c, s in zip(counts, sizes):
        if counts_so_far + c >= num_avgd:
            cts_to_consider = num_avgd - counts_so_far
            cvar_sum += cts_to_consider * s
            break
        else:
            counts_so_far += c
            cvar_sum += c * s

    return - cvar_sum / num_avgd

def compute_gibbs(counts, sizes, eta = 0.5, **kwargs):
    """
    Compute the Gibbs objective function for given measurements

    Parameters
    ----------
    counts : ndarray of ints
        measured counts corresponding to cuts
    sizes : ndarray of ints
        cut sizes (i.e. number of edges crossing the cut)
    eta : float, optional
        Inverse Temperature
    Returns
    -------
    float
        - Gibbs objective function value / optimal value

    """
    # Convert counts and sizes to ndarrays, if they are lists
    counts, sizes = np.array(counts), np.array(sizes)
    ls = max(sizes)#largest size
    shifted_sizes = sizes - ls
    
    # gibbs = - np.log( np.sum(counts * np.exp(eta * sizes)) / np.sum(counts))
    gibbs = - eta * ls - np.log(np.sum (counts / np.sum(counts) * np.exp(eta * shifted_sizes)))
    return gibbs


def compute_best_cut_from_measured(counts, sizes, **kwargs):
    """From the measured cuts, return the size of the largest cut
    """
    return - np.max(sizes)


def compute_quartiles(counts, sizes):
    """
    Compute and return the sizes of the cuts at the three quartile values (i.e. 0.25, 0.5 and 0.75)

    Parameters
    ----------
    counts : ndarray of ints
        measured counts corresponding to cuts
    sizes : ndarray of ints
        cut sizes (i.e. number of edges crossing the cut)

    Returns
    -------
    quantile_sizes : ndarray of of 3 floats
        sizes of cuts corresponding to the three quartile values.
    """
    # Convert counts and sizes to ndarrays, if they are lists
    counts, sizes = np.array(counts), np.array(sizes)

    # Sort sizes and counts in the sequence of non-decreasing values of sizes
    sort_inds = np.argsort(sizes)
    sizes = sizes[sort_inds]
    counts = counts[sort_inds]
    num_shots = np.sum(counts)

    q_vals = [0.25, 0.5, 0.75]
    ct_vals = [math.floor(q * num_shots) for q in q_vals]

    cumsum_counts = np.cumsum(counts)
    locs = np.searchsorted(cumsum_counts, ct_vals)
    quantile_sizes = sizes[locs]
    return quantile_sizes

def uniform_cut_sampling(num_qubits, degree, num_shots, _instances=None):
    """
    For a given problem, i.e. num_qubits and degree values, sample cuts uniformly
    at random from all possible cuts, num_shots number of times. Return the corresponding
    cuts, counts and cut sizes.
    """
    
    # First, load the nodes and edges corresponding to the problem 
    instance_filename = os.path.join(os.path.dirname(__file__),
                                     "..", "_common", common.INSTANCE_DIR, 
                                     f"mc_{num_qubits:03d}_{degree:03d}_000.txt")
    nodes, edges = common.read_maxcut_instance(instance_filename, _instances)
    
    # Obtain num_shots number of uniform random samples between 0 and 2 ** num_qubits
    unif_cuts = np.random.randint(2 ** num_qubits, size=num_shots).tolist()
    unif_cuts_uniq = list(set(unif_cuts))

    # Get counts corresponding to each sampled int/cut
    unif_counts = [unif_cuts.count(cut) for cut in unif_cuts_uniq]
    unif_cuts = list(set(unif_cuts))

    def int_to_bs(numb):
        # Function for converting from an integer to (bit)strings of length num_qubits
        strr = format(numb, "b") #convert to binary
        strr = '0' * (num_qubits - len(strr)) + strr
        return strr

    unif_cuts = [int_to_bs(i) for i in unif_cuts]
    unif_sizes = [common.eval_cut(nodes, edges, cut, reverseStep) for cut in unif_cuts]

    # Also get the corresponding distribution of cut sizes
    unique_counts_unif, unique_sizes_unif, cumul_counts_unif = get_size_dist(unif_counts, unif_sizes)

    return unif_cuts, unif_counts, unif_sizes, unique_counts_unif, unique_sizes_unif, cumul_counts_unif



def get_random_angles(rounds, restarts):
    """Create max_circuit number of random initial conditions

    Args:
        rounds (int): number of rounds in QAOA
        restarts (int): number of random initial conditions 

    Returns:
        restarts (list of lists of floats): list of length restarts. Each list element is a list of angles
    """
    # restarts = min(10, restarts)
    # Create random angles
    theta_min = [0] * 2 * rounds
    # Upper limit for betas=pi; upper limit for gammas=2pi
    theta_max = [np.pi] * rounds + [2 * np.pi] * rounds
    thetas = np.random.uniform(
        low=theta_min, high=theta_max, size=(restarts, 2 * rounds)
    )
    thetas = thetas.tolist()
    return thetas
        


def get_restart_angles(thetas_array, rounds, restarts):
    """
    Create random initial conditions for the restart loop.
    thetas_array takes precedence over restarts.
    If the user inputs valid thetas_array, restarts will be inferred accordingly.
    If thetas_array is None and restarts is 1, return all 1's as initial angles.
    If thetas_array is None and restarts >1, generate restarts number of random initial conditions
    If only one set of random angles are desired, then the user needs to create them and send as thetas_array 

    Args:
        thetas_array (list of lists of floats): list of initial angles.
        restarts (int): number of random initial conditions
        rounds (int): of QAOA

    Returns:
        thetas (list of lists. Shape = (max_circuits, 2 * rounds))
        restarts : int
    """
    assert type(restarts) == int and restarts > 0, "max_circuits must be an integer greater than 0"
    default_angles = [[1] * 2 * rounds]
    default_restarts = 1
    if thetas_array is None:
        if restarts == 1:
            # if the angles are none, but restarts equals 1, use default of all 1's
            return default_angles, default_restarts
        else:
            # restarts can only be greater than 1.
            return get_random_angles(rounds, restarts), restarts
        
    if type(thetas_array) != list:
        # thetas_array is not None, but is also not a list.
        print("thetas_array is not a list. Using random angles.")
        return get_random_angles(rounds, restarts), restarts
    
    # At this point, thetas_array is a list. check if thetas_array is a list of lists
    if not all([type(item) == list for item in thetas_array]):
        # if every list element is not a list, return random angles
        print("thetas_array is not a list of lists. Using random angles.")
        return get_random_angles(rounds, restarts), restarts
        
    if not all([len(item) == 2 * rounds for item in thetas_array]):
        # If not all list elements are lists of the correct length...
        print("Each element of thetas_array must be a list of length 2 * rounds. Using random angles.")
        return get_random_angles(rounds, restarts), restarts
    
    # At this point, thetas_array is a list of lists of length 2*rounds. All conditions are satisfied. Return inputted angles.
    return thetas_array, len(thetas_array)
    
    
#%% Storing final iteration data to json file, and to metrics.circuit_metrics_final_iter

def save_runtime_data(result_dict): # This function will need changes, since circuit metrics dictionaries are now different
    cm = result_dict.get('circuit_metrics')
    detail = result_dict.get('circuit_metrics_detail', None)
    detail_2 = result_dict.get('circuit_metrics_detail_2', None)
    benchmark_inputs = result_dict.get('benchmark_inputs', None)
    final_iter_metrics = result_dict.get('circuit_metrics_final_iter')
    backend_id = result_dict.get('benchmark_inputs').get('backend_id')
    
    metrics.circuit_metrics_detail_2 = detail_2
    
    for width in detail_2:
        # unique_id = restart_ind * 1000 + minimizer_iter_ind
        
        restart_ind_list = list(detail_2.get(width).keys())
        for restart_ind in restart_ind_list:
            degree = cm[width]['1']['degree']
            opt = final_iter_metrics[width]['1']['optimal_value']
            instance_filename = os.path.join(os.path.dirname(__file__),
                "..", "_common", common.INSTANCE_DIR, f"mc_{int(width):03d}_{int(degree):03d}_000.txt")
            metrics.circuit_metrics[width] = detail.get(width)
            metrics.circuit_metrics['subtitle'] = cm.get('subtitle')
            
            finIterDict = final_iter_metrics[width][restart_ind]
            if benchmark_inputs['save_final_counts']:
                # if the final iteration cut counts were stored, retrieve them
                iter_dist = {'cuts' : finIterDict['cuts'], 'counts' : finIterDict['counts'], 'sizes' : finIterDict['sizes']}
            else:
                # The value of iter_dist does not matter otherwise
                iter_dist = None
            # Retrieve the distribution of cut sizes for the final iteration for this width and degree
            iter_size_dist = {'unique_sizes' : finIterDict['unique_sizes'], 'unique_counts' : finIterDict['unique_counts'], 'cumul_counts' : finIterDict['cumul_counts']}

            
            converged_thetas_list = finIterDict.get('converged_thetas_list')
            parent_folder_save = os.path.join('__data', f'{metrics.get_backend_label(backend_id)}')
            store_final_iter_to_metrics_json(
                num_qubits=int(width),
                degree = int(degree),
                restart_ind=int(restart_ind),
                num_shots=int(benchmark_inputs['num_shots']),
                converged_thetas_list=converged_thetas_list,
                opt=opt,
                iter_size_dist=iter_size_dist,
                iter_dist = iter_dist,
                dict_of_inputs=benchmark_inputs,
                parent_folder_save=parent_folder_save,
                save_final_counts=False,
                save_res_to_file=True,
                _instances=None
            )


def store_final_iter_to_metrics_json(num_qubits,
                                     degree,
                                     restart_ind,
                                     num_shots,
                                     converged_thetas_list,
                                     opt,
                                     iter_size_dist,
                                     iter_dist,
                                     parent_folder_save,
                                     dict_of_inputs,
                                     save_final_counts,
                                     save_res_to_file,
                                     _instances=None):
    """
    For a given graph (specified by num_qubits and degree),
    1. For a given restart, store properties of the final minimizer iteration to metrics.circuit_metrics_final_iter, and
    2. Store various properties for all minimizer iterations for each restart to a json file.
    Parameters
    ----------
        num_qubits, degree, restarts, num_shots : ints
        parent_folder_save : string (location where json file will be stored)
        dict_of_inputs : dictionary of inputs that were given to run()
        save_final_counts: bool. If true, save counts, cuts and sizes for last iteration to json file.
        save_res_to_file: bool. If False, do not save data to json file.
        iter_size_dist : dictionary containing distribution of cut sizes.  Keys are 'unique_sizes', 'unique_counts' and         'cumul_counts'
        opt (int) : Max Cut value
    """
    # In order to compare with uniform random sampling, get some samples
    unif_cuts, unif_counts, unif_sizes, unique_counts_unif, unique_sizes_unif, cumul_counts_unif = uniform_cut_sampling(
        num_qubits, degree, num_shots, _instances)
    unif_dict = {'unique_counts_unif': unique_counts_unif,
                 'unique_sizes_unif': unique_sizes_unif,
                 'cumul_counts_unif': cumul_counts_unif}  # store only the distribution of cut sizes, and not the cuts themselves

    # Store properties such as (cuts, counts, sizes) of the final iteration, the converged theta values, as well as the known optimal value for the current problem, in metrics.circuit_metrics_final_iter. Also store uniform cut sampling results
    metrics.store_props_final_iter(num_qubits, restart_ind, 'optimal_value', opt)
    metrics.store_props_final_iter(num_qubits, restart_ind, None, iter_size_dist)
    metrics.store_props_final_iter(num_qubits, restart_ind, 'converged_thetas_list', converged_thetas_list)
    metrics.store_props_final_iter(num_qubits, restart_ind, None, unif_dict)
    # metrics.store_props_final_iter(num_qubits, restart_ind, None, iter_dist) # do not store iter_dist, since it takes a lot of memory for larger widths, instead, store just iter_size_dist


    if save_res_to_file:
        # Save data to a json file
        dump_to_json(parent_folder_save, num_qubits,
                     degree, restart_ind, iter_size_dist, iter_dist, dict_of_inputs, converged_thetas_list, opt, unif_dict,
                     save_final_counts=save_final_counts)

def dump_to_json(parent_folder_save, num_qubits, degree, restart_ind, iter_size_dist, iter_dist,
                 dict_of_inputs, converged_thetas_list, opt, unif_dict, save_final_counts=False):
    """
    For a given problem (specified by number of qubits and graph degree) and restart_index, 
    save the evolution of various properties in a json file.
    Items stored in the json file: Data from all iterations (iterations), inputs to run program ('general properties'), converged theta values ('converged_thetas_list'), max cut size for the graph (optimal_value), distribution of cut sizes for random uniform sampling (unif_dict), and distribution of cut sizes for the final iteration (final_size_dist)
    if save_final_counts is True, then also store the distribution of cuts 
    """
    if not os.path.exists(parent_folder_save): os.makedirs(parent_folder_save)
    store_loc = os.path.join(parent_folder_save,'width_{}_restartInd_{}.json'.format(num_qubits, restart_ind))
    
    # Obtain dictionary with iterations data corresponding to given restart_ind 
    all_restart_ids = list(metrics.circuit_metrics[str(num_qubits)].keys())
    ids_this_restart = [r_id for r_id in all_restart_ids if int(r_id) // 1000 == restart_ind]
    iterations_dict_this_restart =  {r_id : metrics.circuit_metrics[str(num_qubits)][r_id] for r_id in ids_this_restart}
    # Values to be stored in json file
    dict_to_store = {'iterations' : iterations_dict_this_restart}
    dict_to_store['general_properties'] = dict_of_inputs
    dict_to_store['converged_thetas_list'] = converged_thetas_list
    dict_to_store['optimal_value'] = opt
    dict_to_store['unif_dict'] = unif_dict
    dict_to_store['final_size_dist'] = iter_size_dist
    # Also store the value of counts obtained for the final counts
    if save_final_counts:
        dict_to_store['final_counts'] = iter_dist
                                        #iter_dist.get_counts()
    # Now save the output
    with open(store_loc, 'w') as outfile:
        json.dump(dict_to_store, outfile)

#%% Loading saved data (from json files)

def load_data_and_plot(folder=None, backend_id=None, **kwargs):
    """
    The highest level function for loading stored data from a previous run
    and plotting optgaps and area metrics

    Parameters
    ----------
    folder : string
        Directory where json files are saved.
    """
    _gen_prop = load_all_metrics(folder, backend_id=backend_id)
    if _gen_prop != None:
        gen_prop = {**_gen_prop, **kwargs}
        plot_results_from_data(**gen_prop)


def load_all_metrics(folder=None, backend_id=None):
    """
    Load all data that was saved in a folder.
    The saved data will be in json files in this folder

    Parameters
    ----------
    folder : string
        Directory where json files are saved.

    Returns
    -------
    gen_prop : dict
        of inputs that were used in maxcut_benchmark.run method
    """
    
    # if folder not passed in, create its name using standard format
    if folder is None:
        folder = f"__data/{metrics.get_backend_label(backend_id)}"
        
    # Note: folder here should be the folder where only the width=... files are stored, and not a folder higher up in the directory
    assert os.path.isdir(folder), f"Specified folder ({folder}) does not exist."
    
    metrics.init_metrics()
    
    list_of_files = os.listdir(folder)
    width_restart_file_tuples = [(*get_width_restart_tuple_from_filename(fileName), fileName)
                                 for (ind, fileName) in enumerate(list_of_files) if fileName.startswith("width")]  # list with elements that are tuples->(width,restartInd,filename)

    width_restart_file_tuples = sorted(width_restart_file_tuples, key=lambda x:(x[0], x[1])) #sort first by width, and then by restartInd
    distinct_widths = list(set(it[0] for it in width_restart_file_tuples)) 
    list_of_files = [
        [tup[2] for tup in width_restart_file_tuples if tup[0] == width] for width in distinct_widths
        ]
    
    # connot continue without at least one dataset
    if len(list_of_files) < 1:
        print("ERROR: No result files found")
        return None
        
    for width_files in list_of_files:
        # For each width, first load all the restart files
        for fileName in width_files:
            gen_prop = load_from_width_restart_file(folder, fileName)
        
        # next, do processing for the width
        method = gen_prop['method']
        if method == 2:
            num_qubits, _ = get_width_restart_tuple_from_filename(width_files[0])
            metrics.process_circuit_metrics_2_level(num_qubits)
            metrics.finalize_group(str(num_qubits))
            
    # override device name with the backend_id if supplied by caller
    if backend_id != None:
        metrics.set_plot_subtitle(f"Device = {backend_id}")
            
    return gen_prop


def load_from_width_restart_file(folder, fileName):
    """
    Given a folder name and a file in it, load all the stored data and store the values in metrics.circuit_metrics.
    Also return the converged values of thetas, the final counts and general properties.

    Parameters
    ----------
    folder : string
        folder where the json file is located
    fileName : string
        name of the json file

    Returns
    -------
    gen_prop : dict
        of inputs that were used in maxcut_benchmark.run method
    """
    
    # Extract num_qubits and s from file name
    num_qubits, restart_ind = get_width_restart_tuple_from_filename(fileName)
    print(f"Loading from {fileName}, corresponding to {num_qubits} qubits and restart index {restart_ind}")
    with open(os.path.join(folder, fileName), 'r') as json_file:
        data = json.load(json_file)
        gen_prop = data['general_properties']
        converged_thetas_list = data['converged_thetas_list']
        unif_dict = data['unif_dict']
        opt = data['optimal_value']
        if gen_prop['save_final_counts']:
            # Distribution of measured cuts
            final_counts = data['final_counts']
        final_size_dist = data['final_size_dist']
        
        backend_id = gen_prop.get('backend_id')
        metrics.set_plot_subtitle(f"Device = {backend_id}")
        
        # Update circuit metrics
        for circuit_id in data['iterations']:
            # circuit_id = restart_ind * 1000 + minimizer_loop_ind
            for metric, value in data['iterations'][circuit_id].items():
                metrics.store_metric(num_qubits, circuit_id, metric, value)
                
        method = gen_prop['method']
        if method == 2:
            metrics.store_props_final_iter(num_qubits, restart_ind, 'optimal_value', opt)
            metrics.store_props_final_iter(num_qubits, restart_ind, None, final_size_dist)
            metrics.store_props_final_iter(num_qubits, restart_ind, 'converged_thetas_list', converged_thetas_list)
            metrics.store_props_final_iter(num_qubits, restart_ind, None, unif_dict)
            if gen_prop['save_final_counts']:
                metrics.store_props_final_iter(num_qubits, restart_ind, None, final_counts)

    return gen_prop
    

def get_width_restart_tuple_from_filename(fileName):
    """
    Given a filename, extract the corresponding width and degree it corresponds to
    For example the file "width=4_degree=3.json" corresponds to 4 qubits and degree 3

    Parameters
    ----------
    fileName : TYPE
        DESCRIPTION.

    Returns
    -------
    num_qubits : int
        circuit width
    degree : int
        graph degree.

    """
    pattern = 'width_([0-9]+)_restartInd_([0-9]+).json'
    match = re.search(pattern, fileName)

    # assert match is not None, f"File {fileName} found inside folder. All files inside specified folder must be named in the format 'width_int_restartInd_int.json"
    num_qubits = int(match.groups()[0])
    degree = int(match.groups()[1])
    return (num_qubits,degree)

#%% Run method: Benchmarking loop

MAX_QUBITS = 24
iter_dist = {'cuts' : [], 'counts' : [], 'sizes' : []} # (list of measured bitstrings, list of corresponding counts, list of corresponding cut sizes)
iter_size_dist = {'unique_sizes' : [], 'unique_counts' : [], 'cumul_counts' : []} # for the iteration being executed, stores the distribution for cut sizes
saved_result = {  }
instance_filename = None

def run (min_qubits=3, max_qubits=6, skip_qubits=2,
        max_circuits=1, num_shots=100,
        method=1, rounds=1, degree=3, alpha=0.1, thetas_array=None, parameterized= False, do_fidelities=True,
        max_iter=30, score_metric='fidelity', x_metric='cumulative_exec_time', y_metric='num_qubits',
        fixed_metrics={}, num_x_bins=15, y_size=None, x_size=None, use_fixed_angles=False,
        objective_func_type = 'approx_ratio', plot_results = True,
        save_res_to_file = False, save_final_counts = False, detailed_save_names = False, comfort=False,
        backend_id='qasm_simulator', provider_backend=None, eta=0.5,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None,
        _instances=None):
    """
    Parameters
    ----------
    min_qubits : int, optional
        The smallest circuit width for which benchmarking will be done The default is 3.
    max_qubits : int, optional
        The largest circuit width for which benchmarking will be done. The default is 6.
    skip_qubits : int, optional
        Skip at least this many qubits during run loop. The default is 2.
    max_circuits : int, optional
        Number of restarts. The default is None.
    num_shots : int, optional
        Number of times the circut will be measured, for each iteration. The default is 100.
    method : int, optional
        If 1, then do standard metrics, if 2, implement iterative algo metrics. The default is 1.
    rounds : int, optional
        number of QAOA rounds. The default is 1.
    degree : int, optional
        degree of graph. The default is 3.
    thetas_array : list, optional
        list or ndarray of beta and gamma values. The default is None.
    use_fixed_angles : bool, optional
        use betas and gammas obtained from a 'fixed angles' table, specific to degree and rounds
    N : int, optional
        For the max % counts metric, choose the highest N% counts. The default is 10.
    alpha : float, optional
        Value between 0 and 1. The default is 0.1.
    parameterized : bool, optional
        Whether to use parametrized circuits or not. The default is False.
    do_fidelities : bool, optional
        Compute circuit fidelity. The default is True.
    max_iter : int, optional
        Number of iterations for the minimizer routine. The default is 30.
    score_metric : list or string, optional
        Which metrics are to be plotted in area metrics plots. The default is 'fidelity'.
    x_metric : list or string, optional
        Horizontal axis for area plots. The default is 'cumulative_exec_time'.
    y_metric : list or string, optional
        Vertical axis for area plots. The default is 'num_qubits'.
    fixed_metrics : TYPE, optional
        DESCRIPTION. The default is {}.
    num_x_bins : int, optional
        DESCRIPTION. The default is 15.
    y_size : TYPint, optional
        DESCRIPTION. The default is None.
    x_size : string, optional
        DESCRIPTION. The default is None.
    backend_id : string, optional
        DESCRIPTION. The default is 'qasm_simulator'.
    provider_backend : string, optional
        DESCRIPTION. The default is None.
    hub : string, optional
        DESCRIPTION. The default is "ibm-q".
    group : string, optional
        DESCRIPTION. The default is "open".
    project : string, optional
        DESCRIPTION. The default is "main".
    exec_options : string, optional
        DESCRIPTION. The default is None.
    objective_func_type : string, optional
        Objective function to be used by the classical minimizer algorithm. The default is 'approx_ratio'.
    plot_results : bool, optional
        Plot results only if True. The default is True.
    save_res_to_file : bool, optional
        Save results to json files. The default is True.
    save_final_counts : bool, optional
        If True, also save the counts from the final iteration for each problem in the json files. The default is True.
    detailed_save_names : bool, optional
        If true, the data and plots will be saved with more detailed names. Default is False
    confort : bool, optional    
        If true, print comfort dots during execution
    """

    # Store all the input parameters into a dictionary.
    # This dictionary will later be stored in a json file
    # It will also be used for sending parameters to the plotting function
    dict_of_inputs = locals()
    
    # Get angles for restarts. Thetas = list of lists. Lengths are max_circuits and 2*rounds
    thetas, max_circuits = get_restart_angles(thetas_array, rounds, max_circuits)
    
    # Update the dictionary of inputs
    dict_of_inputs = {**dict_of_inputs, **{'thetas_array': thetas, 'max_circuits' : max_circuits}}
    
    # Delete some entries from the dictionary
    for key in ["hub", "group", "project", "provider_backend", "exec_options"]:
        dict_of_inputs.pop(key)
    
    global maxcut_inputs
    maxcut_inputs = dict_of_inputs
    
    global QC_
    global circuits_done
    global minimizer_loop_index
    global opt_ts
    
    print(f"{benchmark_name} ({method}) Benchmark Program - Qiskit")

    QC_ = None
    
    # Create a folder where the results will be saved. Folder name=time of start of computation
    # In particular, for every circuit width, the metrics will be stored the moment the results are obtained
    # In addition to the metrics, the (beta,gamma) values obtained by the optimizer, as well as the counts
    # measured for the final circuit will be stored.
    # Use the following parent folder, for a more detailed 
    if detailed_save_names:
        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        parent_folder_save = os.path.join('__results', f'{backend_id}', objective_func_type, f'run_start_{start_time_str}')
    else:
        parent_folder_save = os.path.join('__results', f'{backend_id}', objective_func_type)
    
    if save_res_to_file and not os.path.exists(parent_folder_save): os.makedirs(os.path.join(parent_folder_save))
    
    # validate parameters
    max_qubits = max(4, max_qubits)
    max_qubits = min(MAX_QUBITS, max_qubits)
    min_qubits = min(max(4, min_qubits), max_qubits)
    skip_qubits = max(2, skip_qubits)
    
    # create context identifier
    if context is None: context = f"{benchmark_name} ({method}) Benchmark"
    
    degree = max(3, degree)
    rounds = max(1, rounds)
    
    # don't compute exectation unless fidelity is is needed
    global do_compute_expectation
    do_compute_expectation = do_fidelities
        
    # given that this benchmark does every other width, set y_size default to 1.5
    if y_size == None:
        y_size = 1.5
        
    # Choose the objective function to minimize, based on values of the parameters
    possible_approx_ratios = {'cvar_ratio', 'approx_ratio', 'gibbs_ratio', 'bestcut_ratio'}
    non_objFunc_ratios = possible_approx_ratios - { objective_func_type }
    function_mapper = {'cvar_ratio' : compute_cvar, 
                       'approx_ratio' : compute_sample_mean,
                       'gibbs_ratio' : compute_gibbs,
                       'bestcut_ratio' : compute_best_cut_from_measured}
                       
    # if using fixed angles, get thetas array from table
    if use_fixed_angles:
    
        # Load the fixed angle tables from data file
        fixed_angles = common.read_fixed_angles(
            os.path.join(os.path.dirname(__file__), '..', '_common', 'angles_regular_graphs.json'),
            _instances)
            
        thetas_array = common.get_fixed_angles_for(fixed_angles, degree, rounds)
        if thetas_array == None:
            print(f"ERROR: no fixed angles for rounds = {rounds}")
            return
           
    ##########
    
    # Initialize metrics module
    metrics.init_metrics()
    
    # Define custom result handler
    def execution_handler (qc, result, num_qubits, s_int, num_shots):  
     
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
        metrics.store_metric(num_qubits, s_int, 'fidelity', fidelity)

    def execution_handler2 (qc, result, num_qubits, s_int, num_shots):
        # Stores the results to the global saved_result variable
        global saved_result
        saved_result = result
     
    # Initialize execution module using the execution result handler above and specified backend_id
    # for method=2 we need to set max_jobs_active to 1, so each circuit completes before continuing
    if method == 2:
        ex.max_jobs_active = 1
        ex.init_execution(execution_handler2)
    else:
        ex.init_execution(execution_handler)
    
    # initialize the execution module with target information
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
        hub=hub, group=group, project=project, 
        exec_options=exec_options,
        context=context
    )

    # for noiseless simulation, set noise model to be None
    # ex.set_noise_model(None)

    ##########
    
    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    # DEVNOTE: increment by 2 to match the collection of problems in 'instance' folder
    for num_qubits in range(min_qubits, max_qubits + 1, 2):
        
        if method == 1:
            print(f"************\nExecuting [{max_circuits}] circuits for num_qubits = {num_qubits}")
        else:
            print(f"************\nExecuting [{max_circuits}] restarts for num_qubits = {num_qubits}")
        
        # If degree is negative, 
        if degree < 0 :
            degree = max(3, (num_qubits + degree))
            
        # Load the problem and its solution
        global instance_filename
        instance_filename = os.path.join(
            os.path.dirname(__file__),
            "..",
            "_common",
            common.INSTANCE_DIR,
            f"mc_{num_qubits:03d}_{degree:03d}_000.txt",
        )
        nodes, edges = common.read_maxcut_instance(instance_filename, _instances)
        opt, _ = common.read_maxcut_solution(
            instance_filename[:-4] + ".sol", _instances
        )
        
        # if the file does not exist, we are done with this number of qubits
        if nodes == None:
            print(f"  ... problem not found.")
            break
        
        for restart_ind in range(1, max_circuits + 1):
            # restart index should start from 1
            # Loop over restarts for a given graph
            
            # if not using fixed angles, get initial or random thetas from array saved earlier
            # otherwise use random angles (if restarts > 1) or [1] * 2 * rounds
            if not use_fixed_angles:
                thetas_array = thetas[restart_ind - 1]
                            
            if method == 1:
                # create the circuit for given qubit size and secret string, store time metric
                ts = time.time()
                
                # if using fixed angles in method 1, need to access first element
                # DEVNOTE: eliminate differences between method 1 and 2 and handling of thetas_array
                thetas_array_0 = thetas_array
                if use_fixed_angles:
                    thetas_array_0 = thetas_array[0]
                                       
                qc, params = MaxCut(num_qubits, restart_ind, edges, rounds, thetas_array_0, parameterized)
                metrics.store_metric(num_qubits, restart_ind, 'create_time', time.time()-ts)

                # collapse the sub-circuit levels used in this benchmark (for qiskit)
                qc2 = qc.decompose()

                # submit circuit for execution on target (simulator, cloud simulator, or hardware)
                ex.submit_circuit(qc2, num_qubits, restart_ind, shots=num_shots, params=params)

            if method == 2:
                # a unique circuit index used inside the inner minimizer loop as identifier
                minimizer_loop_index = 0 # Value of 0 corresponds to the 0th iteration of the minimizer
                start_iters_t = time.time()

                # Always start by enabling transpile ...
                ex.set_tranpilation_flags(do_transpile_metrics=True, do_transpile_for_execute=True)
                    
                logger.info(f'===============  Begin method 2 loop, enabling transpile')
                
                def expectation(thetas_array):
                    
                    # Every circuit needs a unique id; add unique_circuit_index instead of s_int
                    global minimizer_loop_index
                    unique_id = restart_ind * 1000 + minimizer_loop_index
                    # store thetas_array
                    metrics.store_metric(num_qubits, unique_id, 'thetas_array', thetas_array.tolist())
                    
                    #************************************************
                    #*** Circuit Creation and Decomposition start ***
                    # create the circuit for given qubit size, secret string and params, store time metric
                    ts = time.time()
                    qc, params = MaxCut(num_qubits, unique_id, edges, rounds, thetas_array, parameterized)
                    metrics.store_metric(num_qubits, unique_id, 'create_time', time.time()-ts)
                    
                    # also store the 'rounds' and 'degree' for each execution
                    # DEVNOTE: Currently, this is stored for each iteration. Reduce this redundancy
                    metrics.store_metric(num_qubits, unique_id, 'rounds', rounds)
                    metrics.store_metric(num_qubits, unique_id, 'degree', degree)
                    
                    # collapse the sub-circuit levels used in this benchmark (for qiskit)
                    qc2 = qc.decompose()
                    
                    # Circuit Creation and Decomposition end
                    #************************************************
                    
                    #************************************************
                    #*** Quantum Part: Execution of Circuits ***
                    # submit circuit for execution on target with the current parameters
                    ex.submit_circuit(qc2, num_qubits, unique_id, shots=num_shots, params=params)
                    
                    # Must wait for circuit to complete
                    #ex.throttle_execution(metrics.finalize_group)
                    ex.finalize_execution(None, report_end=False)    # don't finalize group until all circuits done
                    
                    # after first execution and thereafter, no need for transpilation if parameterized
                    if parameterized:
                        ex.set_tranpilation_flags(do_transpile_metrics=False, do_transpile_for_execute=False)
                        logger.info(f'**** First execution complete, disabling transpile')
                    #************************************************
                    
                    global saved_result
                    # Fidelity Calculation and Storage
                    _, fidelity = analyze_and_print_result(qc, saved_result, num_qubits, unique_id, num_shots) 
                    metrics.store_metric(num_qubits, unique_id, 'fidelity', fidelity)
                    
                    #************************************************
                    #*** Classical Processing of Results - essential to optimizer ***
                    global opt_ts
                    dict_of_vals = dict()
                    # Start counting classical optimizer time here again
                    tc1 = time.time()
                    cuts, counts, sizes = compute_cutsizes(saved_result, nodes, edges)
                    # Compute the value corresponding to the objective function first
                    dict_of_vals[objective_func_type] = function_mapper[objective_func_type](counts, sizes, alpha = alpha)
                    # Store the optimizer time as current time- tc1 + ts - opt_ts, since the time between tc1 and ts is not time used by the classical optimizer.
                    metrics.store_metric(num_qubits, unique_id, 'opt_exec_time', time.time() - tc1 + ts - opt_ts)
                    # Note: the first time it is stored it is just the initialization time for optimizer
                    #************************************************
                    
                    #************************************************
                    #*** Classical Processing of Results - not essential for optimizer. Used for tracking metrics ***
                    # Compute the distribution of cut sizes; store them under metrics
                    unique_counts, unique_sizes, cumul_counts = get_size_dist(counts, sizes)
                    global iter_size_dist
                    iter_size_dist = {'unique_sizes' : unique_sizes, 'unique_counts' : unique_counts, 'cumul_counts' : cumul_counts}
                    metrics.store_metric(num_qubits, unique_id, None, iter_size_dist)

                    # Compute and the other metrics (eg. cvar, gibbs and max N % if the obj function was set to approx ratio)
                    for s in non_objFunc_ratios:
                        dict_of_vals[s] = function_mapper[s](counts, sizes, alpha = alpha)
                    # Store the ratios
                    dict_of_ratios = { key : -1 * val / opt for (key, val) in dict_of_vals.items()}
                    dict_of_ratios['gibbs_ratio'] = dict_of_ratios['gibbs_ratio'] / eta 
                    metrics.store_metric(num_qubits, unique_id, None, dict_of_ratios)
                    # Get the best measurement and store it
                    best = - compute_best_cut_from_measured(counts, sizes)
                    metrics.store_metric(num_qubits, unique_id, 'bestcut_ratio', best / opt)
                    # Also compute and store the weights of cuts at three quantile values
                    quantile_sizes = compute_quartiles(counts, sizes)
                    # Store quantile_optgaps as a list (allows storing in json files)
                    metrics.store_metric(num_qubits, unique_id, 'quantile_optgaps', (1 - quantile_sizes / opt).tolist()) 
                    
                    # Also store the cuts, counts and sizes in a global variable, to allow access elsewhere
                    global iter_dist
                    iter_dist = {'cuts' : cuts, 'counts' : counts, 'sizes' : sizes}
                    minimizer_loop_index += 1
                    #************************************************
                    
                    if comfort:
                        if minimizer_loop_index == 1: print("")
                        print(".", end ="")

                    # reset timer for optimizer execution after each iteration of quantum program completes
                    opt_ts = time.time()
                    
                    return dict_of_vals[objective_func_type]
                
                # after first execution and thereafter, no need for transpilation if parameterized
                # DEVNOTE: this appears to NOT be needed, as we can turn these off after 
                def callback(xk):
                    if parameterized:
                        ex.set_tranpilation_flags(do_transpile_metrics=False, do_transpile_for_execute=False)

                opt_ts = time.time()
                # perform the complete algorithm; minimizer invokes 'expectation' function iteratively
                ##res = minimize(expectation, thetas_array, method='COBYLA', options = { 'maxiter': max_iter}, callback=callback)

                res = minimize(expectation, thetas_array, method='COBYLA', options = { 'maxiter': max_iter})
                # To-do: Set bounds for the minimizer
                
                unique_id = restart_ind * 1000 + 0
                metrics.store_metric(num_qubits, unique_id, 'opt_exec_time', time.time()-opt_ts)
                
                if comfort:
                    print("")

                # Save final iteration data to metrics.circuit_metrics_final_iter
                # This data includes final counts, cuts, etc.
                store_final_iter_to_metrics_json(num_qubits=num_qubits, 
                                                 degree=degree, 
                                                 restart_ind=restart_ind,
                                                 num_shots=num_shots, 
                                                 converged_thetas_list=res.x.tolist(),
                                                 opt=opt,
                                                 iter_size_dist=iter_size_dist, iter_dist=iter_dist, parent_folder_save=parent_folder_save,
                                                 dict_of_inputs=dict_of_inputs, save_final_counts=save_final_counts,
                                                 save_res_to_file=save_res_to_file, _instances=_instances)

        # for method 2, need to aggregate the detail metrics appropriately for each group
        # Note that this assumes that all iterations of the circuit have completed by this point
        if method == 2:                  
            metrics.process_circuit_metrics_2_level(num_qubits)
            metrics.finalize_group(str(num_qubits))
            
    # Wait for some active circuits to complete; report metrics when groups complete
    ex.throttle_execution(metrics.finalize_group)
        
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    ##########
    
    global print_sample_circuit
    if print_sample_circuit:
        # print a sample circuit
        print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    #if method == 1: print("\nQuantum Oracle 'Uf' ="); print(Uf_ if Uf_ != None else " ... too large!")

    # Plot metrics for all circuit sizes
    if method == 1:
        metrics.plot_metrics(f"Benchmark Results - {benchmark_name} ({method}) - Qiskit",
                options=dict(shots=num_shots,rounds=rounds))
    elif method == 2:
        #metrics.print_all_circuit_metrics()
        if plot_results:
            plot_results_from_data(**dict_of_inputs)

# ******************************

def plot_results_from_data(num_shots=100, rounds=1, degree=3, max_iter=30, max_circuits = 1,
            objective_func_type='approx_ratio', method=2, use_fixed_angles=False,
            score_metric='fidelity', x_metric='cumulative_exec_time', y_metric='num_qubits', fixed_metrics={},
            num_x_bins=15, y_size=None, x_size=None, x_min=None, x_max=None,
            offset_flag=False,      # default is False for QAOA
            detailed_save_names=False, **kwargs):
    """
    Plot results
    """

    if detailed_save_names:
        # If detailed names are desired for saving plots, put date of creation, etc.
        cur_time=datetime.datetime.now()
        dt = cur_time.strftime("%Y-%m-%d_%H-%M-%S")
        short_obj_func_str = metrics.score_label_save_str[objective_func_type]
        suffix = f'-s{num_shots}_r{rounds}_d{degree}_mi{max_iter}_of-{short_obj_func_str}_{dt}' #of=objective function
    else:
        short_obj_func_str = metrics.score_label_save_str[objective_func_type]
        suffix = f'of-{short_obj_func_str}' #of=objective function
        
    obj_str = metrics.known_score_labels[objective_func_type]
    options = {'shots' : num_shots, 'rounds' : rounds, 'degree' : degree, 'restarts' : max_circuits, 'fixed_angles' : use_fixed_angles, '\nObjective Function' : obj_str}
    suptitle = f"Benchmark Results - MaxCut ({method}) - Qiskit"
    
    metrics.plot_all_area_metrics(f"Benchmark Results - MaxCut ({method}) - Qiskit",
                score_metric=score_metric, x_metric=x_metric, y_metric=y_metric,
                fixed_metrics=fixed_metrics, num_x_bins=num_x_bins,
                x_size=x_size, y_size=y_size, x_min=x_min, x_max=x_max,
                offset_flag=offset_flag,
                options=options, suffix=suffix)
    
    metrics.plot_metrics_optgaps(suptitle, options=options, suffix=suffix, objective_func_type = objective_func_type)
    
    # this plot is deemed less useful
    #metrics.plot_ECDF(suptitle=suptitle, options=options, suffix=suffix)

    all_widths = list(metrics.circuit_metrics_final_iter.keys())
    all_widths = [int(ii) for ii in all_widths]
    list_of_widths = [all_widths[-1]]
    metrics.plot_cutsize_distribution(suptitle=suptitle,options=options, suffix=suffix, list_of_widths = list_of_widths)
    
    metrics.plot_angles_polar(suptitle = suptitle, options = options, suffix = suffix)

# if main, execute method
if __name__ == '__main__': run()

# %%
