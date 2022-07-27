"""
MaxCut Benchmark Program - Qiskit
"""

import os
import sys
import time
from collections import namedtuple

import datetime
import json
import math
import numpy as np
from scipy.optimize import minimize
import re

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
            qc.rzz(2 * par.gamma, i, j)

        qc.barrier()
        
        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * par.beta, i)

    return qc
   

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
            beta_params.append(Parameter("𝞫" + str(i)))
        for j, gamma in enumerate(gammas):
            gamma_params.append(Parameter("𝞬" + str(j)))
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
    if do_compute_expectation:
        compute_expectation(qc, num_qubits, secret_int)
   
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
    counts = np.array(list(results.get_counts().values()))
    sizes = np.array([common.eval_cut(nodes, edges, cut) for cut in cuts])

    return cuts, counts, sizes


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


def compute_maxN_mean(counts, sizes, N = 5, **kwargs):
    """
    Compute the average size of the top most frequently measured cuts
    Choose N% of the cuts. 
    The average is weighted by the corresponding counts

    Parameters
    ----------
    counts : ndarray of ints
        measured counts corresponding to cuts
    sizes : ndarray of ints
        cut sizes (i.e. number of edges crossing the cut)
    N : int, optional
        The default is 5.
    **kwargs : optional arguments
        will be ignored

    Returns
    -------
    float
    """
    # Convert counts and sizes to ndarrays, if they are lists
    counts, sizes = np.array(counts), np.array(sizes)

    # Obtain the indices corresponding to the largest N% values of counts
    # Thereafter, sort the counts and sizes arrays in the order specified by sort_inds
    num_cuts = counts.size
    how_many_top_counts = min(math.ceil(N / 100 * num_cuts),
                              num_cuts)
    sort_inds = np.argsort(counts)[-how_many_top_counts:]
    counts = counts[sort_inds]
    sizes = sizes[sort_inds]

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

def compute_best_cut_from_measured(counts, sizes, **kwargs):
    """From the measured cuts, return the size of the largest cut
    """
    return np.max(sizes)


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

def uniform_cut_sampling(num_qubits, degree, num_shots):
    """
    For a given problem, i.e. num_qubits and degree values, sample cuts uniformly
    at random from all possible cuts, num_shots number of times. Return the corresponding
    cuts, counts and cut sizes.
    """
    
    # First, load the nodes and edges corresponding to the problem 
    instance_filename = os.path.join(os.path.dirname(__file__),
                                     "..", "_common", common.INSTANCE_DIR, 
                                     f"mc_{num_qubits:03d}_{degree:03d}_000.txt")
    nodes, edges = common.read_maxcut_instance(instance_filename)
    
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
    unif_sizes = [common.eval_cut(nodes, edges, cut) for cut in unif_cuts]

    return unif_cuts, unif_counts, unif_sizes


#%% Storing final iteration data to json file, and to metrics.circuit_metrics_final_iter

def store_final_iter_to_metrics_json(num_qubits, s_int, num_shots, res,
                                     parent_folder_save, dict_of_inputs, save_final_counts,
                                     save_res_to_file):
    """
    Store various results from the last iteration a problem, i.e. graph of size=num_qubits
    and degree=s_int, into metrics. Store the data also in a json file
    Parameters
    ----------
        num_qubits, s_int, num_shots : ints
        parent_folder_save : string (location where json file will be stored)
        dict_of_inputs : dictionary of inputs that were given to run()
        save_final_counts: bool. If true, save counts, cuts and sizes for last iteration to json file.
        save_res_to_file: bool. If False, do not save data to json file.
    """
    # In order to compare with uniform random sampling, get some samples
    unif_cuts, unif_counts, unif_sizes = uniform_cut_sampling(num_qubits, s_int, num_shots)
    unif_dict = {'unif_cuts' : unif_cuts,
                 'unif_counts' : unif_counts,
                 'unif_sizes' : unif_sizes}
    
    # Store properties such as (cuts, counts, sizes) of the final iteration,
    # the converged theta values, as well as the known optimal value for 
    # the current problem, in metrics.circuit_metrics_final_iter
    # Also store uniform cut sampling results
    opt, _ = common.read_maxcut_solution(instance_filename[:-4]+'.sol')
    metrics.store_props_final_iter(num_qubits, s_int, 'optimal_value', opt)
    metrics.store_props_final_iter(num_qubits, s_int, None, iter_dist)
    metrics.store_props_final_iter(num_qubits, s_int, 'converged_thetas_list', res.x.tolist())
    metrics.store_props_final_iter(num_qubits, s_int, None, unif_dict)
    
    if save_res_to_file:
        # Save data to a json file
        dump_to_json(parent_folder_save, num_qubits,
                     s_int, dict_of_inputs, res, opt, unif_dict,
                     save_final_counts=save_final_counts)

def dump_to_json(parent_folder_save, num_qubits, s_int, 
                 dict_of_inputs, res, opt, unif_dict, save_final_counts=True):
    """
    Save the results to a json file (corresponding to a given regular graph,
    specified by number of nodes and degree)
    """
    store_loc = os.path.join(parent_folder_save,'width_{}_degree_{}.json'.format(num_qubits,s_int))
    dict_to_store = {'iterations' : metrics.circuit_metrics[str(num_qubits)].copy()}
    dict_to_store['general properties'] = dict_of_inputs
    dict_to_store['converged_thetas_list'] = res.x.tolist() #save as list instead of array: this allows us to store in the json file
    dict_to_store['optimal_value'] = opt
    dict_to_store['unif_dict'] = unif_dict
    # Also store the value of counts obtained for the final counts
    if save_final_counts:
        dict_to_store['final_counts'] = iter_dist.copy()
                                        #iter_dist.get_counts()
    # Now save the output
    with open(store_loc, 'w') as outfile:
        json.dump(dict_to_store, outfile)

#%% Loading saved data (from json files)

def load_data_and_plot(folder):
    """
    The highest level function for loading stored data from a previous run
    and plotting optgaps and area metrics

    Parameters
    ----------
    folder : string
        Directory where json files are saved.
    """
    gen_prop = load_all_metrics(folder)
    plot_results_from_data(**gen_prop)


def load_all_metrics(folder):
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
    
    metrics.init_metrics()
    assert os.path.isdir(folder), f"Specified folder ({folder}) does not exist."
    
    list_of_files = os.listdir(folder)
    width_degree_file_tuples = [(*get_width_degree_tuple_from_filename(fileName),fileName) 
                           for (ind,fileName) in enumerate(list_of_files)] # list with elements that are tuples->(width,degree,filename)
    
    width_degree_file_tuples = sorted(width_degree_file_tuples, key=lambda x:(x[0], x[1])) #sort first by width, and then by degree
    list_of_files = [tup[2] for tup in width_degree_file_tuples]
    
    for ind, fileName in enumerate(list_of_files):
        gen_prop = load_from_width_degree_file(folder, fileName)
    
    return gen_prop


def load_from_width_degree_file(folder, fileName):
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
    num_qubits, s_int = get_width_degree_tuple_from_filename(fileName)
    print(f"Loading {fileName}, corresponding to {num_qubits} qubits and degree {s_int}")
    with open(os.path.join(folder, fileName), 'r') as json_file:
        data = json.load(json_file)
        gen_prop = data['general properties']
        converged_thetas_list = data['converged_thetas_list']
        unif_dict = data['unif_dict']
        opt = data['optimal_value']
        final_counts = data['final_counts'] # This is a
        
        ex.set_execution_target(backend_id = gen_prop["backend_id"], 
                                provider_backend = gen_prop["provider_backend"],
                                hub = gen_prop["hub"], 
                                group = gen_prop["group"], 
                                project = gen_prop["project"],
                                exec_options = gen_prop["exec_options"])
        
        # Update circuit metrics
        for circuit_id in data['iterations']:
            for metric, value in data['iterations'][circuit_id].items():
                metrics.store_metric(num_qubits, circuit_id, metric, value)
                
        method = gen_prop['method']
        if method == 2:
            metrics.process_circuit_metrics_2_level(num_qubits)
            metrics.finalize_group(str(num_qubits))
            metrics.store_props_final_iter(num_qubits, s_int, None, final_counts)
            metrics.store_props_final_iter(num_qubits, s_int, 'optimal_value', opt)
            metrics.store_props_final_iter(num_qubits, s_int, 'converged_thetas_list', converged_thetas_list)
            metrics.store_props_final_iter(num_qubits, s_int, None, unif_dict)

    return gen_prop
    

def get_width_degree_tuple_from_filename(fileName):
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
    pattern = 'width_([0-9]+)_degree_([0-9]+).json'
    match = re.search(pattern, fileName)

    num_qubits = int(match.groups()[0])
    degree = int(match.groups()[1])
    return (num_qubits,degree)

#%% Run method: Benchmarking loop

MAX_QUBITS = 24
iter_dist = {'cuts' : [], 'counts' : [], 'sizes' : []} # (list of measured bitstrings, list of corresponding counts, list of corresponding cut sizes)
saved_result = {  }
instance_filename = None
def run (min_qubits=3, max_qubits=6, max_circuits=3, num_shots=100,
        method=1, rounds=1, degree=3, thetas_array=None, N=10, alpha=0.1, parameterized= False, do_fidelities=True,
        max_iter=30, score_metric='fidelity', x_metric='cumulative_exec_time', y_metric='num_qubits',
        fixed_metrics={}, num_x_bins=15, y_size=None, x_size=None,
        objective_func_type = 'approx_ratio', plot_results = True,
        save_res_to_file = False, save_final_counts = False, detailed_save_names = False,
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None):
    """

    Parameters
    ----------
    min_qubits : int, optional
        The smallest circuit width for which benchmarking will be done The default is 3.
    max_qubits : int, optional
        The largest circuit width for which benchmarking will be done. The default is 6.
    max_circuits : int, optional
        Number of problems to solve for each circuit width. The default is 3.
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
    """

    # Store all the input parameters into a dictionary.
    # This dictionary will later be stored in a json file
    # It will also be used for sending parameters to the plotting function
    dict_of_inputs = locals()

    global QC_
    global circuits_done
    global unique_circuit_index
    global opt_ts
    
    print("MaxCut Benchmark Program - Qiskit")

    QC_ = None
    
    # Create a folder where the results will be saved. Folder name=time of start of computation
    # In particular, for every circuit width, the metrics will be stored the moment the results are obtained
    # In addition to the metrics, the (beta,gamma) values obtained by the optimizer, as well as the counts
    # measured for the final circuit will be stored.
    # Use the following parent folder, for a more detailed 
    if detailed_save_names:
        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        parent_folder_save = os.path.join('__results', objective_func_type,
                                          f'{backend_id}/run_start_{start_time_str}')
    else:
        parent_folder_save = os.path.join('__results', 'objectiveFunction_' + objective_func_type)
    
    if save_res_to_file and not os.path.exists(parent_folder_save): os.makedirs(os.path.join(parent_folder_save))
    
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
        
    # Choose the objective function to minimize, based on values of the parameters
    possible_approx_ratios = {'cvar_approx_ratio', 'Max_N_approx_ratio', 'approx_ratio'}
    non_objFunc_ratios = possible_approx_ratios - { objective_func_type }
    function_mapper = {'cvar_approx_ratio' : compute_cvar, 
                       'Max_N_approx_ratio' : compute_maxN_mean,
                       'approx_ratio' : compute_sample_mean}

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
            
            # Load the problem and its solution
            global instance_filename
            instance_filename = os.path.join(os.path.dirname(__file__),
                "..", "_common", common.INSTANCE_DIR, f"mc_{num_qubits:03d}_{i:03d}_000.txt")
            nodes, edges = common.read_maxcut_instance(instance_filename)
            opt, _ = common.read_maxcut_solution(instance_filename[:-4]+'.sol')
            
            # if the file does not exist, we are done with this number of qubits
            if nodes == None:
                print(f"  ... problem {i:03d} not found, limiting to {circuits_complete} circuit(s).")
                break;

            circuits_complete += 1
        
            if method == 1:
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
                    
                    # Every circuit needs a unique id; add unique_circuit_index instead of s_int
                    global unique_circuit_index
                    unique_id = s_int * 1000 + unique_circuit_index
                    
                    #************************************************
                    #*** Circuit Creation and Decomposition start ***
                    # create the circuit for given qubit size and secret string, store time metric
                    ts = time.time()
                    qc = MaxCut(num_qubits, unique_id, edges, rounds, thetas_array, parameterized)
                    metrics.store_metric(num_qubits, unique_id, 'create_time', time.time()-ts)
                    # also store the 'rounds' for each execution
                    metrics.store_metric(num_qubits, unique_id, 'rounds', rounds)
                    # collapse the sub-circuit levels used in this benchmark (for qiskit)
                    qc2 = qc.decompose()
                    # Circuit Creation and Decomposition end
                    #************************************************
                    
                    
                    #************************************************
                    #*** Quantum Part: Execution of Circuits ***
                    # submit circuit for execution on target (simulator, cloud simulator, or hardware)
                    ex.submit_circuit(qc2, num_qubits, unique_id, shots=num_shots)
                    # Must wait for circuit to complete
                    #ex.throttle_execution(metrics.finalize_group)
                    ex.finalize_execution(None, report_end=False)    # don't finalize group until all circuits done
                    #************************************************
                    
                    
                    global saved_result
                    # Fidelity Calculation and Storage
                    counts, fidelity = analyze_and_print_result(qc, saved_result, num_qubits, int(s_int), num_shots) 
                    metrics.store_metric(num_qubits, unique_id, 'fidelity', fidelity)
                    
                    
                    #************************************************
                    #*** Classical Processing of Results - essential to optimizer ***
                    global opt_ts
                    dict_of_vals = dict()
                    # Start counting classical optimizer time here again
                    tc1 = time.time()
                    cuts, counts, sizes = compute_cutsizes(saved_result, nodes, edges)
                    # Compute the value corresponding to the objective function first
                    dict_of_vals[objective_func_type] = function_mapper[objective_func_type](counts, sizes, alpha = alpha, N = N)
                    # Store the optimizer time as current time- tc1 + ts - opt_ts, since the time between tc1 and ts is not time used by the classical optimizer.
                    metrics.store_metric(num_qubits, unique_id, 'opt_exec_time', time.time() - tc1 + ts - opt_ts)
                    # Note: the first time it is stored it is just the initialization time for optimizer
                    #************************************************
                    
                    
                    #************************************************
                    #*** Classical Processing of Results - not essential for optimizer. Used for tracking metrics ***
                    # Compute and the other two metrics (eg. cvar and max N % if the obj function was set to approx ratio)
                    for s in non_objFunc_ratios:
                        dict_of_vals[s] = function_mapper[s](counts, sizes, alpha = alpha, N = N)
                    # Store the ratios
                    dict_of_ratios = { key : -1 * val / opt for (key, val) in dict_of_vals.items()}
                    metrics.store_metric(num_qubits, unique_id, None, dict_of_ratios)
                    # Get the best measurement and store it
                    best = compute_best_cut_from_measured(counts, sizes)
                    metrics.store_metric(num_qubits, unique_id, 'bestCut_approx_ratio', best / opt)
                    # Also compute and store the weights of cuts at three quantile values
                    quantile_sizes = compute_quartiles(counts, sizes)
                    # Store quantile_optgaps as a list (allows storing in json files)
                    metrics.store_metric(num_qubits, unique_id, 'quantile_optgaps', (1 - quantile_sizes / opt).tolist()) 
                    
                    # Also store the cuts, counts and sizes, in a global variable, to allow access elsewhere
                    global iter_dist
                    iter_dist = {'cuts' : cuts, 'counts' : counts.tolist(), 'sizes' : sizes.tolist()}
                    unique_circuit_index += 1
                    #************************************************
                    
                    
                    # reset timer for optimizer execution after each iteration of quantum program completes
                    opt_ts = time.time()
                    
                    return dict_of_vals[objective_func_type]

                opt_ts = time.time()
                # perform the complete algorithm; minimizer invokes 'expectation' function iteratively
                res = minimize(expectation, thetas_array, method='COBYLA', options = { 'maxiter': max_iter})
                # To-do: Set bounds for the minimizer
                
                unique_circuit_index = 0
                unique_id = s_int*1000 + unique_circuit_index
                metrics.store_metric(num_qubits, unique_id, 'opt_exec_time', time.time()-opt_ts)
                
                # Save final iteration data to metrics.circuit_metrics_final_iter
                # This data includes final counts, cuts, etc.
                store_final_iter_to_metrics_json(num_qubits, s_int, num_shots, res,
                                                 parent_folder_save=parent_folder_save,
                                                 dict_of_inputs=dict_of_inputs,save_final_counts=save_final_counts,
                                                 save_res_to_file=save_res_to_file)

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
        if plot_results:
            plot_results_from_data(**dict_of_inputs)

def plot_results_from_data(num_shots=100, rounds=1, degree=3, max_iter=30, 
                 objective_func_type='approx_ratio', method=2, score_metric='fidelity',
                 x_metric='cumulative_exec_time', y_metric='num_qubits', fixed_metrics={},
                 num_x_bins=15, y_size=None, x_size=None,
                 detailed_save_names=False, **kwargs):
    """
    Plot results
    """
    if detailed_save_names:
        # If detailed names are desired for saving plots, put date of creation, etc.
        cur_time=datetime.datetime.now()
        dt = cur_time.strftime("%Y-%m-%d_%H-%M-%S")
        suffix = f'-s{num_shots}_r{rounds}_d{degree}_mi{max_iter}_method={objective_func_type}_{dt}'
    else:
        suffix = objective_func_type
        
    obj_str = metrics.known_score_labels[objective_func_type]
    options = {'shots' : num_shots, 'rounds' : rounds, 'degree' : degree,
               'Objective Function' : obj_str}

    metrics.plot_all_area_metrics(f"Benchmark Results - MaxCut ({method}) - Qiskit",
            score_metric=score_metric, x_metric=x_metric, 
            y_metric=y_metric, fixed_metrics=fixed_metrics,
            num_x_bins=num_x_bins, x_size=x_size, y_size=y_size,
            options=options,
            suffix=suffix)
    
    metrics.plot_metrics_optgaps(f"Benchmark Results - MaxCut ({method}) - Qiskit",
                                 options=options,
                                 suffix=suffix, objective_func_type = objective_func_type)

    metrics.plot_ECDF(suptitle=f"Benchmark Results - MaxCut ({method}) - Qiskit",
                                 options=options, suffix=suffix)

    all_widths = list(metrics.circuit_metrics_final_iter.keys())
    all_widths = [int(ii) for ii in all_widths]
    list_of_widths = [all_widths[-1]]
    metrics.plot_cutsize_distribution(suptitle=f"Benchmark Results - MaxCut ({method}) - Qiskit",
                                  options=options, suffix=suffix, list_of_widths = list_of_widths)

# if main, execute method
if __name__ == '__main__': run()
