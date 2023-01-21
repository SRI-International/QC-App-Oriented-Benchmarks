"""
MaxCut Benchmark Program - Ocean
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

sys.path[1:1] = [ "_common", "_common/ocean", "maxcut/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/ocean", "../../maxcut/_common/" ]
import common
import execute as ex
import metrics as metrics
import HamiltonianCircuitProxy

logger = logging.getLogger(__name__)
fname, _, ext = os.path.basename(__file__).partition(".")
log_to_file = True

# Big-endian format is used in dwave. no need to reverse bitstrings
reverseStep = 1

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
            level=logging.INFO,
            format='%(asctime)s %(name)s - %(levelname)s:%(message)s')
        
except Exception as e:
    print(f'Exception {e} occured while configuring logger: bypassing logger config to prevent data loss')
    pass

np.random.seed(0)

maxcut_inputs = dict() #inputs to the run method

verbose = False

print_sample_circuit = True

# Indicates whether to perform the (expensive) pre compute of expectations
do_compute_expectation = True

# saved circuits for display
QC_ = None
Uf_ = None


#%% MaxCut circuit creation and fidelity analaysis functions
def create_circ(nqubits, edges):

    h = {}
    J = {}
    for e in edges:
        if e in J:
            J[e] += 1
        else:
            J[e] = 1

    circuitProxy = HamiltonianCircuitProxy.HamiltonianCircuitProxy()
    circuitProxy.h = h
    circuitProxy.J = J

    return circuitProxy
   

def MaxCut (num_qubits, secret_int, edges, measured = True):
    
    logger.info(f'*** Constructing NON-parameterized circuit for {num_qubits = } {secret_int}')
           
    # and create the hamiltonian
    circuitProxy = create_circ(num_qubits, edges)   

    # pre-compute and save an array of expected measurements
    if do_compute_expectation:
        logger.info('Computing expectation')
        #compute_expectation(qc, num_qubits, secret_int)

    # return a handle on the circuit
    return circuitProxy, None


_qc = []
beta_params = []
gamma_params = []
        
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
    cuts = list(results.keys())
    counts = list(results.values())
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

    
#%% Storing final iteration data to json file, and to metrics.circuit_metrics_final_iter
### DEVNOTE: this only applies for Qiskit, not Ocean
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
            parent_folder_save = os.path.join('__data', f'{backend_id}')
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
    ''' DEVNOTE: Not needed
    unif_cuts, unif_counts, unif_sizes, unique_counts_unif, unique_sizes_unif, cumul_counts_unif = uniform_cut_sampling(
        num_qubits, degree, num_shots, _instances)
    unif_dict = {'unique_counts_unif': unique_counts_unif,
                 'unique_sizes_unif': unique_sizes_unif,
                 'cumul_counts_unif': cumul_counts_unif}  # store only the distribution of cut sizes, and not the cuts themselves
    '''
    unif_dict = {'unique_counts_unif': [],
                 'unique_sizes_unif': [],
                 'cumul_counts_unif': []
                 }
    
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

def load_data_and_plot(folder, backend_id=None, **kwargs):
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


def load_all_metrics(folder, backend_id=None):
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
                
        # after loading data, need to convert times to deltas (compatibility with metrics plots)   
        convert_times_to_deltas(num_qubits)

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

MAX_QUBITS = 320
iter_dist = {'cuts' : [], 'counts' : [], 'sizes' : []} # (list of measured bitstrings, list of corresponding counts, list of corresponding cut sizes)
iter_size_dist = {'unique_sizes' : [], 'unique_counts' : [], 'cumul_counts' : []} # for the iteration being executed, stores the distribution for cut sizes
saved_result = {  }
instance_filename = None

minimizer_loop_index = 0

def run (min_qubits=3, max_qubits=6, max_circuits=1, num_shots=100,
        method=1, degree=3, alpha=0.1, thetas_array=None, parameterized= False, do_fidelities=True,
        max_iter=30, min_annealing_time=1, max_annealing_time=200, score_metric='fidelity', x_metric='cumulative_exec_time', y_metric='num_qubits',
        fixed_metrics={}, num_x_bins=15, y_size=None, x_size=None,
        objective_func_type = 'approx_ratio', plot_results = True,
        save_res_to_file = False, save_final_counts = False, detailed_save_names = False, comfort=False,
        backend_id='qasm_simulator', provider_backend=None, eta=0.5,
        hub="ibm-q", group="open", project="main", exec_options=None, _instances=None):
    """
    Parameters
    ----------
    min_qubits : int, optional
        The smallest circuit width for which benchmarking will be done The default is 3.
    max_qubits : int, optional
        The largest circuit width for which benchmarking will be done. The default is 6.
    max_circuits : int, optional
        Number of restarts. The default is None.
    num_shots : int, optional
        Number of times the circut will be measured, for each iteration. The default is 100.
    method : int, optional
        If 1, then do standard metrics, if 2, implement iterative algo metrics. The default is 1.
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
    min_annealing_time : int, optional
        Minimum annealing time. The default is 1.
    max_annealing_time : int, optional
        Maximum annealing time. The default is 200.
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

    # Delete some entries from the dictionary
    for key in ["hub", "group", "project", "provider_backend"]:
        dict_of_inputs.pop(key)
    
    global maxcut_inputs
    maxcut_inputs = dict_of_inputs
    
    #print(f"{dict_of_inputs = }")
    
    print("MaxCut Benchmark Program - Ocean")

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
    degree = max(3, degree)
    
    # don't compute exectation unless fidelity is is needed
    global do_compute_expectation
    do_compute_expectation = False
    
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
    ex.init_execution(execution_handler2)
    
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options)

    # Execute Benchmark Program N times for anneal times in powers of 2
    # Accumulate metrics asynchronously as circuits complete
    
    # loop over available problem sizes from min_qubits up to max_qubits
    for num_qubits in [4, 8, 12, 16, 20, 24, 40, 80, 160, 320]:
    
        if num_qubits < min_qubits:
            continue
            
        if num_qubits > max_qubits:
            break
        
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
            if method == 1:
            
                # create the circuit for given qubit size and secret string, store time metric
                ts = time.time()
                qc, params = MaxCut(num_qubits, restart_ind, edges, parameterized)   ### DEVNOTE: remove param?
                metrics.store_metric(num_qubits, restart_ind, 'create_time', time.time()-ts)

                # submit circuit for execution on target (simulator, cloud simulator, or hardware)
                ex.submit_circuit(qc, num_qubits, restart_ind, shots=num_shots, params=params)

            if method == 2:
                global minimizer_loop_index
                
                # a unique circuit index used inside the inner minimizer loop as identifier
                minimizer_loop_index = 0 # Value of 0 corresponds to the 0th iteration of the minimizer
                
                start_iters_t = time.time()
                
                # Always start by enabling embed ...
                ex.set_embedding_flag(embedding_flag=True)
                
                if verbose:
                    print(f'===============  Begin method 2 loop, enabling embed')

                annealing_time = min_annealing_time
                while annealing_time <= max_annealing_time:
                    
                    if verbose:
                        print(f"... using anneal time: {annealing_time}")

                    # Every circuit needs a unique id; add unique_circuit_index instead of s_int
                    #global minimizer_loop_index
                    unique_id = restart_ind * 1000 + minimizer_loop_index
                    #************************************************
                    #*** Circuit Creation
                
                    # create the circuit for given qubit size, secret string and params, store time metric
                    ts = time.time()
                    qc, params = MaxCut(num_qubits, unique_id, edges, parameterized)
                    params = [annealing_time]
                    metrics.store_metric(num_qubits, unique_id, 'create_time', time.time()-ts)
                        
                    # also store the 'degree' for each execution
                    # DEVNOTE: Currently, this is stored for each iteration. Reduce this redundancy
                    metrics.store_metric(num_qubits, unique_id, 'degree', degree)
                    
                    #************************************************
                    #*** Quantum Part: Execution of Circuits ***
                
                    # submit circuit for execution on target with the current parameters
                    ex.submit_circuit(qc, num_qubits, unique_id, shots=num_shots, params=params)
                        
                    # Must wait for circuit to complete
                    #ex.throttle_execution(metrics.finalize_group)
                    ex.finalize_execution(None, report_end=False)    # don't finalize group until all circuits done
                    
                    '''
                    # after first execution and thereafter, no need for embed (actually NOT)
                    #since we are benchmarking, we want to compare performance across anneal times
                    # so we do not want to use embedding, or it wouldn't be a valid comparison
                    #ex.set_embedding_flag(embedding_flag=False)
                    ex.set_embedding_flag(embedding_flag=True)
                    if verbose:
                        print(f'**** First execution complete, disabling embed')
                    '''  
                    global saved_result
                    #print(saved_result)
                    
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
                    # metrics.store_metric(num_qubits, unique_id, 'opt_exec_time', time.time() - tc1 + ts - opt_ts)
                    # Note: the first time it is stored it is just the initialization time for optimizer
                    #************************************************
                    
                    if verbose:
                        print(dict_of_vals)
                        
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
                    # reset timer for optimizer execution after each iteration of quantum program completes
                    opt_ts = time.time()
                    
                    # double the annealing time for the next iteration
                    annealing_time *= 2
            
            # DEVNOTE: Do this here, if we want to save deltas to file (which we used to do)         
            # for this benchmark, need to convert times to deltas (for compatibility with metrics)   
            #convert_times_to_deltas(num_qubits)
            
            # Save final iteration data to metrics.circuit_metrics_final_iter
            # This data includes final counts, cuts, etc.
            store_final_iter_to_metrics_json(num_qubits=num_qubits, 
                                                degree=degree, 
                                                restart_ind=restart_ind,
                                                num_shots=num_shots, 
                                                #converged_thetas_list=res.x.tolist(),
                                                converged_thetas_list=[[0],[0]],
                                                opt=opt,
                                                iter_size_dist=iter_size_dist, iter_dist=iter_dist, parent_folder_save=parent_folder_save,
                                                dict_of_inputs=dict_of_inputs, save_final_counts=save_final_counts,
                                                save_res_to_file=save_res_to_file, _instances=_instances)
                                                
            # after saving data, convert times to deltas (for compatibility with metrics plots)   
            convert_times_to_deltas(num_qubits)
                                                
        # for method 2, need to aggregate the detail metrics appropriately for each group
        # Note that this assumes that all iterations of the circuit have completed by this point
        if method == 2:                  
            metrics.process_circuit_metrics_2_level(num_qubits)
            metrics.finalize_group(str(num_qubits))
        
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)
    
    # Plot metrics for all problem sizes
    if method == 1:
        metrics.plot_metrics(f"Benchmark Results - MaxCut ({method}) - Ocean",
                options=dict(shots=num_shots))
                
    elif method == 2:
        #metrics.print_all_circuit_metrics()
        if plot_results:
            plot_results_from_data(**dict_of_inputs)

# Convert elapsed/exec/opt_exec times to deltas from absolute
# for this benchmark, need to convert the times to deltas (for compatibility with metrics)
# since there is wobble in some of the times, don't go below delta = 0
def convert_times_to_deltas(num_qubits):

    elapsed_time = exec_time = opt_exec_time = 0
    for circuit_id in metrics.circuit_metrics[str(num_qubits)]:
        circuit = metrics.circuit_metrics[str(num_qubits)][circuit_id]
        #print(f"... id = {circuit_id}, times = {circuit['elapsed_time']} {circuit['exec_time']} {circuit['opt_exec_time']}")
        
        d_elapsed_time = max(0, circuit['elapsed_time'] - elapsed_time)
        d_exec_time = max(0, circuit['exec_time'] - exec_time)
        d_opt_exec_time = max(0, circuit['opt_exec_time'] - opt_exec_time)
        
        elapsed_time = max(elapsed_time, circuit['elapsed_time'])
        exec_time = max(exec_time, circuit['exec_time'])
        opt_exec_time = max(opt_exec_time, circuit['opt_exec_time'])
        
        #print(f"  ... max times = {elapsed_time} {exec_time} {opt_exec_time}")
        #print(f"  ... delta times = {d_elapsed_time} {d_exec_time} {d_opt_exec_time}")
        
        circuit['elapsed_time'] = d_elapsed_time
        circuit['exec_time'] = d_exec_time
        circuit['opt_exec_time'] = d_opt_exec_time
        
# Method to plot the results from the collected data
def plot_results_from_data(num_shots=100, degree=3, max_iter=30, max_circuits = 1,
                 objective_func_type='approx_ratio', method=2, score_metric='fidelity',
                 x_metric='cumulative_exec_time', y_metric='num_qubits', fixed_metrics={},
                 num_x_bins=15, y_size=None, x_size=None, x_min=None, x_max=None,
                 offset_flag=True,            # default is True for QA
                 detailed_save_names=False, **kwargs):
    """
    Plot results
    """

    if detailed_save_names:
        # If detailed names are desired for saving plots, put date of creation, etc.
        cur_time=datetime.datetime.now()
        dt = cur_time.strftime("%Y-%m-%d_%H-%M-%S")
        short_obj_func_str = metrics.score_label_save_str[objective_func_type]
        suffix = f'-s{num_shots}_d{degree}_mi{max_iter}_of-{short_obj_func_str}_{dt}' #of=objective function
    else:
        short_obj_func_str = metrics.score_label_save_str[objective_func_type]
        suffix = f'of-{short_obj_func_str}' #of=objective function
        
    obj_str = metrics.known_score_labels[objective_func_type]
    options = {'shots' : num_shots, 'degree' : degree, 'restarts' : max_circuits, '\nObjective Function' : obj_str}
    suptitle = f"Benchmark Results - MaxCut ({method}) - Ocean"
    
    metrics.plot_all_area_metrics(f"Benchmark Results - MaxCut ({method}) - Ocean",
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
    
    # not needed for Ocean version
    #metrics.plot_angles_polar(suptitle = suptitle, options = options, suffix = suffix)

# if main, execute method
if __name__ == '__main__': run()

# %%
