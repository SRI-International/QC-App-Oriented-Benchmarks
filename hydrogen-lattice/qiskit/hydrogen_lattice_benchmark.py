
"""
Hydogen Lattice Benchmark Program - Qiskit
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
from typing import Any, List, Optional
from pathlib import Path
from qiskit.opflow.primitive_ops import PauliSumOp
import matplotlib.pyplot as plt
import glob

import numpy as np
from scipy.optimize import minimize

from qiskit import (Aer, ClassicalRegister,  # for computing expectation tables
                    QuantumCircuit, QuantumRegister, execute, transpile)
from qiskit.circuit import ParameterVector
from typing import Dict, List, Optional
from qiskit import Aer, execute
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.opflow import ComposedOp, PauliExpectation, StateFn, SummedOp
from qiskit.quantum_info import Statevector,Pauli
from qiskit.result import sampled_expectation_value

sys.path[1:1] = [ "_common", "_common/qiskit", "hydrogen-lattice/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit", "../../hydrogen-lattice/_common/" ]

import common
import execute as ex
import metrics as metrics
from matplotlib import cm

# import h-lattice_metrics from _common folder
import h_lattice_metrics as h_metrics


logger = logging.getLogger(__name__)
fname, _, ext = os.path.basename(__file__).partition(".")
log_to_file = True

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

# Hydrogen Lattice inputs  ( Here input is Hamiltonian matrix --- Need to change)
hl_inputs = dict() #inputs to the run method
verbose = False
print_sample_circuit = True
# Indicates whether to perform the (expensive) pre compute of expectations
do_compute_expectation = True


# saved circuits for display
QC_ = None
Uf_ = None

# #theta parameters
vqe_parameter = namedtuple('vqe_parameter','theta')
# QAOA_Parameter  = namedtuple('QAOA_Parameter', ['beta', 'gamma'])

# Qiskit uses the little-Endian convention. Hence, measured bit-strings need to be reversed while evaluating cut sizes
reverseStep = -1

# DEBUG prints
# give argument to the python script as "debug" or "true" or "1" to enable debug prints
if len(sys.argv) > 1:
    DEBUG = (sys.argv[1].lower() in ['debug', 'true', '1'])
else:
    DEBUG = False


# Create the  ansatz quantum circuit for the VQE algorithm.
def VQE_ansatz(num_qubits: int, thetas_array, num_occ_pairs: Optional[int] = None, *args, **kwargs) -> QuantumCircuit:
    # Generate the ansatz circuit for the VQE algorithm.
    if num_occ_pairs is None:
        num_occ_pairs = (num_qubits // 2)  # e.g., half-filling, which is a reasonable chemical case

        # do all possible excitations if not passed a list of excitations directly
    excitation_pairs = []
    for i in range(num_occ_pairs):
        for a in range(num_occ_pairs, num_qubits):
            excitation_pairs.append([i, a])

    circuit = QuantumCircuit(num_qubits)
    
    # Hartree Fock initial state
    for occ in range(num_occ_pairs):
        circuit.x(occ)
    
    # if thetas_array is not None:
    #     parameter_vector = ParameterVector(thetas_array)
    # else:
    parameter_vector = ParameterVector("t", length=len(excitation_pairs))
    
    thetas_array = np.repeat(thetas_array, len(excitation_pairs))
    # Hartree Fock initial state


    for idx, pair in enumerate(excitation_pairs):
        # parameter
        
        theta = parameter_vector[idx]            
        # apply excitation
        i, a = pair[0], pair[1]

        # implement the magic gate
        circuit.s(i)
        circuit.s(a)
        circuit.h(a)
        circuit.cx(a, i)

        # Ry rotation
        circuit.ry(theta, i)
        circuit.ry(theta, a)

        # implement M^-1
        circuit.cx(a, i)
        circuit.h(a)
        circuit.sdg(a)
        circuit.sdg(i)

    return circuit,parameter_vector,thetas_array
    


# Create the benchmark program circuit
# Accepts optional rounds and array of thetas (betas and gammas)
def HydrogenLattice (num_qubits, operator, secret_int = 000000, thetas_array = None, parameterized = None):
    # if no thetas_array passed in, create defaults 
    
    # here we are filling this th
    if thetas_array is None:
        thetas_array = [1.0]
    
    
    #print(f"... actual thetas_array={thetas_array}")
    
    # create parameters in the form expected by the ansatz generator
    # this is an array of betas followed by array of gammas, each of length = rounds
    global _qc
    global theta
    # global gammas
    
    # create the circuit the first time, add measurements
    if ex.do_transpile_for_execute:
        logger.info(f'*** Constructing parameterized circuit for {num_qubits = } {secret_int}')
        # theta = ParameterVector("θ", 1)
        
        # Here we are doing the equivalent code inside VQE_ansatz
        # betas = ParameterVector("𝞫", p)
        # gammas = ParameterVector("𝞬", p)
        # params = {betas: thetas_array[:p], gammas: thetas_array[p:]}   

        _qc ,parameter_vector,thetas_array = VQE_ansatz(num_qubits=num_qubits, thetas_array=thetas_array, num_occ_pairs = None )

    _measurable_expression = StateFn(operator, is_measurement=True)
    _observables = PauliExpectation().convert(_measurable_expression)
    _qc_array, _formatted_observables = prepare_circuits(_qc, observables=_observables)

    # Here Parametee
    params = {parameter_vector: thetas_array}
        
    # if thetas_array is None and len(_qc.parameters) > 0:
    #     _qc.assign_parameters([np.random.choice([-1e-3, 1e-3]) for _ in range(len(_qc.parameters))],inplace=True)
    
    # add the measure here, only after circuit is created
    #_qc.measure_all()
    
    # params = {theta: thetas_array[1]}   
    #logger.info(f"Binding parameters {params = }")
    logger.info(f"Create binding parameters for {thetas_array}")
    
    qc = _qc
    #print(qc)
    
    # pre-compute and save an array of expected measurements
    #Imp Note This is different to  compute_expectation in simulator.py and it is renamed as calculate_expectation
    # This is not important as of now
    if do_compute_expectation:
        logger.info('Computing expectation')
        compute_expectation(qc, num_qubits, secret_int, params=params)
   
    # save small circuit example for display
    global QC_
    if QC_ == None or num_qubits <= 6:
        if num_qubits < 9: QC_ = qc

    # return a handle on the circuit
    return _qc_array, _formatted_observables, params


# ############### Circuit Definition - Parameterized version   
    
    
# ############### Expectation Tables

# DEVNOTE: We are building these tables on-demand for now, but for larger circuits
# this will need to be pre-computed ahead of time and stored in a data file to avoid run-time delays.

# dictionary used to store pre-computed expectations, keyed by num_qubits and secret_string
# these are created at the time the circuit is created, then deleted when results are processed
expectations = {}

# Compute array of expectation values in range 0.0 to 1.0
# Use statevector_simulator to obtain exact expectation
def compute_expectation(qc, num_qubits, secret_int, backend_id='statevector_simulator', params=None):
    
    pass  # For now
    # #ts = time.time()
    # if params != None:
    #     qc = qc.bind_parameters(params)
    
    # #execute statevector simulation
    # sv_backend = Aer.get_backend(backend_id)
    # sv_result = execute(qc, sv_backend).result()

    # # get the probability distribution
    # counts = sv_result.get_counts()

    # #print(f"... statevector expectation = {counts}")
    
    # # store in table until circuit execution is complete
    # id = f"_{num_qubits}_{secret_int}"
    # expectations[id] = counts

    #print(f"  ... time to execute statevector simulator: {time.time() - ts}")
# Return expected measurement array scaled to number of shots executed
def get_expectation(num_qubits,  num_shots):
    pass
    # find expectation counts for the given circuit 
    # id = f"_{num_qubits}"
    # if id in expectations:
    #     counts = expectations[id]
        
    #     # scale to number of shots
    #     for k in counts.items():
    #         counts[k] = num_shots
        
    #     # delete from the dictionary
    #     del expectations[id]
        
    #     return counts
        
    # else:
    #     return None
    
    
# ############### Result Data Analysis

# expected_dist = {}


# Note :- Not related to initial execution
# Compare the measurement results obtained with the expected measurements to determine fidelity
def analyze_and_print_result (qc, result, num_qubits, secret_int, num_shots):
    global expected_dist
    
    # obtain counts from the result object
    counts = result.get_counts(qc)
    
    # retrieve pre-computed expectation values for the circuit that just completed
    expected_dist = get_expectation(num_qubits,  num_shots)
    
    # if the expectation is not being calculated (only need if we want to compute fidelity)
    # assume that the expectation is the same as measured counts, yielding fidelity = 1
    if expected_dist == None:
        expected_dist = counts
    
    if verbose: print(f"For width {num_qubits}   measured: {counts}\n  expected: {expected_dist}")
    # if verbose: print(f"For width {num_qubits} problem {secret_int}\n  measured: {counts}\n  expected: {expected_dist}")


    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, expected_dist)

    # if verbose: print(f"For secret int {secret_int} fidelity: {fidelity}")    
    
    return counts, fidelity


# Not needed for initial execution
def get_random_angles(num_occ_pairs):
    """Create max_circuit number of random initial conditions"""
    
    thetas = []
    for i in range(len(num_occ_pairs)):
        thetas[i] = np.random.choice([-1e-3, 1e-3])
    return thetas


# For initial development
statevector_backend = Aer.get_backend("statevector_simulator")
qasm_backend = Aer.get_backend("qasm_simulator")

#----------------- start of simulator expectation value code-----------------------------------

def get_measured_qubits(circuit: QuantumCircuit) -> List[int]:
    """
    Get a list of indices of the qubits being measured in a given quantum circuit.
    """
    measured_qubits = []

    for gate, qubits, clbits in circuit.data:
        if gate.name == "measure":
            measured_qubits.extend([qubit.index for qubit in qubits])

    measured_qubits = sorted(list(set(measured_qubits)))

    return measured_qubits

def expectation_run(circuit: QuantumCircuit, shots: Optional[int] = None) -> Dict[str, float]:
        """Run a quantum circuit on the noise-free simulator and return the probabilities."""

        # Refactored error check
        # if circuit.num_parameters != 0:
            # raise QiskitError(ErrorMessages.UNDEFINED_PARAMS.value)
        if len(get_measured_qubits(circuit)) == 0:
            circuit.measure_all()

        if shots is None:
            measured_qubits = get_measured_qubits(circuit)
            statevector = (
                execute(
                    circuit.remove_final_measurements(inplace=False),
                    backend= statevector_backend,
                )
                .result()
                .get_statevector()
            )
            probs = Statevector(statevector).probabilities_dict(qargs=measured_qubits)
        elif isinstance(shots, int):
            counts = (execute(circuit, backend= qasm_backend, shots=shots).result().get_counts())
            if DEBUG:
                print("DEBUG : \n method: expectation_run \n\t counts: "+str(counts) + "\n\t circuit: "+str(circuit))
            probs = normalize_counts(counts, num_qubits=circuit.num_qubits)
        # else:
        #     raise TypeError(ErrorMessages.UNRECOGNIZED_SHOTS.value.format(shots=shots))

        return probs

def normalize_counts(counts, num_qubits=None):
    """
    Normalize the counts to get probabilities and convert to bitstrings.
    """
    normalizer = sum(counts.values())

    try:
        dict({str(int(key, 2)): value for key, value in counts.items()})
        if num_qubits is None:
            num_qubits = max(len(key) for key in counts)
        bitstrings = {key.zfill(num_qubits): value for key, value in counts.items()}
    except ValueError:
        bitstrings = counts

    probabilities = dict({key: value / normalizer for key, value in bitstrings.items()})
    assert abs(sum(probabilities.values()) - 1) < 1e-9
    return probabilities

def calculate_expectation(base_circuit, operator, parameters=None, shots=None):
    """
    Compute the expected value of the operator given a base-circuit and pauli operator.

    No checks are made to ensure consistency between operator and base circuit.

    Parameters
    ----------
    base_circuit : :obj:`QuantumCircuit`
        Base circuit in computational basis. Basis rotation gates will be appended as needed given the operator.
    operator : :obj:`PauliSumOp`
        Operator expressed as a sum of Pauli operators. This is assumed to be consistent with the base circuit.
    parameters : :obj:`Optional[Union[List, ndarray]]`
        Optional parameters to pass in if the circuit is parameterized
    """
    # if parameters is not None and base_circuit.num_parameters != len(parameters):
    #     raise ValueError(f"Circuit has {base_circuit.num_parameters} but parameter length is {len(parameters)}.")

    measurable_expression = StateFn(operator, is_measurement=True)
    observables = PauliExpectation().convert(measurable_expression)
    circuits, formatted_observables = prepare_circuits(base_circuit, observables)

    if DEBUG:
        print("DEBUG : \n method: calculate_expectation \n\t base_circuit: "+str(base_circuit))

    probabilities = compute_probabilities(circuits, parameters, shots)
    expectation_values = calculate_expectation_values(probabilities, formatted_observables)
    if DEBUG:
        print("DEBUG : \n method: calculate_expectation \n\t probabilities: "+str(probabilities) + "\n\t expectation_values: "+str(expectation_values))
    return sum(expectation_values)
    
def prepare_circuits(base_circuit, observables):
    """
    Prepare the qubit-wise commuting circuits for a given operator.
    """
    circuits = list()

    if isinstance(observables, ComposedOp):
        observables = SummedOp([observables])
    for obs in observables:
        circuit = base_circuit.copy()
        circuit.append(obs[1], qargs=list(range(base_circuit.num_qubits)))
        circuit.measure_all()
        circuits.append(circuit)
    return circuits, observables

def compute_probabilities(circuits, parameters=None, shots=None):
    """
    Compute the probabilities for a list of circuits with given parameters.
    """
    probabilities = list()
    for my_circuit in circuits:
        if parameters is not None:
            circuit = my_circuit.assign_parameters(parameters, inplace=False)
        else:
            circuit = my_circuit.copy()
        result = expectation_run(circuit, shots)
        if DEBUG:
            print("DEBUG : \n method: compute_probabilities \n\t result_probabilities: "+str(result))
        probabilities.append(result)

    return probabilities

def calculate_expectation_values(probabilities, observables):
    """
    Return the expectation values for an operator given the probabilities.
    """
    expectation_values = list()
    for idx, op in enumerate(observables):
        expectation_value = sampled_expectation_value(probabilities[idx], op[0].primitive)
        expectation_values.append(expectation_value)

    return expectation_values

# -------------------------------------end of simulator expectation value code-----------------------------------

# ------------------Main objective Function to calculate expectation value------------------
# objective Function to compute the energy of a circuit with given parameters and operator
# # Initialize an empty list to store the lowest energy values
lowest_energy_values = []

def compute_energy(result_array, formatted_observables, num_qubits): 
    
    
    # Compute the expectation value of the circuit with respect to the Hamiltonian for optimization

    _probabilities = list()

    for _res in result_array:
        _counts = _res.get_counts()
        _probs = normalize_counts(_counts, num_qubits=num_qubits)
        _probabilities.append(_probs)


    _expectation_values = calculate_expectation_values(_probabilities, formatted_observables)

    energy = sum(_expectation_values)

    # Append the energy value to the list
    lowest_energy_values.append(energy)

    
    return energy    


# Function to save final iteration data to file
def store_final_iter_to_metrics_json(num_qubits,
                                     radius,
                                     instance_num,
                                     num_shots,
                                     converged_thetas_list,
                                     energy,
                                     #iter_size_dist,
                                     #iter_dist,
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
    '''
    unif_cuts, unif_counts, unif_sizes, unique_counts_unif, unique_sizes_unif, cumul_counts_unif = uniform_cut_sampling(
        num_qubits, degree, num_shots, _instances)
    unif_dict = {'unique_counts_unif': unique_counts_unif,
                 'unique_sizes_unif': unique_sizes_unif,
                 'cumul_counts_unif': cumul_counts_unif}  # store only the distribution of cut sizes, and not the cuts themselves
    '''
    # Store properties such as (cuts, counts, sizes) of the final iteration, the converged theta values, as well as the known optimal value for the current problem, in metrics.circuit_metrics_final_iter. Also store uniform cut sampling results
    metrics.store_props_final_iter(num_qubits, instance_num, 'energy', energy)
    #metrics.store_props_final_iter(num_qubits, instance_num, None, iter_size_dist)
    metrics.store_props_final_iter(num_qubits, instance_num, 'converged_thetas_list', converged_thetas_list)
    #metrics.store_props_final_iter(num_qubits, instance_num, None, unif_dict)
    # metrics.store_props_final_iter(num_qubits, instance_num, None, iter_dist) # do not store iter_dist, since it takes a lot of memory for larger widths, instead, store just iter_size_dist
    #global radius
    if save_res_to_file:
        # Save data to a json file
        dump_to_json(parent_folder_save, num_qubits,
                     radius, instance_num, 
                     #iter_size_dist, 
                     #iter_dist, 
                     dict_of_inputs, 
                     converged_thetas_list, 
                     energy,
                     #unif_dict,
                     save_final_counts=save_final_counts)

def dump_to_json(parent_folder_save, num_qubits, radius, instance_num, 
                 #iter_size_dist, iter_dist,
                 dict_of_inputs, 
                 converged_thetas_list, energy,
                 #unif_dict,
                 save_final_counts=False):
    """
    For a given problem (specified by number of qubits and graph degree) and instance_numex, 
    save the evolution of various properties in a json file.
    Items stored in the json file: Data from all iterations (iterations), inputs to run program ('general properties'), converged theta values ('converged_thetas_list'), max cut size for the graph (optimal_value), distribution of cut sizes for random uniform sampling (unif_dict), and distribution of cut sizes for the final iteration (final_size_dist)
    if save_final_counts is True, then also store the distribution of cuts 
    """
    
    print(f"... saving data for width {num_qubits} radius {radius} instance_num {instance_num}")
    
    if not os.path.exists(parent_folder_save): os.makedirs(parent_folder_save)
    store_loc = os.path.join(parent_folder_save,'width_{}_instance_{}.json'.format(num_qubits, instance_num))

    print(f"  ... to file {store_loc}")
    
    # Obtain dictionary with iterations data corresponding to given instance_num 
    all_restart_ids = list(metrics.circuit_metrics[str(num_qubits)].keys())
    ids_this_restart = [r_id for r_id in all_restart_ids if int(r_id) // 1000 == instance_num]
    iterations_dict_this_restart =  {r_id : metrics.circuit_metrics[str(num_qubits)][r_id] for r_id in ids_this_restart}
    # Values to be stored in json file
    dict_to_store = {'iterations' : iterations_dict_this_restart}
    dict_to_store['general_properties'] = dict_of_inputs
    dict_to_store['converged_thetas_list'] = converged_thetas_list
    dict_to_store['energy'] = energy
    #dict_to_store['unif_dict'] = unif_dict
    #dict_to_store['final_size_dist'] = iter_size_dist
    # Also store the value of counts obtained for the final counts
    '''
    if save_final_counts:
        dict_to_store['final_counts'] = iter_dist
    '''                                   #iter_dist.get_counts()
    # Now save the output
    with open(store_loc, 'w') as outfile:
        json.dump(dict_to_store, outfile)
        
# %% Loading saved data (from json files)

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
    print(list_of_files)
    
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


# # ----------------need to revise the below code------------------

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
        #unif_dict = data['unif_dict']
        energy = data['energy']
        if gen_prop['save_final_counts']:
            # Distribution of measured cuts
            final_counts = data['final_counts']
        #final_size_dist = data['final_size_dist']
        
        backend_id = gen_prop.get('backend_id')
        metrics.set_plot_subtitle(f"Device = {backend_id}")
        
        # Update circuit metrics
        for circuit_id in data['iterations']:
            # circuit_id = restart_ind * 1000 + minimizer_loop_ind
            for metric, value in data['iterations'][circuit_id].items():
                metrics.store_metric(num_qubits, circuit_id, metric, value)
                
        method = gen_prop['method']
        if method == 2:
            metrics.store_props_final_iter(num_qubits, restart_ind, 'energy', energy)
            #metrics.store_props_final_iter(num_qubits, restart_ind, None, final_size_dist)
            metrics.store_props_final_iter(num_qubits, restart_ind, 'converged_thetas_list', converged_thetas_list)
            #metrics.store_props_final_iter(num_qubits, restart_ind, None, unif_dict)
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
    pattern = 'width_([0-9]+)_instance_([0-9]+).json'
    match = re.search(pattern, fileName)
    print(match)

    # assert match is not None, f"File {fileName} found inside folder. All files inside specified folder must be named in the format 'width_int_restartInd_int.json"
    num_qubits = int(match.groups()[0])
    degree = int(match.groups()[1])
    return (num_qubits,degree)

# # ----------------need to revise the above code------------------


################ Run Method

MAX_QUBITS = 24
# # iter_dist = {'cuts' : [], 'counts' : [], 'sizes' : []} # (list of measured bitstrings, list of corresponding counts, list of corresponding cut sizes)
# # iter_size_dist = {'unique_sizes' : [], 'unique_counts' : [], 'cumul_counts' : []} # for the iteration being executed, stores the distribution for cut sizes
# saved_result = {  }
# instance_filename = None

#radius = None

def run (min_qubits=2, max_qubits=4, max_circuits=3, num_shots=100,
        method=2, radius=None, thetas_array=None, parameterized= False, do_fidelities=True,
        max_iter=30, score_metric=['solution_quality', 'accuracy_volume'], x_metric=['cumulative_exec_time','cumulative_elapsed_time'], y_metric='num_qubits',
        fixed_metrics={}, num_x_bins=15, y_size=None, x_size=None, plot_results = True,
        save_res_to_file = False, save_final_counts = False, detailed_save_names = False, comfort=False,
        backend_id='qasm_simulator', provider_backend=None, eta=0.5,
        hub="ibm-q", group="open", project="main", exec_options=None, _instances=None) :
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
    thetas_array : list, optional
        list or ndarray of theta values. The default is None.
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
    
    thetas = []   #Need to change
    # # Get angles for restarts. Thetas = list of lists. Lengths are max_circuits and 2*rounds
    # thetas, max_circuits = get_restart_angles(thetas_array, rounds, max_circuits)
    
    # Update the dictionary of inputs
    dict_of_inputs = {**dict_of_inputs, **{'thetas_array': thetas, 'max_circuits' : max_circuits}}
    
    # Delete some entries from the dictionary
    for key in ["hub", "group", "project", "provider_backend"]:
        dict_of_inputs.pop(key)
    
    global hydrogen_lattice_inputs
    hydrogen_lattice_inputs = dict_of_inputs
    
    global QC_
    global circuits_done
    global minimizer_loop_index
    global opt_ts
    
    print("Hydrogen Lattice Benchmark Program - Qiskit")

    QC_ = None
    
    # Create a folder where the results will be saved. Folder name=time of start of computation
    # In particular, for every circuit width, the metrics will be stored the moment the results are obtained
    # In addition to the metrics, the (beta,gamma) values obtained by the optimizer, as well as the counts
    # measured for the final circuit will be stored.
    # Use the following parent folder, for a more detailed 
    if detailed_save_names:
        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        parent_folder_save = os.path.join('__results', f'{backend_id}', f'run_start_{start_time_str}')
    else:
        parent_folder_save = os.path.join('__results', f'{backend_id}')
    
    if save_res_to_file and not os.path.exists(parent_folder_save): os.makedirs(os.path.join(parent_folder_save))
    
    # validate parameters
    # max_qubits = max(2, max_qubits)
    # max_qubits = min(MAX_QUBITS, max_qubits)
    # min_qubits = min(max(2, min_qubits), max_qubits)
    # radius = max(0.75, radius)
    
    # don't compute exectation unless fidelity is is needed
    global do_compute_expectation
    do_compute_expectation = do_fidelities
        
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
        metrics.store_metric(num_qubits, s_int, 'solution_quality', fidelity)

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
    #ex.set_noise_model(None)
    
# -------------classical Pauli sum operator from list of Pauli operators and coefficients----
    # Below function is to reduce some dependency on qiskit ( String data type issue)-------------
    # def pauli_sum_op(ops, coefs):
    #     if len(ops) != len(coefs):
    #         raise ValueError("The number of Pauli operators and coefficients must be equal.")
    #     pauli_sum_op_list = [(op, coef) for op, coef in zip(ops, coefs)]
    #     return pauli_sum_op_list
#-------------classical Pauli sum operator from list of Pauli operators and coefficients-----------------

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    # DEVNOTE: increment by 2 for paired electron circuits
    global instance_filepath 
    for num_qubits in range(min_qubits, max_qubits + 1, 2):
        
        if method == 1:
            print(f"************\nExecuting [{max_circuits}] circuits for num_qubits = {num_qubits}")
        else:
            print(f"************\nExecuting [{max_circuits}] restarts for num_qubits = {num_qubits}")
        
        # # If radius is negative, 
        # if radius < 0 :
        #     radius = max(3, (num_qubits + radius))
            
        # looping all instance files according to max_circuits given        
        for instance_num in range(1, max_circuits + 1):
            # Index should start from 1
            
            # if radius is given we should do same radius for max_circuits times
            if radius is not None:
                instance_filepath = os.path.join(os.path.dirname(__file__),"..", "_common",
                        common.INSTANCE_DIR, f"h{num_qubits:03}_chain_{radius:06.2f}.json")   
            # if radius is not given we should do all the radius for max_circuits times
            else:
               instance_filepath_list = [file \
               for file in glob.glob(os.path.join(os.path.dirname(__file__), "..", "_common", \
               common.INSTANCE_DIR, f"h{num_qubits:03}_*_*_*.json"))]
               print(instance_filepath_list)
               if len(instance_filepath_list) >= instance_num :
                   instance_filepath = instance_filepath_list[instance_num-1]
               else:
                   print("problem not found")

            # operator is paired hamiltonian  for each instance in the loop  
            ops,coefs = common.read_paired_instance(instance_filepath)
            # operator = Paulisumop(list(zip(ops, coefs)))
            operator = PauliSumOp.from_list(list(zip(ops, coefs)))
            # solution has list of classical solutions for each instance in the loop
            sol_file_name = instance_filepath[:-5] + ".sol"
            method_names, values = common.read_puccd_solution(sol_file_name)
            solution = list(zip(method_names, values))   
            
                    # if the file does not exist, we are done with this number of qubits
            if operator == None:
                print(f"  ... problem not found.")
                break
                
            # get thetas for each instance in the loop
            # thetas_array = thetas[instance_num - 1]
            # thetas_array = np.random.random(size=1) #len(circuit.parameters))

            if method == 1:
                # create the circuit for given qubit size and secret string, store time metric
                ts = time.time()
            # DEVNOTE:  Primary focus is on method 2
                thetas_array_0 = thetas_array
                qc_array, frmt_obs, params = HydrogenLattice(num_qubits = num_qubits,operator = operator, thetas_array= thetas_array_0, parameterized= parameterized)
                for qc in qc_array:
                    metrics.store_metric(num_qubits, instance_num, 'create_time', time.time()-ts)
                    # collapse the sub-circuit levels used in this benchmark (for qiskit)
                    qc.bind_parameters(params)
                    qc2 = qc.decompose()

                    # submit circuit for execution on target (simulator, cloud simulator, or hardware)
                    ex.submit_circuit(qc2, num_qubits, instance_num, shots=num_shots, params=params)

            if method == 2:
                # a unique circuit index used inside the inner minimizer loop as identifier
                minimizer_loop_index = 0 # Value of 0 corresponds to the 0th iteration of the minimizer
                start_iters_t = time.time()

                # Always start by enabling transpile ...
                ex.set_tranpilation_flags(do_transpile_metrics=True, do_transpile_for_execute=True)
                    
                logger.info(f'===============  Begin method 2 loop, enabling transpile')
                

                # def expectation(thetas_array):
                    
                #     # Every circuit needs a unique id; add unique_circuit_index instead of s_int
                #     global minimizer_loop_index
                #     unique_id = instance_num * 1000 + minimizer_loop_index
                #     # store thetas_array
                #     metrics.store_metric(num_qubits, unique_id, 'thetas_array', thetas_array.tolist())
                    
                #     #************************************************
                #     #*** Circuit Creation and Decomposition start ***
                #     # create the circuit for given qubit size, secret string and params, store time metric
                #     ts = time.time()
                #     qc, params = HydrogenLattice(num_qubits, unique_id, thetas_array, parameterized)
                #     metrics.store_metric(num_qubits, unique_id, 'create_time', time.time()-ts)
                    
                #     # also store the 'rounds' and 'degree' for each execution
                #     # DEVNOTE: Currently, this is stored for each iteration. Reduce this redundancy
                #     # metrics.store_metric(num_qubits, unique_id, 'rounds', rounds)
                #     metrics.store_metric(num_qubits, unique_id, 'radius', radius)
                    
                #     # collapse the sub-circuit levels used in this benchmark (for qiskit)
                #     qc2 = qc.decompose()
                    
                #     # Circuit Creation and Decomposition end
                #     #************************************************
                    
                #     #************************************************
                #     #*** Quantum Part: Execution of Circuits ***
                #     # submit circuit for execution on target with the current parameters
                #     ex.submit_circuit(qc2, num_qubits, unique_id, shots=num_shots, params=params)
                    
                #     # Must wait for circuit to complete
                #     #ex.throttle_execution(metrics.finalize_group)
                #     ex.finalize_execution(None, report_end=False)    # don't finalize group until all circuits done
                    
                #     # after first execution and thereafter, no need for transpilation if parameterized
                #     if parameterized:
                #         ex.set_tranpilation_flags(do_transpile_metrics=False, do_transpile_for_execute=False)
                #         logger.info(f'**** First execution complete, disabling transpile')
                #     #************************************************
                    
                #     global saved_result
                #     # # Fidelity Calculation and Storage
                #     # _, fidelity = analyze_and_print_result(qc, saved_result, num_qubits, unique_id, num_shots) 
                #     # metrics.store_metric(num_qubits, unique_id, 'fidelity', fidelity)
                    
                #     #************************************************
                #     #*** Classical Processing of Results - essential to optimizer ***
                #     global opt_ts
                #     dict_of_vals = dict()
                #     # Start counting classical optimizer time here again
                #     tc1 = time.time()
                #     energy = compute_energy(qc2, operator,num_shots,parameters=params)
                #     # Compute the value corresponding to the objective function first
                #     # dict_of_vals[objective_func_type] = function_mapper[objective_func_type](counts, sizes, alpha = alpha)
                #     # Store the optimizer time as current time- tc1 + ts - opt_ts, since the time between tc1 and ts is not time used by the classical optimizer.
                #     metrics.store_metric(num_qubits, unique_id, 'opt_exec_time', time.time() - tc1 + ts - opt_ts)
                #     # Note: the first time it is stored it is just the initialization time for optimizer
                    
                #     return energy
                
                # # # objective Function to compute the energy of a circuit with given parameters and operator
                # def compute_energy_temp(qc, operator, shots, parameters): 
                    
                #     # Bind the parameters to the circuit
                #     bound_circuit = qc.bind_parameters(parameters)
                    
                #     # Compute the expectation value of the circuit with respect to the Hamiltonian for optimization
                #     energy = calculate_expectation(bound_circuit, operator=operator, shots=shots)
                    
                #     # Append the energy value to the list
                #     lowest_energy_values.append(energy)
                    
                #     return energy
                
                # # after first execution and thereafter, no need for transpilation if parameterized
                # # DEVNOTE: this appears to NOT be needed, as we can turn these off after 
                # def callback(xk):
                #     if parameterized:
                #         ex.set_tranpilation_flags(do_transpile_metrics=False, do_transpile_for_execute=False)

                # opt_ts = time.time()
                # # perform the complete algorithm; minimizer invokes 'expectation' function iteratively
                # ##res = minimize(expectation, thetas_array, method='COBYLA', options = { 'maxiter': max_iter}, callback=callback)

                # # res = minimize(expectation, thetas_array, method='COBYLA', options = { 'maxiter': max_iter})
                # # To-do: Set bounds for the minimizer
                
 #------------------------ end of old objective function ------------------------#               
                
                doci_energy = float(next(value for key, value in solution if key == 'doci_energy'))
                fci_energy = float(next(value for key, value in solution if key == 'fci_energy'))
                cumlative_iter_time = [0]

                current_radius = float(os.path.basename(instance_filepath).split('_')[2])
                current_radius += float(os.path.basename(instance_filepath).split('_')[3][:2])*.01

                def objective_function(thetas_array):
                    
                    # Every circuit needs a unique id; add unique_circuit_index instead of s_int
                    global minimizer_loop_index
                    unique_id = instance_num * 1000 + minimizer_loop_index
                    res = []
                    quantum_execution_time = 0.0
                    quantum_elapsed_time = 0.0


                    ts = time.time()
                    qc_array, frmt_obs, params = HydrogenLattice(num_qubits=num_qubits, secret_int=unique_id, thetas_array= thetas_array, parameterized= parameterized, operator=operator)
                    metrics.store_metric(num_qubits, unique_id, 'create_time', time.time()-ts)
                    if DEBUG:
                        print("create time:" + str(time.time() -ts))
                    #print("store metrics" +str(metrics.circuit_metrics[str(method)]['1000']))
                    # collapse the sub-circuit levels used in this benchmark (for qiskit)
                    for qc in qc_array:
                        
                        if DEBUG:
                            print("DEBUG : \n method: compute_energy \n\t binding parameters: "+str(params))

                        qc.bind_parameters(params)
                        qc2 = qc.decompose()

                    
                        # Circuit Creation and Decomposition end
                        #************************************************
                        
                        #************************************************
                        #*** Quantum Part: Execution of Circuits ***
                        # submit circuit for execution on target with the current parameters
                        
                        ex.submit_circuit(qc2, num_qubits, unique_id, shots=num_shots, params=params)
                        if DEBUG:
                            print("submit circuit id" + str(unique_id) )
    
                        
                        # Must wait for circuit to complete
                        #ex.throttle_execution(metrics.finalize_group)
    
                        # finalize execution of group of circuits
                        ex.finalize_execution(None, report_end=False)    # don't finalize group until all circuits done
                        
                        # after first execution and thereafter, no need for transpilation if parameterized
                        if parameterized:
                            ex.set_tranpilation_flags(do_transpile_metrics=False, do_transpile_for_execute=False)
                            logger.info(f'**** First execution complete, disabling transpile')
                        #************************************************
                        
                        global saved_result
                        res.append(saved_result)
                        if DEBUG:
                            print("saved result: "+ str(saved_result))

                        # tapping into circuit metric exect time:
                        if DEBUG:
                            print("circuit metrics method: " + str(num_qubits) + " id: " + str(unique_id) )
                        quantum_execution_time = quantum_execution_time + metrics.circuit_metrics[str(num_qubits)][str(unique_id)]['exec_time']
                        quantum_elapsed_time = quantum_elapsed_time + metrics.circuit_metrics[str(num_qubits)][str(unique_id)]['elapsed_time']
                        
                        # Fidelity Calculation and Storage
                        # _, fidelity = analyze_and_print_result(qc, saved_result, num_qubits, unique_id, num_shots) 
                        
                        #************************************************
                        #*** Classical Processing of Results - essential to optimizer ***
                    global opt_ts
                    if DEBUG:
                        print("iteration time :" +str(quantum_execution_time))
                    global cumulative_iter_time
                    cumlative_iter_time.append(cumlative_iter_time[-1] + quantum_execution_time)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'exec_time', quantum_execution_time)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'elapsed_time', quantum_elapsed_time)
                    dict_of_vals = dict()
                    # Start counting classical optimizer time here again
                    tc1 = time.time()

                    # increment the minimizer loop index, the index is increased by one for three circuits created ( three measurement basis circuits)
                    minimizer_loop_index += 1
                    
                    if comfort:
                        if minimizer_loop_index == 1: print("")
                        print(".", end ="")

                    energy = compute_energy(result_array = res, formatted_observables = frmt_obs, num_qubits=num_qubits)


                    # calculate the solution quality
                    solution_quality, accuracy_volume = calculate_quality_metric(energy, fci_energy, precision=0.5)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'energy', energy)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'fci_energy', fci_energy)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'solution_quality', solution_quality)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'accuracy_volume', accuracy_volume)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'fci_energy', fci_energy)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'doci_energy', doci_energy)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'radius', current_radius)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'iteration_count', minimizer_loop_index)
                    

                    return energy
                
                           
                def callback_thetas_array(thetas_array):
                    if DEBUG:
                        print("callback_thetas_array" + str(thetas_array))
                    else:
                        pass
                
                if comfort:
                    print("")
                    
                initial_parameters = np.random.random(size=1)     
                # objective_function(thetas_array=None)       
                if DEBUG:
                    print("The initial parameters are : "+ str(initial_parameters))

                thetas_array = minimize(objective_function,
                        x0=initial_parameters.ravel(),
                        method='COBYLA',
                        tol=1e-3,
                        options={'maxiter': max_iter, 'disp': False},
                        callback=callback_thetas_array) 
                
                ideal_energy = objective_function(thetas_array.x)

                current_radius = float(os.path.basename(instance_filepath).split('_')[2])
                current_radius += float(os.path.basename(instance_filepath).split('_')[3][:2])*.01

                print(f"\nBelow Energies are for problem file {os.path.basename(instance_filepath)} is for {num_qubits} qubits and radius {current_radius} of paired hamiltionians")
                print(f"PUCCD calculated energy : {ideal_energy}")

                
                print(f"\nBelow Classical Energies are in solution file {os.path.basename(sol_file_name)} is {num_qubits} qubits and radius {current_radius} of paired hamiltionians")
            
                print(f"DOCI calculated energy : {doci_energy}")
                print(f"FCI calculated energy : {fci_energy}")
             
             
                # pLotting each instance of qubit count given 
                cumlative_iter_time = cumlative_iter_time[1:]
                #print(len(lowest_energy_values), len(cumlative_iter_time))
                #print("lowest energy array" + str(lowest_energy_values))
                #print("cumutaltive : " + str(cumlative_iter_time))
                #
                #print("difference :" + str(np.subtract( np.array(lowest_energy_values), fci_energy)))
                #print("relative difference" + str(np.divide(np.subtract( np.array(lowest_energy_values), fci_energy), fci_energy)))
                #print("absolute relative difference :" + str(np.absolute(np.divide(np.subtract( np.array(lowest_energy_values), fci_energy), fci_energy))))

                approximation_ratio = (np.absolute(np.divide(np.subtract( np.array(lowest_energy_values), fci_energy), fci_energy)))
                # precision factor
                precision = 0.5
                # take arctan of the approximation ratio and scale it to 0 to 1
                approximation_ratio_scaled = np.subtract(1, np.divide(np.arctan(np.multiply(precision,approximation_ratio)), np.pi/2))
                #print("approximation ratio" + str(approximation_ratio))

                # # plot two subplots in the same plot
                # fig, ax = plt.subplots(2, 1, figsize=(10, 10))

                # ax[0].plot(cumlative_iter_time, lowest_energy_values, label='Quantum Energy')
                # ax[0].axhline(y=doci_energy, color='r', linestyle='--', label='DOCI Energy for given Hamiltonian')
                # ax[0].axhline(y=fci_energy, color='g', linestyle='solid', label='FCI Energy for given Hamiltonian')
                # ax[0].set_xlabel('Quantum Execution time (s)')
                # ax[0].set_ylabel('Energy')
                # ax[0].set_title('Energy Comparison: Quantum vs. Classical')

                # ax[1].plot(cumlative_iter_time, approximation_ratio_scaled, label='Solution Quality')
                # #ax[1].plot(cumlative_iter_time,approximation_ratio_scaled, c=cm.hot(np.abs(approximation_ratio_scaled)), edgecolor='none')
                # ax[1].set_xlabel('Quantum Execution time (s)')
                # ax[1].set_ylabel('Solution Quality')
                # ax[1].set_title('Solution Quality')

                # ax[0].grid()
                # ax[1].grid()
                
                
                # plt.legend()
                # # Generate the text to display
                # energy_text = f'Ideal Energy: {ideal_energy:.2f} | DOCI Energy: {doci_energy:.2f} | FCI Energy: {fci_energy:.2f} | Num of Qubits: {num_qubits} | Radius: {current_radius}'

                # # Add the text annotation at the top of the plot
                # plt.annotate(energy_text, xy=(0.5, 0.97), xycoords='figure fraction', ha='center', va='top')

                # #block plot until closed for the last iteration
                # if instance_num == max_circuits:
                #     print("Close plots to continue")
                #     plt.show(block=True)
                # else:
                #     plt.show(block=False)

                # DEVNOTE: not yet capturing classical time, to do
                # unique_id = instance_num * 1000 + 0
                # metrics.store_metric(num_qubits, unique_id, 'opt_exec_time', time.time()-opt_ts)


                # Save final iteration data to metrics.circuit_metrics_final_iter
                # This data includes final counts, cuts, etc.
                parent_folder_save = os.path.join('__data', f'{backend_id}')
            
                # sve the data for this qubit width and instance number
                store_final_iter_to_metrics_json(num_qubits=num_qubits, 
                              radius = radius,
                              instance_num=instance_num,
                              num_shots=num_shots, 
                              converged_thetas_list=thetas_array.x.tolist(),
                              energy = lowest_energy_values[-1],
                             #  iter_size_dist=iter_size_dist, iter_dist=iter_dist,
                              parent_folder_save=parent_folder_save,
                              dict_of_inputs=dict_of_inputs, save_final_counts=save_final_counts,
                              save_res_to_file=save_res_to_file, _instances=_instances)

            lowest_energy_values.clear()

        # for method 2, need to aggregate the detail metrics appropriately for each group
        # Note that this assumes that all iterations of the circuit have completed by this point
        if method == 2:                  
            metrics.process_circuit_metrics_2_level(num_qubits)
            metrics.finalize_group(str(num_qubits))
            
    # Wait for some active circuits to complete; report metrics when groups complete
    ex.throttle_execution(metrics.finalize_group)
        
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)       
             
#     global print_sample_circuit
#     if print_sample_circuit:
#         # print a sample circuit
#         print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
#     #if method == 1: print("\nQuantum Oracle 'Uf' ="); print(Uf_ if Uf_ != None else " ... too large!")

    # Plot metrics for all circuit sizes
    if method == 1:
        metrics.plot_metrics(f"Benchmark Results - Hydrogen Lattice ({method}) - Qiskit",
                options=dict(shots=num_shots))
    elif method == 2:
        #metrics.print_all_circuit_metrics()
        if plot_results:
            plot_results_from_data(**dict_of_inputs)

def plot_results_from_data(num_shots=100, radius = 0.75, max_iter=30, max_circuits = 1,
             method=2,
            score_metric='solution_quality', x_metric='cumulative_exec_time', y_metric='num_qubits', fixed_metrics={},
            num_x_bins=15, y_size=None, x_size=None, x_min=None, x_max=None,
            offset_flag=False,      # default is False for QAOA
            detailed_save_names=False, **kwargs):
    """
    Plot results
    """

    if type(score_metric) == str:
            score_metric = [score_metric]
    suffix = []
    options = []

    

    for sm in score_metric:
        if sm not in metrics.score_label_save_str:
            raise Exception(f"score_metric {sm} not found in metrics.score_label_save_str")
        
        if detailed_save_names:
            # If detailed names are desired for saving plots, put date of creation, etc.
            cur_time=datetime.datetime.now()
            dt = cur_time.strftime("%Y-%m-%d_%H-%M-%S")
            #short_obj_func_str = metrics.score_label_save_str["compute_energy"]
            short_obj_func_str = (metrics.score_label_save_str[sm])
            suffix.append(f'-s{num_shots}_r{radius}_mi{max_iter}_of-{short_obj_func_str}_{dt}') #of=objective function

        else:
            #short_obj_func_str = metrics.score_label_save_str["compute_energy"]
            short_obj_func_str = metrics.score_label_save_str[sm]
            suffix.append(f'of-{short_obj_func_str}') #of=objective function

        obj_str = (metrics.known_score_labels[sm])
        options.append({'shots' : num_shots, 'radius' : radius, 'restarts' : max_circuits, '\nObjective Function' : obj_str})
    suptitle = f"Benchmark Results - Hydrogen Lattice ({method}) - Qiskit"

    h_metrics.plot_all_line_metrics(score_metrics=["energy", "solution_quality", "accuracy_volume"], x_vals=["iteration_count", "cumulative_exec_time"], subplot=True)
    
    metrics.plot_all_area_metrics(f"Benchmark Results - Hydrogen Lattice ({method}) - Qiskit",
                score_metric=score_metric, x_metric=x_metric, y_metric=y_metric,
                fixed_metrics=fixed_metrics, num_x_bins=num_x_bins,
                x_size=x_size, y_size=y_size, x_min=x_min, x_max=x_max,
                offset_flag=offset_flag,
                options=options, suffix=suffix, which_metric='solution_quality', save_metric_label_flag=True)
    
#     metrics.plot_metrics_optgaps(suptitle, options=options, suffix=suffix, objective_func_type = objective_func_type)
    
#     # this plot is deemed less useful
#     #metrics.plot_ECDF(suptitle=suptitle, options=options, suffix=suffix)

#     all_widths = list(metrics.circuit_metrics_final_iter.keys())
#     all_widths = [int(ii) for ii in all_widths]
#     list_of_widths = [all_widths[-1]]
#     metrics.plot_cutsize_distribution(suptitle=suptitle,options=options, suffix=suffix, list_of_widths = list_of_widths)
    
    #metrics.plot_angles_polar(suptitle = suptitle, options = options, suffix = suffix)


def calculate_quality_metric(energy, fci_energy, precision = 4, num_electrons = 2):
    """
    Returns the quality of the solution which is a number between zero and one indicating how close the energy is to the FCI energy.
    """
    _relative_energy = np.absolute(np.divide(np.subtract( np.array(energy), fci_energy), fci_energy))
    
    #scale the solution quality to 0 to 1 using arctan 
    _solution_quality = np.subtract(1, np.divide(np.arctan(np.multiply(precision,_relative_energy)), np.pi/2))

    # define accuracy volume as the absolute energy difference between the FCI energy and the energy of the solution normalized per electron
    _accuracy_volume = np.divide(np.absolute(np.subtract( np.array(energy), fci_energy)), num_electrons)

    return _solution_quality, _accuracy_volume

# # if main, execute method
if __name__ == '__main__': run(max_circuits=2, min_qubits=2,  max_qubits=4)

# # %%

# run()
