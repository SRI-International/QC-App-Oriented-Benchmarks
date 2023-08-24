
"""
Image Recognition Benchmark Program - Qiskit
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
from matplotlib import cm
import glob
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,log_loss
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit import (Aer, ClassicalRegister, QuantumCircuit, QuantumRegister, execute, transpile)
from qiskit.circuit import ParameterVector
from typing import Dict, List, Optional
from qiskit import Aer, execute
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.opflow import ComposedOp, PauliExpectation, StateFn, SummedOp
from qiskit.quantum_info import Statevector,Pauli
from qiskit.result import sampled_expectation_value

sys.path[1:1] = [ "_common", "_common/qiskit", "image-recognition/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit", "../../image-recognition/_common/" ]

import common
import execute as ex
import metrics as metrics



# Image recognition metrics import when needed

global debug
debug = False
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

ir_inputs = dict() #inputs to the run method      #--
verbose = False
print_sample_circuit = True
# Indicates whether to perform the (expensive) pre compute of expectations
do_compute_expectation = True


# saved circuits for display
QC_ = None
Uf_ = None

# #theta parameters
classifier_parameter = namedtuple('classifier_param','theta')

# Qiskit uses the little-Endian convention. 
reverseStep = -1

global train_loss_history,train_accuracy_history
train_loss_history = []
train_accuracy_history = []
# DEBUG prints
# give argument to the python script as "debug" or "true" or "1" to enable debug prints

def fetch_data(train_size = 200):
    
    # Fetch the MNIST dataset from openml
    mnist = fetch_openml('mnist_784')

    # Access the data and target
    # x has all the pixel values of image and has Data shape of (70000, 784)  here 784 is 28*28
    x = mnist.data
    print("shape of x:",x.shape)

    # y has all the labels of the images and has Target shape of (70000,)
    y = mnist.target
    print("shape of y:",y.shape)


    # convert the given y data to integers so that we can filter the data
    y = y.astype(int)

    # Filtering only values with 7 or 9 as we are doing binary classification for now and we will extend it to multi-class classification
    binary_filter = (y == 0) | (y == 1)

    # Filtering the x and y data with the binary filter
    x = x[binary_filter]
    y = y[binary_filter]

    # create a new y data with 0 and 1 values with 7 as 0 and 9 as 1 so that we can use it for binary classification
    # y = (y == 9).astype(int)


    ''' Here X_train is training features , y_train is training labels, X_test is testing features , 
                                                                  y_test is testing labels ,'''

    test_size = int(abs(train_size * 0.25 ))    # Testing size is 25 perc of training size
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, train_size = train_size, random_state=42)
    
    return x, x_train,x_test,y_train,y_test

def preprocess_data(num_qubits, x, x_train, x_test):
    # Step 1: Apply PCA on the entire dataset
    pca = PCA(n_components = num_qubits).fit(x)
    x_pca = pca.transform(x)
    x_train_pca  = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    # Step 2: Apply MinMax scaling to the PCA-transformed data
    scaler = MinMaxScaler(feature_range=(0, 2 * np.pi)).fit(x_pca)
    x_final_train =  scaler.transform(x_train_pca)
    x_final_test  =  scaler.transform(x_test_pca)

    return x_final_train,x_final_test
    
#---------------------------------Start of layers Declaration-----------------------------------
# Define the convolutional circuits
def conv_circ_1(thetas, first, second):
    conv_circ = QuantumCircuit(4)
    conv_circ.rx(thetas[0], first)
    conv_circ.rx(thetas[1], second)
    conv_circ.rz(thetas[2], first)
    conv_circ.rz(thetas[3], second)
    conv_circ.crx(thetas[4], second, first)  
    conv_circ.crx(thetas[5], first, second)
    conv_circ.rx(thetas[6], first)
    conv_circ.rx(thetas[7], second)
    conv_circ.rz(thetas[8], first)
    conv_circ.rz(thetas[9], second)
    # print(conv_circ)
    return conv_circ

# Define the pooling circuits
def pool_circ_1(thetas, first, second):
    pool_circ = QuantumCircuit(4)
    pool_circ.crz(thetas[0], first, second)
    pool_circ.crx(thetas[1], first, second)
    return pool_circ

def pool_circ_2(first, second):
    pool_circ = QuantumCircuit(4)
    pool_circ.crz(first, second)
    return pool_circ

# Quantum Circuits for Convolutional layers
def conv_layer_1(qc, thetas):
    qc = qc.compose(conv_circ_1(thetas, 0, 1))
    qc = qc.compose(conv_circ_1(thetas, 2, 3))
    return qc

def conv_layer_2(qc, thetas):
    qc = qc.compose(conv_circ_1(thetas, 0, 2))
    return qc

# Quantum Circuits for Pooling layers
def pooling_layer_1(qc, thetas):
    qc = qc.compose(pool_circ_1(thetas, 1, 0))
    qc = qc.compose(pool_circ_1(thetas, 3, 2))
    return qc

def pooling_layer_2(qc, thetas):
    qc = qc.compose(pool_circ_2(thetas, 0, 2))
    return qc

#---------------------------------End of layers Declaration-----------------------------------    

# Should change this to dynamic length once working 
global parameter_size
parameter_size = 24


# Create the  ansatz quantum circuit for the VQE algorithm.
def classification_ansatz(x, num_qubits: int,  model_type, theta,*args, **kwargs) -> QuantumCircuit:
    # Generate the ansatz circuit for the VQE algorithm.
    
    if model_type == 'QCNN':
        
        qc = QuantumCircuit(num_qubits,1)  # just to measure the 5 th qubit (4 indexed)\
            
    # Encode the pixel data into the quantum circuit here  x is the input data which is list of 14 values   
    # feature mapping 
        for j in range(num_qubits):
            qc.rx(x[j], j )
            
        qc = qcnn_circ(qc, num_qubits,theta)
        
    # Dynamic QCNN circuit creation ( Will use this once code works fine with existing implementation)
        dynamic_circuit = False
        
        if dynamic_circuit:
            qc = ansatz_qcnn(num_qubits,theta)
        
        
        # Observable to calculate expectation value for Quantum convolutional neural network
        if num_qubits == 8:
            operator = PauliSumOp(SparsePauliOp("IIIIZIII"))
        elif num_qubits == 4:
            operator = PauliSumOp(SparsePauliOp("IIZI"))      
                  
    elif model_type == "QVC":                  # Quantum Variational Circuit 
        
        qc = QuantumCircuit(num_qubits)
        
    # Encode the pixel data into the quantum circuit here  x is the input data which is list of 14 values   
    # feature mapping 
        for j in range(num_qubits):
            qc.rx(x[j], j )
            
        qc = qvc_circ(qc, num_qubits,theta)

        # if  num_qubits == 4:
        operator = PauliSumOp(SparsePauliOp("Z" * num_qubits))

    return qc,operator
    
def qcnn_circ(qc,num_qubits,theta, layer_size = 10):

    if debug == True:
        print(theta)
    start_size = 0
    end_size = layer_size
    # for i in range(num_qubits):
    qc = conv_layer_1(qc, theta[start_size:end_size])
    qc = pooling_layer_1(qc, theta[end_size:end_size+2])
    qc = conv_layer_2(qc, theta[end_size+2:(end_size*2)+2])
    qc = pooling_layer_2(qc,theta[(end_size*2)+2:(end_size*2)+4])
    
    # Pooling Ansatz1 is used by default
    # qc = conv_layer_1(qc, theta1)
    # qc = pooling_layer_1(qc, theta4)
    # qc = conv_layer_2(qc, theta2)
    # qc = pooling_layer_2(qc, theta5)
    # # qc = conv_layer_3(qc, theta3)
    print("Check after circuit creation",qc)
    return qc

def qvc_circ (qc,num_qubits,theta, reps = 3):
        
    # thetas = ParameterVector("t", length=num_qubits*reps*2)
    # print("parameter_vector",parameter_vector)
    counter = 0
    for rep in range(reps):
        for i in range(num_qubits):
            theta = theta[counter]
            qc.ry(theta, i)
            counter += 1
        
        for i in range(num_qubits):
            theta = theta[counter]
            qc.rx(theta, i)
            counter += 1
    
        for j in range(0, num_qubits - 1, 1):
            if rep<reps-1:
                qc.cx(j, j + 1)
    # print("counter",counter)
    return qc

#---------------------------------Start of layers Declaration-----------------------------------
# Define the convolutional circuits
def conv_circ_1(thetas, first, second):
    conv_circ = QuantumCircuit(4)
    conv_circ.rx(thetas[0], first)
    conv_circ.rx(thetas[1], second)
    conv_circ.rz(thetas[2], first)
    conv_circ.rz(thetas[3], second)
    conv_circ.crx(thetas[4], second, first)  
    conv_circ.crx(thetas[5], first, second)
    conv_circ.rx(thetas[6], first)
    conv_circ.rx(thetas[7], second)
    conv_circ.rz(thetas[8], first)
    conv_circ.rz(thetas[9], second)
    # print(conv_circ)
    return conv_circ

# Define the pooling circuits
def pool_circ_1(thetas, first, second):
    pool_circ = QuantumCircuit(4)
    pool_circ.crz(thetas[0], first, second)
    pool_circ.crx(thetas[1], first, second)
    return pool_circ


# Quantum Circuits for Convolutional layers
def conv_layer_1(qc, thetas):
    qc = qc.compose(conv_circ_1(thetas, 0, 1))
    qc = qc.compose(conv_circ_1(thetas, 2, 3))
    return qc

def conv_layer_2(qc, thetas):
    qc = qc.compose(conv_circ_1(thetas, 0, 2))
    return qc

# Quantum Circuits for Pooling layers
def pooling_layer_1(qc, thetas):
    qc = qc.compose(pool_circ_1(thetas, 1, 0))
    qc = qc.compose(pool_circ_1(thetas, 3, 2))
    return qc

def pooling_layer_2(qc, thetas):
    qc = qc.compose(pool_circ_1(thetas, 0, 2))
    return qc


# Define the convolutional circuits
def conv_ansatz(thetas):
    # Your implementation for conv_circ_1 function here
    # print(thetas)
    conv_circ = QuantumCircuit(2)
    conv_circ.rx(thetas[0], 0)
    conv_circ.rx(thetas[1], 1)
    conv_circ.rz(thetas[2], 0)
    conv_circ.rz(thetas[3], 1)


    conv_circ.crx(thetas[4], 0, 1)  
    conv_circ.crx(thetas[5], 1, 0)

    conv_circ.rx(thetas[6], 0)
    conv_circ.rx(thetas[7], 1)
    conv_circ.rz(thetas[8], 0)
    conv_circ.rz(thetas[9], 1)

    conv_circ.crz(thetas[10], 1, 0)  
    conv_circ.x(1)
    conv_circ.crx(thetas[11], 1, 0)
    conv_circ.x(1)

    #conv_circ = QuantumCircuit(2)
    #conv_circ.crx(thetas[0], 0, 1)

    # print(conv_circ)
    return conv_circ
#---------------------------------End of layers Declaration-----------------------------------

# Another dynamic way of declaring QCNN 

def ansatz_qcnn(num_qubits,theta):

    qc = QuantumCircuit(num_qubits)

    num_layers = int(np.ceil(np.log2(num_qubits)))
    num_parameters_per_layer = 12
    num_parameters=num_layers*num_parameters_per_layer
    # theta = ParameterVector("t", length=num_parameters)    

    for i_layer in range(num_layers):
        for i_sub_layer in [0 , 2**i_layer]:            
            for i_q1 in range(i_sub_layer, num_qubits, 2**(i_layer+1)):
                i_q2=2**i_layer+i_q1
                if i_q2<num_qubits:
                    qc=qc.compose(conv_ansatz(theta[num_parameters_per_layer*i_layer:num_parameters_per_layer*(i_layer+1)]), qubits=(i_q1,i_q2))
                    #print("i_q1",i_q1,"i_q2",i_q2)
    
    #print(qc)
    return qc




# Create the benchmark program circuit
# Accepts optional rounds and array of thetas (betas and gammas)
def ImageRecognition (x, num_qubits, model_type, secret_int = 000000, thetas_array = None, parameterized = None):
    # if no thetas_array passed in, create defaults 
    
    # here we are filling this th
    if thetas_array is None:
        thetas_array = np.random.rand(parameter_size)

    global _qc
    global theta
   
   
    # create the circuit the first time, add measurements
    if ex.do_transpile_for_execute:
        
        logger.info(f'*** Constructing parameterized circuit for {num_qubits = } {secret_int}') 
       
    
        theta =  ParameterVector("t", parameter_size)
        
        _qc, operator = classification_ansatz(x = x, num_qubits = num_qubits, model_type = model_type, theta = theta)  #thetas_array = thetas_array) 

    # if model_type == 'QCNN':
    #     _qc ,parameter_vector,thetas_array = qcnn_circ(num_qubits=num_qubits, thetas_array=thetas_array)
    # elif model_type == 'QV'
    # betas = ParameterVector("𝞫", )
    
         
    if debug:
        print("first check of parameters",_qc)
    _measurable_expression = StateFn(operator, is_measurement=True)
    _observables = PauliExpectation().convert(_measurable_expression)
    _qc_array, _formatted_observables = prepare_circuits(_qc, num_qubits, observables=_observables, model_type = model_type)
    
    params = {theta: thetas_array} 
    # Here Parameter
    # params = thetas
        
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


# ############### Expectation Tables

# DEVNOTE: We are building these tables on-demand for now, but for larger circuits
# this will need to be pre-computed ahead of time and stored in a data file to avoid run-time delays.

# dictionary used to store pre-computed expectations, keyed by num_qubits and secret_string
# these are created at the time the circuit is created, then deleted when results are processed
expectations = {}

# Compute array of expectation values in range 0.0 to 1.0
# Use statevector_simulator to obtain exact expectation
def compute_expectation(qc, num_qubits, secret_int, backend_id='statevector_simulator', params=None):
    
    pass  


# Return expected measurement array scaled to number of shots executed
def get_expectation(num_qubits,  num_shots):
   
    pass

    
    
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
def get_random_angles(num_params):
    """Create max_circuit number of random initial conditions"""
    thetas = []
    for i in range(len(num_params)):
        thetas[i] = np.random.choice([-1e-3, 1e-3])
    return thetas


# For initial development
statevector_backend = Aer.get_backend("statevector_simulator")
qasm_backend = Aer.get_backend("qasm_simulator")

#------------------ start of Tentative code may remove later 

# def get_measured_qubits(circuit: QuantumCircuit) -> List[int]:
#     """
#     Get a list of indices of the qubits being measured in a given quantum circuit.
#     """
#     measured_qubits = []

#     for gate, qubits, clbits in circuit.data:
#         if gate.name == "measure":
#             measured_qubits.extend([qubit.index for qubit in qubits])

#     measured_qubits = sorted(list(set(measured_qubits)))

#     return measured_qubits

#------------------ end of Tentative code may remove later 

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

    
def prepare_circuits(base_circuit, num_qubits, observables, model_type):
    """
    Prepare the qubit-wise commuting circuits for a given operator.
    """
    # circuits = list()

    if isinstance(observables, ComposedOp):
        observables = SummedOp([observables])
    # for obs in observables:
    circuit = base_circuit.copy()
    if debug:
        print(observables)
    print(observables)
    # circuit.append(obs[1], qargs=list(range(base_circuit.num_qubits)))
    circuit.append(observables[1], qargs=list(range(base_circuit.num_qubits)))
    if model_type != 'QCNN':
        circuit.measure_all()
    else:
        if num_qubits == 8:
            circuit.measure(4,0)
        elif num_qubits == 4:
            circuit.measure(2,0)
    # circuits.append(circuit)
    # return circuits, observables
    return circuit, observables


def calculate_expectation_values(probabilities, observables):
    """
    Return the expectation values for an operator given the probabilities.
    """
    expectation_values = list()
    print("observables", observables)
    print("probabilities", probabilities)
    for idx, op in enumerate(observables):
        expectation_value = sampled_expectation_value(probabilities[idx], op.primitive)
        expectation_values.append(expectation_value)

    return expectation_values

# -------------------------------------end of simulator expectation value code-----------------------------------

# ------------------Main objective Function to calculate expectation value------------------
# objective Function to compute the energy of a circuit with given parameters and operator
# # Initialize an empty list to store the lowest energy values
lowest_energy_values = []

def compute_exp_sum(result_array, formatted_observables, num_qubits): 
    
    
    # Compute the expectation value of the circuit with respect to the Hamiltonian for optimization

    _probabilities = list()

    for _res in result_array:
        _counts = _res.get_counts()
        _probs = normalize_counts(_counts, num_qubits=num_qubits)
        _probabilities.append(_probs)


    _expectation_values = calculate_expectation_values(_probabilities, formatted_observables)

    result = sum(_expectation_values)

    # Append the energy value to the list
    lowest_energy_values.append(result)

    
    return result   
# # ------------------Simulation Function------------------
# # #%% Storing final iteration data to json file, and to metrics.circuit_metrics_final_iter

# def save_runtime_data(result_dict): # This function will need changes, since circuit metrics dictionaries are now different
#     cm = result_dict.get('circuit_metrics')
#     detail = result_dict.get('circuit_metrics_detail', None)
#     detail_2 = result_dict.get('circuit_metrics_detail_2', None)
#     benchmark_inputs = result_dict.get('benchmark_inputs', None)
#     final_iter_metrics = result_dict.get('circuit_metrics_final_iter')
#     backend_id = result_dict.get('benchmark_inputs').get('backend_id')
    
#     metrics.circuit_metrics_detail_2 = detail_2
 
 
#  #----------------------Removed restart index from here----------------------   
# def save_runtime_data(result_dict):
#     cm = result_dict.get('circuit_metrics')
#     detail = result_dict.get('circuit_metrics_detail', None)
#     detail_2 = result_dict.get('circuit_metrics_detail_2', None)
#     benchmark_inputs = result_dict.get('benchmark_inputs', None)
#     final_iter_metrics = result_dict.get('circuit_metrics_final_iter')
#     backend_id = result_dict.get('benchmark_inputs').get('backend_id')

#     metrics.circuit_metrics_detail_2 = detail_2

#     for width in detail_2:
#         degree = cm[width]['1']['degree']
#         opt = final_iter_metrics[width]['1']['optimal_value']
#         instance_filename = os.path.join(os.path.dirname(__file__),
#             "..", "_common", common.INSTANCE_DIR, f"mc_{int(width):03d}_{int(degree):03d}_000.txt")
#         metrics.circuit_metrics[width] = detail.get(width)
#         metrics.circuit_metrics['subtitle'] = cm.get('subtitle')
        
#         finIterDict = final_iter_metrics[width]['1']
#         if benchmark_inputs['save_final_counts']:
#             iter_dist = {'cuts': finIterDict['cuts'], 'counts': finIterDict['counts'], 'sizes': finIterDict['sizes']}
#         else:
#             iter_dist = None

#         iter_size_dist = {'unique_sizes': finIterDict['unique_sizes'], 'unique_counts': finIterDict['unique_counts'], 'cumul_counts': finIterDict['cumul_counts']}

#         converged_thetas_list = finIterDict.get('converged_thetas_list')
#         parent_folder_save = os.path.join('__data', f'{backend_id}')
#         store_final_iter_to_metrics_json(
#             num_qubits=int(width),
#             degree=int(degree),
#             num_shots=int(benchmark_inputs['num_shots']),
#             converged_thetas_list=converged_thetas_list,
#             opt=opt,
#             iter_size_dist=iter_size_dist,
#             iter_dist=iter_dist,
#             dict_of_inputs=benchmark_inputs,
#             parent_folder_save=parent_folder_save,
#             save_final_counts=False,
#             save_res_to_file=True,
#             _instances=None
#         )



# def store_final_iter_to_metrics_json(num_qubits,
#                                      degree,
#                                      restart_ind,
#                                      num_shots,
#                                      converged_thetas_list,
#                                      opt,
#                                      iter_size_dist,
#                                      iter_dist,
#                                      parent_folder_save,
#                                      dict_of_inputs,
#                                      save_final_counts,
#                                      save_res_to_file,
#                                      _instances=None):
#     """
#     For a given graph (specified by num_qubits and degree),
#     1. For a given restart, store properties of the final minimizer iteration to metrics.circuit_metrics_final_iter, and
#     2. Store various properties for all minimizer iterations for each restart to a json file.
#     Parameters
#     ----------
#         num_qubits, degree, restarts, num_shots : ints
#         parent_folder_save : string (location where json file will be stored)
#         dict_of_inputs : dictionary of inputs that were given to run()
#         save_final_counts: bool. If true, save counts, cuts and sizes for last iteration to json file.
#         save_res_to_file: bool. If False, do not save data to json file.
#         iter_size_dist : dictionary containing distribution of cut sizes.  Keys are 'unique_sizes', 'unique_counts' and         'cumul_counts'
#         opt (int) : Max Cut value
#     """
#     # In order to compare with uniform random sampling, get some samples
#     unif_cuts, unif_counts, unif_sizes, unique_counts_unif, unique_sizes_unif, cumul_counts_unif = uniform_cut_sampling(
#         num_qubits, degree, num_shots, _instances)
#     unif_dict = {'unique_counts_unif': unique_counts_unif,
#                  'unique_sizes_unif': unique_sizes_unif,
#                  'cumul_counts_unif': cumul_counts_unif}  # store only the distribution of cut sizes, and not the cuts themselves

#     # Store properties such as (cuts, counts, sizes) of the final iteration, the converged theta values, as well as the known optimal value for the current problem, in metrics.circuit_metrics_final_iter. Also store uniform cut sampling results
#     metrics.store_props_final_iter(num_qubits, restart_ind, 'optimal_value', opt)
#     metrics.store_props_final_iter(num_qubits, restart_ind, None, iter_size_dist)
#     metrics.store_props_final_iter(num_qubits, restart_ind, 'converged_thetas_list', converged_thetas_list)
#     metrics.store_props_final_iter(num_qubits, restart_ind, None, unif_dict)
#     # metrics.store_props_final_iter(num_qubits, restart_ind, None, iter_dist) # do not store iter_dist, since it takes a lot of memory for larger widths, instead, store just iter_size_dist

#     global radius
#     if save_res_to_file:
#         # Save data to a json file
#         dump_to_json(parent_folder_save, num_qubits,
#                      radius, restart_ind, iter_size_dist, iter_dist, dict_of_inputs, converged_thetas_list, opt, unif_dict,
#                      save_final_counts=save_final_counts)

# def dump_to_json(parent_folder_save, num_qubits, radius, restart_ind, iter_size_dist, iter_dist,
#                  dict_of_inputs, converged_thetas_list, opt, unif_dict, save_final_counts=False):
#     """
#     For a given problem (specified by number of qubits and graph degree) and restart_index, 
#     save the evolution of various properties in a json file.
#     Items stored in the json file: Data from all iterations (iterations), inputs to run program ('general properties'), converged theta values ('converged_thetas_list'), max cut size for the graph (optimal_value), distribution of cut sizes for random uniform sampling (unif_dict), and distribution of cut sizes for the final iteration (final_size_dist)
#     if save_final_counts is True, then also store the distribution of cuts 
#     """
#     if not os.path.exists(parent_folder_save): os.makedirs(parent_folder_save)
#     store_loc = os.path.join(parent_folder_save,'width_{}_restartInd_{}.json'.format(num_qubits, restart_ind))
    
#     # Obtain dictionary with iterations data corresponding to given restart_ind 
#     all_restart_ids = list(metrics.circuit_metrics[str(num_qubits)].keys())
#     ids_this_restart = [r_id for r_id in all_restart_ids if int(r_id) // 1000 == restart_ind]
#     iterations_dict_this_restart =  {r_id : metrics.circuit_metrics[str(num_qubits)][r_id] for r_id in ids_this_restart}
#     # Values to be stored in json file
#     dict_to_store = {'iterations' : iterations_dict_this_restart}
#     dict_to_store['general_properties'] = dict_of_inputs
#     dict_to_store['converged_thetas_list'] = converged_thetas_list
#     dict_to_store['optimal_value'] = opt
#     dict_to_store['unif_dict'] = unif_dict
#     dict_to_store['final_size_dist'] = iter_size_dist
#     # Also store the value of counts obtained for the final counts
#     if save_final_counts:
#         dict_to_store['final_counts'] = iter_dist
#                                         #iter_dist.get_counts()
#     # Now save the output
#     with open(store_loc, 'w') as outfile:
#         json.dump(dict_to_store, outfile)

# # %% Loading saved data (from json files)

# def load_data_and_plot(folder, backend_id=None, **kwargs):
#     """
#     The highest level function for loading stored data from a previous run
#     and plotting optgaps and area metrics

#     Parameters
#     ----------
#     folder : string
#         Directory where json files are saved.
#     """
#     _gen_prop = load_all_metrics(folder, backend_id=backend_id)
#     if _gen_prop != None:
#         gen_prop = {**_gen_prop, **kwargs}
#         plot_results_from_data(**gen_prop)


# def load_all_metrics(folder, backend_id=None):
#     """
#     Load all data that was saved in a folder.
#     The saved data will be in json files in this folder

#     Parameters
#     ----------
#     folder : string
#         Directory where json files are saved.

#     Returns
#     -------
#     gen_prop : dict
#         of inputs that were used in maxcut_benchmark.run method
#     """
#     # Note: folder here should be the folder where only the width=... files are stored, and not a folder higher up in the directory
#     assert os.path.isdir(folder), f"Specified folder ({folder}) does not exist."
    
#     metrics.init_metrics()
    
#     list_of_files = os.listdir(folder)
#     width_restart_file_tuples = [(*get_width_restart_tuple_from_filename(fileName), fileName)
#                                  for (ind, fileName) in enumerate(list_of_files) if fileName.startswith("width")]  # list with elements that are tuples->(width,restartInd,filename)

#     width_restart_file_tuples = sorted(width_restart_file_tuples, key=lambda x:(x[0], x[1])) #sort first by width, and then by restartInd
#     distinct_widths = list(set(it[0] for it in width_restart_file_tuples)) 
#     list_of_files = [
#         [tup[2] for tup in width_restart_file_tuples if tup[0] == width] for width in distinct_widths
#         ]
    
#     # connot continue without at least one dataset
#     if len(list_of_files) < 1:
#         print("ERROR: No result files found")
#         return None
        
#     # for width_files in list_of_files:
#     #     # For each width, first load all the restart files
#     #     for fileName in width_files:
#     #         gen_prop = load_from_width_restart_file(folder, fileName)
        
#     #     # next, do processing for the width
#     #     method = gen_prop['method']
#     #     if method == 2:
#     #         num_qubits, _ = get_width_restart_tuple_from_filename(width_files[0])
#     #         metrics.process_circuit_metrics_2_level(num_qubits)
#     #         metrics.finalize_group(str(num_qubits))
            
#     # override device name with the backend_id if supplied by caller
#     if backend_id != None:
#         metrics.set_plot_subtitle(f"Device = {backend_id}")
            
#     return gen_prop


# # ----------------need to revise the below code------------------

# # def load_from_width_restart_file(folder, fileName):
# #     """
# #     Given a folder name and a file in it, load all the stored data and store the values in metrics.circuit_metrics.
# #     Also return the converged values of thetas, the final counts and general properties.

# #     Parameters
# #     ----------
# #     folder : string
# #         folder where the json file is located
# #     fileName : string
# #         name of the json file

# #     Returns
# #     -------
# #     gen_prop : dict
# #         of inputs that were used in maxcut_benchmark.run method
# #     """
    
# #     # Extract num_qubits and s from file name
# #     num_qubits, restart_ind = get_width_restart_tuple_from_filename(fileName)
# #     print(f"Loading from {fileName}, corresponding to {num_qubits} qubits and restart index {restart_ind}")
# #     with open(os.path.join(folder, fileName), 'r') as json_file:
# #         data = json.load(json_file)
# #         gen_prop = data['general_properties']
# #         converged_thetas_list = data['converged_thetas_list']
# #         unif_dict = data['unif_dict']
# #         opt = data['optimal_value']
# #         if gen_prop['save_final_counts']:
# #             # Distribution of measured cuts
# #             final_counts = data['final_counts']
# #         final_size_dist = data['final_size_dist']
        
# #         backend_id = gen_prop.get('backend_id')
# #         metrics.set_plot_subtitle(f"Device = {backend_id}")
        
# #         # Update circuit metrics
# #         for circuit_id in data['iterations']:
# #             # circuit_id = restart_ind * 1000 + minimizer_loop_ind
# #             for metric, value in data['iterations'][circuit_id].items():
# #                 metrics.store_metric(num_qubits, circuit_id, metric, value)
                
# #         method = gen_prop['method']
# #         if method == 2:
# #             metrics.store_props_final_iter(num_qubits, restart_ind, 'optimal_value', opt)
# #             metrics.store_props_final_iter(num_qubits, restart_ind, None, final_size_dist)
# #             metrics.store_props_final_iter(num_qubits, restart_ind, 'converged_thetas_list', converged_thetas_list)
# #             metrics.store_props_final_iter(num_qubits, restart_ind, None, unif_dict)
# #             if gen_prop['save_final_counts']:
# #                 metrics.store_props_final_iter(num_qubits, restart_ind, None, final_counts)

# #     return gen_prop
    

# # def get_width_restart_tuple_from_filename(fileName):
# #     """
# #     Given a filename, extract the corresponding width and degree it corresponds to
# #     For example the file "width=4_degree=3.json" corresponds to 4 qubits and degree 3

# #     Parameters
# #     ----------
# #     fileName : TYPE
# #         DESCRIPTION.

# #     Returns
# #     -------
# #     num_qubits : int
# #         circuit width
# #     degree : int
# #         graph degree.

# #     """
# #     pattern = 'width_([0-9]+)_restartInd_([0-9]+).json'
# #     match = re.search(pattern, fileName)

# #     # assert match is not None, f"File {fileName} found inside folder. All files inside specified folder must be named in the format 'width_int_restartInd_int.json"
# #     num_qubits = int(match.groups()[0])
# #     degree = int(match.groups()[1])
# #     return (num_qubits,degree)

# # ----------------need to revise the above code------------------

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if p < 0.5:
            p = 0
        else:
            p = 1
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss



MAX_QUBITS = 8
# # iter_dist = {'cuts' : [], 'counts' : [], 'sizes' : []} # (list of measured bitstrings, list of corresponding counts, list of corresponding cut sizes)
# # iter_size_dist = {'unique_sizes' : [], 'unique_counts' : [], 'cumul_counts' : []} # for the iteration being executed, stores the distribution for cut sizes
# saved_result = {  }
# instance_filename = None

#radius = None


debug = False 

def run (min_qubits= 4 , max_qubits = 8, model_type = 'QCNN',  train_size = 200,  max_circuits=3, num_shots=1000,
        method=2, loss_type = 'cross_entropy', print_status = True, batch_size = 35, epochs = 1, thetas_array=None, parameterized= False, do_fidelities=True,
        max_iter=30, score_metric='fidelity', x_metric='cumulative_exec_time', y_metric='num_qubits',
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
    # dict_of_inputs = {**dict_of_inputs, **{'thetas_array': thetas, 'max_circuits' : max_circuits}}
    dict_of_inputs = {**dict_of_inputs, **{'thetas_array': thetas}}
    
    # Delete some entries from the dictionary
    for key in ["hub", "group", "project", "provider_backend"]:
        dict_of_inputs.pop(key)
    
    global image_recogntion_inputs
    image_recogntion_inputs = dict_of_inputs
    
    global QC_
    global circuits_done
    global minimizer_loop_index
    global opt_ts
    global predictions,test_accuracy_history
    
    print("Image Recognition Benchmark Program - Qiskit")

    QC_ = None
    
    # Importing the data from dataset preprocess done in next step
    x, x_train,x_test,y_train,y_test = fetch_data(train_size)
    
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
        metrics.store_metric(num_qubits, s_int, 'fidelity', fidelity)

    def execution_handler2 (qc, result, num_qubits, s_int, num_shots):
        # Stores the results to the global saved_result variable
        global saved_result
        saved_result = result
     
    # Initialize execution module using the execution result handler above and specified backend_id
    # for method=2 we need to set max_jobs_active to 1, so each circuit completes before continuing
    if method == 2:
        ex.max_jobs_active = batch_size
        ex.init_execution(execution_handler2)
    else:
        ex.init_execution(execution_handler)
    
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options)

    # for noiseless simulation, set noise model to be None
    # ex.set_noise_model(None)
    
    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    # DEVNOTE: increment by 4 for QML circuits
    global parameter_filepath
    for num_qubits in range(min_qubits, max_qubits + 1, 4):
        
        if method == 1:
            print(f"************\nExecuting fixed parameters and testing for num_qubits = {num_qubits}")
        else:
            print(f"************\nExecuting training and testing for num_qubits = {num_qubits}")
        # looping all instance files according to max_circuits given        
# ***************************** for instance_num in range(1, max_circuits + 1):

        # Pre processing the data inside Qubit loop as we are doing PCA accorind to qubit dimensions
        x_final_train, x_final_test = preprocess_data(num_qubits, x, x_train, x_test)
        
        
        print("num_qubits", num_qubits , "min qubits" , min_qubits)
        parameter_filepath = os.path.join(os.path.dirname(__file__),"..", "_common",
                    common.INSTANCE_DIR, f"{num_qubits}_qubit.json")   
        
        if debug:
            print( " parameter_filepath", parameter_filepath )

        # # operator is paired hamiltonian  for each instance in the loop  
        parameters = common.read_parameters(parameter_filepath)
            
        if debug:
            print("params" , parameters)
            

        if method == 1:
            # create the circuit for given qubit size and secret string, store time metric
            ts = time.time()
        # DEVNOTE:  Primary focus is on method 2
            thetas_array_0 = parameters  # Can be modified to work for input parameters as well
            for x ,y in zip(x_final_test,y_test):
                qc, frmt_obs, params = ImageRecognition(x = x , model_type = model_type,
                                                            num_qubits = num_qubits,thetas_array= thetas_array_0, parameterized= parameterized)
            # for qc in qc_array:
                metrics.store_metric(num_qubits, 'create_time', time.time()-ts)
                # collapse the sub-circuit levels used in this benchmark (for qiskit)
                qc.bind_parameters(params)
                qc2 = qc.decompose()

                # submit circuit for execution on target (simulator, cloud simulator, or hardware)
                ex.submit_circuit(qc2, num_qubits, shots=num_shots, params=params)
                
                
                
        if method == 2:
            # A unique circuit index used inside the inner minimizer loop as identifier
            minimizer_loop_index = 0 # Value of 0 corresponds to the 0th iteration of the minimizer
            start_iters_t = time.time()

            # Always start by enabling transpile ...
            ex.set_tranpilation_flags(do_transpile_metrics=True, do_transpile_for_execute=True)
                
            logger.info(f'===============  Begin method 2 loop, enabling transpile')
            
            instance_num = 1
     
                            
            # function to calculate the loss function
            def loss_function(theta, x_batch, y_batch, is_draw_circ=False, is_print=False ,final_run = False):
                # Every circuit needs a unique id; add unique_circuit_index instead of s_int
                if debug:
                    print('theta', theta , 'x_batch' , x_batch , 'y_batch' , y_batch)
                global minimizer_loop_index
                instance_num , minimizer_loop_index = 1,1
                unique_id = instance_num * 1000 + minimizer_loop_index
                res = []
                quantum_execution_time = 0.0
                quantum_elapsed_time = 0.0
                prediction_label = []
                i_draw = 0
                # create the ansatz from the ooperator, resulting in multiple circuits, one for each measured basis
                ts = time.time()
                for data_point, label in zip(x_batch, y_batch):
                # Create the quantum circuit for the data point
                    qc, frmt_obs, params = ImageRecognition(x = data_point , model_type= model_type, num_qubits=num_qubits,
                                        secret_int=unique_id, thetas_array= thetas_array, parameterized= parameterized)
                    
                    metrics.store_metric(num_qubits, unique_id, 'create_time', time.time()-ts)
                    if debug:
                        print("create time:" + str(time.time() -ts))
                    #print("store metrics" +str(metrics.circuit_metrics[str(method)]['1000']))
                    
                    # loop over each of the circuits that are generated with basis measurements and execute
        # ************* for qc in qc_array:
                    print("initial circuit",qc) 
                    # qc_upd = qc.bind_parameters(params)
                    # qc2 = qc_upd.decompose()
                    # qc2 = qc_upd
                    
                        # Circuit Creation and Decomposition end
                        #************************************************
                        
                        #************************************************
                        #*** Quantum Part: Execution of Circuits ***
                        # submit circuit for execution on target with the current parameters
                    # print("final circuit",qc2)
                    ex.submit_circuit(qc, num_qubits, unique_id, shots=num_shots, params=params)
                    # debug = True
                    # if debug:
                    print("abc")
                    print("submit circuit id" + str(unique_id) )

                        
                        # Must wait for circuit to complete
                        #ex.throttle_execution(metrics.finalize_group)

                        # finalize execution of group of circuits
                    ex.finalize_execution(None, report_end=False)    # don't finalize group until all circuits done
                    print("ABC")
                    # after first execution and thereafter, no need for transpilation if parameterized
                    if parameterized:
                        ex.set_tranpilation_flags(do_transpile_metrics=False, do_transpile_for_execute=False)
                        logger.info(f'**** First execution complete, disabling transpile')
                    #************************************************
                    
                    global saved_result
                    res.append(saved_result)
                    print("Result" , res)
                    if debug:
                        print("saved result: "+ str(saved_result))

                        # tapping into circuit metric exect time:
                    if debug:
                        print("circuit metrics method: " + str(num_qubits) + " id: " + str(unique_id) )
                    quantum_execution_time = quantum_execution_time + metrics.circuit_metrics[str(num_qubits)][str(unique_id)]['exec_time']
                    quantum_elapsed_time = quantum_elapsed_time + metrics.circuit_metrics[str(num_qubits)][str(unique_id)]['elapsed_time']
                    
                        # Fidelity Calculation and Storage
                        # _, fidelity = analyze_and_print_result(qc, saved_result, num_qubits, unique_id, num_shots) 
                        
                        #************************************************
                        #*** Classical Processing of Results - essential to optimizer ***
                    global opt_ts
                    if debug:
                        print("iteration time :" +str(quantum_execution_time))
                    global cumulative_iter_time
        # ************* cumlative_iter_time.append(cumlative_iter_time[-1] + quantum_execution_time)
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

                    expectation_value = compute_exp_sum(result_array = res, formatted_observables = frmt_obs, num_qubits=num_qubits)

                    prediction_label.append(expectation_value)
                    
                if loss_type == 'square_loss':
                    loss = square_loss(y_batch, prediction_label)
                elif loss_type == 'cross_entropy':
                    loss = log_loss(y_batch, prediction_label)
                    # calculate the solution quality
                    # solution_quality, accuracy_volume = calculate_quality_metric(energy, fci_energy, precision=0.5, num_electrons=num_qubits)
                    # metrics.store_metric(str(num_qubits), str(unique_id), 'energy', energy)
                    # metrics.store_metric(str(num_qubits), str(unique_id), 'fci_energy', fci_energy)
                    # metrics.store_metric(str(num_qubits), str(unique_id), 'solution_quality', solution_quality)
                    # metrics.store_metric(str(num_qubits), str(unique_id), 'accuracy_volume', accuracy_volume)
                    # metrics.store_metric(str(num_qubits), str(unique_id), 'fci_energy', fci_energy)
                    # metrics.store_metric(str(num_qubits), str(unique_id), 'doci_energy', doci_energy)
                    # metrics.store_metric(str(num_qubits), str(unique_id), 'radius', current_radius)
                    # metrics.store_metric(str(num_qubits), str(unique_id), 'iteration_count', minimizer_loop_index)
                if print_status:
                    predictions = []
                    for i in range(len(y_batch)):
                            if prediction_label[i] > 0.5:
                                predictions.append(1)
                            else:
                                predictions.append(0)
                        # print(predictions)
                    accuracy = accuracy_score(y_batch, predictions)
                    print("Accuracy:", accuracy, "loss:", loss)
                    train_loss_history.append(loss)
                    train_accuracy_history.append(accuracy)
                
                if final_run == True:
                    accuracy = accuracy_score(y_batch, predictions)
                    print(("Accuracy:", accuracy, "loss:", loss))
                    # Calculate the test accuracy on the test set after each data point
                    test_accuracy = accuracy_score(y_test[:len(predictions)], predictions)

                    # Store the test accuracy in the history list after each data point
                    test_accuracy_history.append(test_accuracy)
                return loss
                    
                    # training samples data size 
            data_size  = len(x_final_train)

                # Number of batches 
            batch_count =  (data_size + batch_size - 1) // batch_size   
            
            weights = np.random.rand(24)

            def callback():
                pass
            
            for epoch in range(epochs):
                for batch in range(batch_count):
                    
                    # indices = np.random.choice(len(x_final_train), size=batch_size, replace=False)
                    # x_batch = x_final_train[indices]
                    # y_batch = y_train[indices]

                # start and end indeces for current_batch( Here batch is index of above batch)
                    start_index = batch * batch_size   # batch will be 0 for first index
                    end_index   = min((batch + 1 ) * batch_size, data_size)
                
                # batch extraction 
                    x_batch = x_final_train[start_index:end_index]
                    y_batch = y_train[start_index:end_index]
            
            
                    print(f"Current batch is {batch}")
                # Minimize the loss using SPSA optimizer
                    is_draw,is_print = False,True
                    theta = minimize(loss_function, x0 = weights, args=(x_batch,y_batch,is_draw,is_print), method="COBYLA", tol=0.001, 
                                            callback=callback, options={'maxiter': 150, 'disp': False} )
            #theta=SPSA(maxiter=100).minimize(loss_function, x0=weights)
                    weights = theta.x
                    print(weights)
                    print(len(weights))
                loss = theta.fun
                print(f"Epoch {epoch+1}/{epochs}, loss = {loss:.4f}")

            theta = loss_function(weights,x_final_test,y_test,print_status = False,final_run=True)
            # test_accuracy_history = []
            # for data_point in x_final_test:
            #     qc = qcnn_model(data_point, num_qubits)
            #     qc_upd = qc.bind_parameters(theta.x)
            #         # Simulate the quantum circuit and get the result
            #     if expectation_calc_method == True:
            #         # val = expectation_calc_qcnn.calculate_expectation(qc,shots=num_shots,num_qubits=num_qubits)   
            #         val = exp_cal(qc_upd)   
            #         val=(val+1)*0.5
            #         if val > 0.5:
            #             predicted_label = 1
            #         else:
            #             predicted_label = 0
            #     print("predicted_label",predicted_label)
            #     predictions.append(predicted_label)
                
            #     # Calculate the test accuracy on the test set after each data point
            #     test_accuracy = accuracy_score(y_test[:len(predictions)], predictions)

            #     # Store the test accuracy in the history list after each data point
            #     test_accuracy_history.append(test_accuracy)


        # Evaluate the QCNN accuracy on the test set once the model is trained and tested
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy:", accuracy)
        print(f"This is for {num_qubits} qubits")
        print("predictions",predictions)


        # training loss after each iteration
        # plt.plot(range(1, num_epochs * 100 + 1), train_loss_history)
        plt.plot(range(1, len(train_loss_history)+1), train_loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.title(f'Training Loss History for {num_qubits} qubits')
        plt.show()

        # training accuracy after each iteration
        plt.plot(range(1, len(train_accuracy_history)+1), train_accuracy_history)
        plt.xlabel('Iteration')
        plt.ylabel('Training Accuracy')
        plt.title(f'Training Accuracy History for {num_qubits} qubits')
        plt.show()


        # testing accuracy after each data point
        plt.plot(range(1, len(x_final_test) + 1), test_accuracy_history)
        plt.xlabel('Data Point')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy after Each Data Point')
        plt.show()
        
    # Wait for some active circuits to complete; report metrics when groups complete
    ex.throttle_execution(metrics.finalize_group)
        
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)
# ------------------start of not needed code
            
            # def callback_thetas_array(thetas_array):
            #     if DEBUG:
            #         print("callback_thetas_array" + str(thetas_array))
            #     else:
            #         pass
                
            # if comfort and verbose:
            #     print("")
                
            # initial_parameters = np.random.random(size=1)     
            # # objective_function(thetas_array=None)       
            # if debug:
            #     print("The initial parameters are : "+ str(initial_parameters))

            #     # thetas_array = minimize(objective_function,
            #     #         x0=initial_parameters.ravel(),
            #     #         method='COBYLA',
            #     #         tol=1e-3,
            #     #         options={'maxiter': max_iter, 'disp': False},
            #     #         callback=callback_thetas_array) 
                
            #     ideal_energy = objective_function(thetas_array.x)

            #     current_radius = float(os.path.basename(instance_filepath).split('_')[2])
            #     current_radius += float(os.path.basename(instance_filepath).split('_')[3][:2])*.01

            #     print(f"\nBelow Energies are for problem file {os.path.basename(instance_filepath)} is for {num_qubits} qubits and radius {current_radius} of paired hamiltionians")
            #     print(f"PUCCD calculated energy : {ideal_energy}")

                
            #     print(f"\nBelow Classical Energies are in solution file {os.path.basename(sol_file_name)} is {num_qubits} qubits and radius {current_radius} of paired hamiltionians")
            
            #     print(f"DOCI calculated energy : {doci_energy}")
            #     print(f"FCI calculated energy : {fci_energy}")
            
# ------------------End of not needed code
             
                # # pLotting each instance of qubit count given 
                # cumlative_iter_time = cumlative_iter_time[1:]
                # #print(len(lowest_energy_values), len(cumlative_iter_time))
                # #print("lowest energy array" + str(lowest_energy_values))
                # #print("cumutaltive : " + str(cumlative_iter_time))
                # #
                # #print("difference :" + str(np.subtract( np.array(lowest_energy_values), fci_energy)))
                # #print("relative difference" + str(np.divide(np.subtract( np.array(lowest_energy_values), fci_energy), fci_energy)))
                # #print("absolute relative difference :" + str(np.absolute(np.divide(np.subtract( np.array(lowest_energy_values), fci_energy), fci_energy))))

                # approximation_ratio = (np.absolute(np.divide(np.subtract( np.array(lowest_energy_values), fci_energy), fci_energy)))
                # # precision factor
                # precision = 0.5
                # # take arctan of the approximation ratio and scale it to 0 to 1
                # approximation_ratio_scaled = np.subtract(1, np.divide(np.arctan(np.multiply(precision,approximation_ratio)), np.pi/2))
                # #print("approximation ratio" + str(approximation_ratio))

                # # # plot two subplots in the same plot
                # # fig, ax = plt.subplots(2, 1, figsize=(10, 10))

                # # ax[0].plot(cumlative_iter_time, lowest_energy_values, label='Quantum Energy')
                # # ax[0].axhline(y=doci_energy, color='r', linestyle='--', label='DOCI Energy for given Hamiltonian')
                # # ax[0].axhline(y=fci_energy, color='g', linestyle='solid', label='FCI Energy for given Hamiltonian')
                # # ax[0].set_xlabel('Quantum Execution time (s)')
                # # ax[0].set_ylabel('Energy')
                # # ax[0].set_title('Energy Comparison: Quantum vs. Classical')

                # # ax[1].plot(cumlative_iter_time, approximation_ratio_scaled, label='Solution Quality')
                # # #ax[1].plot(cumlative_iter_time,approximation_ratio_scaled, c=cm.hot(np.abs(approximation_ratio_scaled)), edgecolor='none')
                # # ax[1].set_xlabel('Quantum Execution time (s)')
                # # ax[1].set_ylabel('Solution Quality')
                # # ax[1].set_title('Solution Quality')

                # # ax[0].grid()
                # # ax[1].grid()
                
                
                # # plt.legend()
                # # # Generate the text to display
                # # energy_text = f'Ideal Energy: {ideal_energy:.2f} | DOCI Energy: {doci_energy:.2f} | FCI Energy: {fci_energy:.2f} | Num of Qubits: {num_qubits} | Radius: {current_radius}'

                # # # Add the text annotation at the top of the plot
                # # plt.annotate(energy_text, xy=(0.5, 0.97), xycoords='figure fraction', ha='center', va='top')

                # # #block plot until closed for the last iteration
                # # if instance_num == max_circuits:
                # #     print("Close plots to continue")
                # #     plt.show(block=True)
                # # else:
                # #     plt.show(block=False)

                # # DEVNOTE: not yet capturing classical time, to do
                # # unique_id = instance_num * 1000 + 0
                # # metrics.store_metric(num_qubits, unique_id, 'opt_exec_time', time.time()-opt_ts)


                # # Save final iteration data to metrics.circuit_metrics_final_iter
                # # This data includes final counts, cuts, etc.
                # parent_folder_save = os.path.join('__data', f'{backend_id}')
            
                # # sve the data for this qubit width and instance number
                # store_final_iter_to_metrics_json(num_qubits=num_qubits, 
                #               radius = radius,
                #               instance_num=instance_num,
                #               num_shots=num_shots, 
                #               converged_thetas_list=thetas_array.x.tolist(),
                #               energy = lowest_energy_values[-1],
                #              #  iter_size_dist=iter_size_dist, iter_dist=iter_dist,
                #               parent_folder_save=parent_folder_save,
                #               dict_of_inputs=dict_of_inputs, save_final_counts=save_final_counts,
                #               save_res_to_file=save_res_to_file, _instances=_instances)

            # lowest_energy_values.clear()

    #     # for method 2, need to aggregate the detail metrics appropriately for each group
    #     # Note that this assumes that all iterations of the circuit have completed by this point
    #     if method == 2:                  
    #         metrics.process_circuit_metrics_2_level(num_qubits)
    #         metrics.finalize_group(str(num_qubits))
            
    # # Wait for some active circuits to complete; report metrics when groups complete
    # ex.throttle_execution(metrics.finalize_group)
        
    # # Wait for all active circuits to complete; report metrics when groups complete
    # ex.finalize_execution(metrics.finalize_group)       
             
#     global print_sample_circuit
#     if print_sample_circuit:
#         # print a sample circuit
#         print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
#     #if method == 1: print("\nQuantum Oracle 'Uf' ="); print(Uf_ if Uf_ != None else " ... too large!")

    # # Plot metrics for all circuit sizes
    # if method == 1:
    #     metrics.plot_metrics(f"Benchmark Results - Hydrogen Lattice ({method}) - Qiskit",
    #             options=dict(shots=num_shots))
    # elif method == 2:
    #     #metrics.print_all_circuit_metrics()
    #     if plot_results:
    #         plot_results_from_data(**dict_of_inputs)

# def plot_results_from_data(num_shots=100, radius = 0.75, max_iter=30, max_circuits = 1,
#              method=2,
#             score_metric='solution_quality', x_metric='cumulative_exec_time', y_metric='num_qubits', fixed_metrics={},
#             num_x_bins=15, y_size=None, x_size=None, x_min=None, x_max=None,
#             offset_flag=False,      # default is False for QAOA
#             detailed_save_names=False, **kwargs):
#     """
#     Plot results
#     """

#     if type(score_metric) == str:
#             score_metric = [score_metric]
#     suffix = []
#     options = []

    

#     for sm in score_metric:
#         if sm not in metrics.score_label_save_str:
#             raise Exception(f"score_metric {sm} not found in metrics.score_label_save_str")
        
#         if detailed_save_names:
#             # If detailed names are desired for saving plots, put date of creation, etc.
#             cur_time=datetime.datetime.now()
#             dt = cur_time.strftime("%Y-%m-%d_%H-%M-%S")
#             #short_obj_func_str = metrics.score_label_save_str["ompute_exp_sum"]
#             short_obj_func_str = (metrics.score_label_save_str[sm])
#             suffix.append(f'-s{num_shots}_r{radius}_mi{max_iter}_of-{short_obj_func_str}_{dt}') #of=objective function

#         else:
#             #short_obj_func_str = metrics.score_label_save_str["compute_exp_sum"]
#             short_obj_func_str = metrics.score_label_save_str[sm]
#             suffix.append(f'of-{short_obj_func_str}') #of=objective function

#         obj_str = (metrics.known_score_labels[sm])
#         options.append({'shots' : num_shots, 'radius' : radius, 'restarts' : max_circuits, '\nObjective Function' : obj_str})
#     suptitle = f"Benchmark Results - Hydrogen Lattice ({method}) - Qiskit"

#     h_metrics.plot_all_line_metrics(score_metrics=["energy", "solution_quality", "accuracy_volume"], x_vals=["iteration_count", "cumulative_exec_time"], subplot=True)
    
#     metrics.plot_all_area_metrics(f"Benchmark Results - Hydrogen Lattice ({method}) - Qiskit",
#                 score_metric=score_metric, x_metric=x_metric, y_metric=y_metric,
#                 fixed_metrics=fixed_metrics, num_x_bins=num_x_bins,
#                 x_size=x_size, y_size=y_size, x_min=x_min, x_max=x_max,
#                 offset_flag=offset_flag,
#                 options=options, suffix=suffix, which_metric='solution_quality', save_metric_label_flag=True)
    
# #     metrics.plot_metrics_optgaps(suptitle, options=options, suffix=suffix, objective_func_type = objective_func_type)
    
# #     # this plot is deemed less useful
# #     #metrics.plot_ECDF(suptitle=suptitle, options=options, suffix=suffix)

# #     all_widths = list(metrics.circuit_metrics_final_iter.keys())
# #     all_widths = [int(ii) for ii in all_widths]
# #     list_of_widths = [all_widths[-1]]
# #     metrics.plot_cutsize_distribution(suptitle=suptitle,options=options, suffix=suffix, list_of_widths = list_of_widths)
    
#     #metrics.plot_angles_polar(suptitle = suptitle, options = options, suffix = suffix)


# def calculate_quality_metric(energy, fci_energy, precision = 4, num_electrons = 2):
#     """
#     Returns the quality of the solution which is a number between zero and one indicating how close the energy is to the FCI energy.
#     """
#     _relative_energy = np.absolute(np.divide(np.subtract( np.array(energy), fci_energy), fci_energy))
    
#     #scale the solution quality to 0 to 1 using arctan 
#     _solution_quality = np.subtract(1, np.divide(np.arctan(np.multiply(precision,_relative_energy)), np.pi/2))

#     # define accuracy volume as the absolute energy difference between the FCI energy and the energy of the solution normalized per electron
#     _accuracy_volume = np.divide(np.absolute(np.subtract( np.array(energy), fci_energy)), num_electrons)

#     return _solution_quality, _accuracy_volume

# # # if main, execute method
if __name__ == '__main__': run( )

# # # %%

# # run()