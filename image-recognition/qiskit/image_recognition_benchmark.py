"""
Image Recognition Benchmark Program - Qiskit
"""

import datetime
import json
import logging
import os
import re
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import sampled_expectation_value

# machine learning libraries
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics import accuracy_score, mean_squared_error
from noisyopt import minimizeSPSA

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit", '../../image-recognition/_common/']

# benchmark-specific imports
import execute as ex
import metrics as metrics
import image_recognition_metrics as img_metrics

# DEVNOTE: this logging feature should be moved to common level
logger = logging.getLogger(__name__)
fname, _, ext = os.path.basename(__file__).partition(".")
log_to_file = False

# supress deprecation warnings
# TODO update warning filters
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# set path for saving the thetas_array
thetas_array_path = "../_common/instances/"

try:
    if log_to_file:
        logging.basicConfig(
            # filename=f"{fname}_{datetime.datetime.now().strftime('%Y_%m_%d_%S')}.log",
            filename=f"{fname}.log",
            filemode="w",
            encoding="utf-8",
            level=logging.INFO,
            format="%(asctime)s %(name)s - %(levelname)s:%(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s %(name)s - %(levelname)s:%(message)s')
        
except Exception as e:
    print(f"Exception {e} occured while configuring logger: bypassing logger config to prevent data loss")
    pass

# Image Recognition inputs  ( Here input is Hamiltonian matrix --- Need to change)
hl_inputs = dict()  # inputs to the run method
verbose = False
print_sample_circuit = True

# Indicates whether to perform the (expensive) pre compute of expectations
do_compute_expectation = True

# Array of energy values collected during iterations of VQE
#lowest_energy_values = []

# Key metrics collected on last iteration of VQE
#key_metrics = {}

# saved circuits for display
QC_ = None
Uf_ = None

# DEBUG prints
# give argument to the python script as "debug" or "true" or "1" to enable debug prints
if len(sys.argv) > 1:
    DEBUG = sys.argv[1].lower() in ["debug", "true", "1"]
else:
    DEBUG = False

# Add custom metric names to metrics module
def add_custom_metric_names():
    metrics.known_x_labels.update(
        {
            "iteration_count": "Iterations"
        }
    )
    metrics.known_score_labels.update(
        {
            "train_accuracy" : "Training Accuracy",
            "test_accuracy" : "Test Accuracy",
            "train_loss" : "Training Loss",
            "solution_quality": "Solution Quality",
            "accuracy_volume": "Accuracy Volume",
            "accuracy_ratio": "Accuracy Ratio",
            "energy": "Energy (Hartree)",
            "standard_error": "Std Error",
        }
    )
    metrics.score_label_save_str.update(
        {
            "train_accuracy" : "train_accuracy",
            "test_accuracy" : "test_accuracy",
            "train_loss" : "train_loss",
            "solution_quality": "solution_quality",
            "accuracy_volume": "accuracy_volume",
            "accuracy_ratio": "accuracy_ratio",
            "energy": "energy",
        }
    )

###################################
# fetch mnist data

def fetch_mnist_data(int1=7, int2=9, test_size=50, train_size=200,
            random_state=42, verbose=False, normalize=True):
    
    if verbose:
        print(f"... fetching MNIST data, train_size={train_size}, test_size={test_size}")
        
    # Load the image data from MNIST database
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    # x has all the pixel values of image and has Data shape of (70000, 784)  here 784 is 28*28
    x = mnist.data
    
    # y has all the labels of the images and has Target shape of (70000,)
    y = mnist.target
    
    if verbose:
        print(f"    shape of x, y: {x.shape}, {y.shape}")

    # convert the given y data to integers so that we can filter the data
    y = y.astype(int)

    # Filtering only values with 7 or 9 as we are doing binary classification for now and we will extend it to multi-class classification
    binary_filter = (y == int1) | (y == int2)

    # Filtering the x and y data with the binary filter
    x = x[binary_filter]
    y = y[binary_filter]

    # create a new y data with 0 and 1 values with 7 as 0 and 9 as 1 so that we can use it for binary classification
    y = (y == int2).astype(int)

    # split the data into train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, train_size=train_size, random_state=random_state)
    
    if verbose:
        print(f"    sizes: x_train={len(x_train)}, x_test={len(x_test)}, y_train={len(y_train)}, y_test={len(y_test)}")
    
    return x, x_train, x_test, y, y_train, y_test

def preprocess_image_data(x, x_train, x_test, num_qubits, norm=True):

    if verbose:
        print(f"... preprocessing image data for num_qubits={num_qubits}") 
        
    # normalize the data between 0 and 1
    if norm:
        if verbose:
            print("  ... normalizing data")
            
        # normalize the data between 0 and 1
        x = normalize(x, norm='max', axis=1)
        x_train = normalize(x_train, norm='max', axis=1)
        x_test = normalize(x_test, norm='max', axis=1)
        
    # apply pca
    pca = PCA(n_components=num_qubits).fit(x)
    x_pca = pca.transform(x)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    # To visualize if prinicipal components are enough and to decide the number of principal components to keep 
    pca_check = False 
    if pca_check == True:
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.show()

    #scale the data between 0 and pi
    scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(x_pca)
    x_scaled = scaler.transform(x_pca)
    x_train_scaled = scaler.transform(x_train_pca)
    x_test_scaled = scaler.transform(x_test_pca)

    if verbose:
        print(f"    scaled sizes: x_scaled={len(x_scaled)}, x_train_scaled={len(x_train_scaled)}, x_test_scaled={len(x_test_scaled)}")
        
    return x_scaled, x_train_scaled, x_test_scaled


###################################
# Data handling methods

def read_dict_from_json(instance_filepath):
    """
    Read a dictionary from a json file.
    """
    try:
        with open(instance_filepath) as f:
            instance_dict = json.load(f)
    except Exception as e:
        print(f"Exception {e} occured while reading instance file {instance_filepath}")
        logger.error(e)
        instance_dict = None
    return instance_dict

def write_dict_to_json(instance_dict, instance_filepath):
    """
    Write a dictionary to a json file.
    """
    try:
        with open(instance_filepath, "w") as f:
            json.dump(instance_dict, f, indent='\t')
    except Exception as e:
        print(f"Exception {e} occured while writing instance file {instance_filepath}")
        logger.error(e)

def update_dict_in_json(instance_dict, instance_filepath):
    """
    Update a dictionary in a json file.
    """
    try:
        if not os.path.exists(instance_filepath):
            write_dict_to_json(instance_dict, instance_filepath)
            return
        with open(instance_filepath, "r+") as f:
            data = json.load(f)
            data.update(instance_dict)
            f.seek(0)
            json.dump(data, f)
    except Exception as e:
        print(f"Exception {e} occured while updating instance file {instance_filepath}")
        logger.error(e)


###################################
# Image Recognition CIRCUIT

# parameter mode to control length of initial thetas_array (global during dev phase)
# 1 - length 1
# 2 - length N, where N is number of excitation pairs
saved_parameter_mode = 1

def get_initial_parameters(num_qubits: int, ansatz_type:str='qcnn uniform', thetas_array=None, reps=1, verbose=False):
    """
    Generate an initial set of parameters given the number of qubits and user-provided thetas_array.
    If thetas_array is None, generate random array of parameters based on the parameter_mode
    If thetas_array is of length 1, repeat to the required length based on parameter_mode
    Otherwise, use the provided thetas_array.

    Parameters
    ----------
    num_qubits : int
        number of qubits in circuit
    thetas_array : array of floats
        user-supplied array of initial values

    Returns
    -------
    initial_parameters : array
        array of parameter values required
    """

    # calculate number of parameters based on type of ansatz
    # Initialize the weights for the QNN model
    if ansatz_type == 'block':
        num_parameters= num_qubits * reps * 2
    elif ansatz_type == 'qcnn uniform':
        num_parameters_per_layer=15*reps
        num_layers = int(np.ceil(np.log2(num_qubits)))
        num_parameters = num_parameters_per_layer*num_layers
    elif ansatz_type == 'qcnn unique':
        num_parameters_per_conv=15*reps
        num_parameters = num_parameters_per_conv*(2*num_qubits-2-int(np.ceil(np.log2(num_qubits))))
    else:
        print("Invalid ansatz_type " + str(ansatz_type))

    
    size = num_parameters

    # if None passed in, create array of random values
    if thetas_array is None:
        initial_parameters = np.random.random(size=size)

    # if single value passed in, extend to required size
    elif size > 1 and len(thetas_array) == 1:
        initial_parameters = np.repeat(thetas_array, size)

    # otherwise, use what user provided
    else:
        if len(thetas_array) != size:
            print(f"WARNING: length of thetas_array {len(thetas_array)} does not equal required length {size}")
            print("         Generating random values instead.")
            initial_parameters = get_initial_parameters(num_qubits, None)
        else:
            initial_parameters = np.array(thetas_array)

    if verbose:
        print(f"... get_initial_parameters(num_qubits={num_qubits}, mode={saved_parameter_mode})")
        print(f"    --> initial_parameter[{size}]={initial_parameters}")

    return initial_parameters

# create the qcnn ansatz
# Variational circuit ansatzes
def parametrized_block(thetas, num_qubits, reps):
    
    # reps is Number of times ry and cx gates are repeated
    qc = QuantumCircuit(num_qubits)    
    # print("parameter_vector",parameter_vector)
    counter = 0
    for rep in range(reps):
        for i in range(num_qubits):
            theta = thetas[counter]
            qc.ry(theta, i)
            counter += 1
        
        for i in range(num_qubits):
            theta = thetas[counter]
            qc.rx(theta, i)
            counter += 1
    
        for j in range(0, num_qubits - 1, 1):
            if rep<reps-1:
                qc.cx(j, j + 1)
    # print("counter",counter)
    return qc

# Ansatz from paper https://arxiv.org/pdf/2108.00661.pdf
def parameterized_2q_gate_1(thetas, num_reps=1):
    # Your implementation for conv_circ_1 function here
    # print(thetas)
    conv_circ = QuantumCircuit(2)

    for i in range(num_reps):
        conv_circ.rx(thetas[12*i+0], 0)
        conv_circ.rx(thetas[12*i+1], 1)
        conv_circ.rz(thetas[12*i+2], 0)
        conv_circ.rz(thetas[12*i+3], 1)


        conv_circ.crx(thetas[12*i+4], 0, 1)  
        conv_circ.crx(thetas[12*i+5], 1, 0)

        conv_circ.rx(thetas[12*i+6], 0)
        conv_circ.rx(thetas[12*i+7], 1)
        conv_circ.rz(thetas[12*i+8], 0)
        conv_circ.rz(thetas[12*i+9], 1)

        conv_circ.crz(thetas[12*i+10], 1, 0)  
        conv_circ.x(1)
        conv_circ.crx(thetas[12*i+11], 1, 0)
        conv_circ.x(1)

    #conv_circ = QuantumCircuit(2)
    #conv_circ.crx(thetas[0], 0, 1)

    # print(conv_circ)
    return conv_circ

#Most general two qubit gate ansatz
def parameterized_2q_gate_2(thetas, num_reps=1):

    conv_circ = QuantumCircuit(2)

    for i in range(num_reps):
        conv_circ.rx(thetas[15*i+0], 0)
        conv_circ.rz(thetas[15*i+1], 0)
        conv_circ.rx(thetas[15*i+2], 0)
    
        conv_circ.rx(thetas[15*i+3], 1)
        conv_circ.rz(thetas[15*i+4], 1)
        conv_circ.rx(thetas[15*i+5], 1)
    
        conv_circ.cx(1, 0)
        conv_circ.rz(thetas[15*i+6], 0)
        conv_circ.ry(thetas[15*i+7], 1)
        conv_circ.cx(0, 1)
        conv_circ.ry(thetas[15*i+8], 1)
        conv_circ.cx(1, 0)
    
        conv_circ.rx(thetas[15*i+9], 0)
        conv_circ.rz(thetas[15*i+10], 0)
        conv_circ.rx(thetas[15*i+11], 0)
    
        conv_circ.rx(thetas[15*i+12], 1)
        conv_circ.rz(thetas[15*i+13], 1)
        conv_circ.rx(thetas[15*i+14], 1)

    return conv_circ

def ansatz(ansatz_type,num_qubits, num_reps=1):

    qc = QuantumCircuit(num_qubits) 

    if ansatz_type == "block":
        parameter_vector = ParameterVector("t", length=num_qubits*num_reps*2)
        qc=qc.compose(parametrized_block(parameter_vector,num_qubits, num_reps), qubits=range(num_qubits))
    elif ansatz_type == "qcnn uniform":
        num_layers = int(np.ceil(np.log2(num_qubits)))
        num_parameters_per_layer = 15 * num_reps
        num_parameters=num_layers*num_parameters_per_layer
        parameter_vector = ParameterVector("t", length=num_parameters)  
        qc = QuantumCircuit(num_qubits)  
        for i_layer in range(num_layers):
            for i_sub_layer in [0 , 2**i_layer]:            
                for i_q1 in range(i_sub_layer, num_qubits, 2**(i_layer+1)):
                    i_q2=2**i_layer+i_q1
                    if i_q2<num_qubits:
                        qc=qc.compose(parameterized_2q_gate_2(parameter_vector[num_parameters_per_layer*i_layer:num_parameters_per_layer*(i_layer+1)], num_reps=num_reps), qubits=(i_q1,i_q2))  
                        #print("i_q1",i_q1,"i_q2",i_q2)
    elif ansatz_type == "qcnn unique":
        num_layers = int(np.ceil(np.log2(num_qubits)))
        num_parameters_per_conv = 15 * num_reps
        parameter_vector = ParameterVector("t", length=0)  
        qc = QuantumCircuit(num_qubits)  
        i_conv=0
        for i_layer in range(num_layers):
            for i_sub_layer in [0 , 2**i_layer]:            
                for i_q1 in range(i_sub_layer, num_qubits, 2**(i_layer+1)):
                    i_q2=2**i_layer+i_q1
                    if i_q2<num_qubits:
                        parameter_vector.resize((i_conv+1)*num_parameters_per_conv)
                        qc=qc.compose(parameterized_2q_gate_2(parameter_vector[num_parameters_per_conv*i_conv:num_parameters_per_conv*(i_conv+1)], num_reps=num_reps), qubits=(i_q1,i_q2)) 
                        i_conv+=1
    else:
        print("ansatz_type not recognized")
    
    #print(qc)
    return qc, parameter_vector


def feature_map(num_qubits = 8, x_data=None):
    qc = QuantumCircuit(num_qubits)

    # create feature map
    for i in range(num_qubits):
        qc.ry(x_data[i], i)

    return qc

def prepare_circuit(base_circuit: QuantumCircuit, total_operator=None):
    """
    Return the circuit and operator to be used in the image recognition. 

    base_circuit has diagonalizing gates appended to it and is returned in that state to be measured. 

    total_operator is returned is if it not None, otherwise return a default operator. 
    """
    num_qubits = base_circuit.num_qubits

    # Define the default operator if none provided
    z_operator = SparsePauliOp("I" * (num_qubits-1) + "Z")

    if total_operator is None:
        total_operator = z_operator

    circuits = []

    for pauli_label, _ in total_operator.to_list():
        circuit = base_circuit.copy()

        # Apply gates that diagonalize the pauli string with respect to the computational (Z) basis. 
        # This is done as later in the code we sample from the circuit to find the expectation value.
        # Go in reverse to account for Qiskit ordering- most significant qubit is on the left
        for i, pauli_char in enumerate(pauli_label[::-1]):
            if pauli_char == 'X':
                circuit.h(i)
            elif pauli_char == 'Y':
                circuit.sdg(i)
                circuit.h(i)
            elif pauli_char == 'Z':
                # Z -> Z, no operation needed for diagonalization
                continue
            elif pauli_char == 'I':
                # I -> I, no operation needed
                continue
            else:
                raise ValueError("Pauli string contains character that is not I, X, Y, Z.")

        circuit.measure_all()  # Add measurement to all qubits
        circuits.append(circuit)

    return circuit, total_operator

def ImageRecognition(num_qubits: int,
                    thetas_array,
                    parameterized,
                    ansatz_type: str = "qcnn uniform",
                    x_data=None,
                    reps: int = 1,
                    verbose: bool = False,
                    *args, **kwargs) -> QuantumCircuit:
    """
    Create the ansatz quantum circuit for the QCNN algorithm.

    Parameters
    ----------
    num_qubits : int
        number of qubits in circuit

    thetas_array : array of floats
        array of parameters to be bound to circuit

    parameterized : bool

    verbose : bool
        verbose flag
    """
    if verbose:
        print(f"... ImageRecognition_ansatz(num_qubits={num_qubits}, ansatz_type={ansatz_type}, thetas_array={thetas_array}")

    # create the feature map circuit
    _feature_map_circuit = feature_map(num_qubits, x_data)

    # create the ansatz circuit
    _variational_circuit, _parameter_vector = ansatz(ansatz_type, num_qubits, reps)

    # bind the parameters to the variational circuit
    _variational_circuit.assign_parameters(thetas_array, inplace=True)

    # compose the circuits
    _circuit = _feature_map_circuit.compose(_variational_circuit)

    # prepare the circuit for execution
    qc, observables = prepare_circuit(_circuit)

    params = {_parameter_vector: thetas_array} if parameterized else None

    # save small circuit example for display
    global QC_
    if QC_ is None or num_qubits <= 4:
        if num_qubits <= 8:
            QC_ = qc
            
    return qc, observables, params


def loss_function(result, y_data, num_qubits, formatted_observables, verbose=False):
    """
    Compute the expectation value of the circuit with respect to the Hamiltonian for optimization
    """

    # calculate probabilities from the result count
    pass

    _predictions = list()
    _prediction_labels = list()

    for _res in result:
        _counts = _res.get_counts()
        _probs = normalize_counts(_counts, num_qubits=num_qubits)

        _expectation_values = calculate_diagonalized_expectation_values(_probs, formatted_observables)
        value = sum(_expectation_values)

        # # now get <H^2>, assuming Cov[si,si'] = 0
        # formatted_observables_sq = [(obs @ obs).simplify(atol=0) for obs in formatted_observables]
        # _expectation_values_sq = calculate_expectation_values(_probabilities, formatted_observables_sq)

        # # now since Cov is assumed to be zero, we compute each term's variance and sum the result.
        # # see Eq 5, e.g. in https://arxiv.org/abs/2004.06252
        # variance = sum([exp_sq - exp**2 for exp_sq, exp in zip(_expectation_values_sq, _expectation_values)])

        value = (value + 1)*0.5
        _predictions.append(value)

        # calculate accuracy
        
        if value > 0.5:
            _prediction_labels.append(1)
        else:
            _prediction_labels.append(0)

    accuracy = accuracy_score(y_data, _prediction_labels)
    
    loss = mean_squared_error(y_data, _predictions)
    
    return loss, accuracy


##########################################################

def post_diagonalization_op_converter(op: SparsePauliOp):
    """
    Return a pauli string with X and Y replaced with Z. This is useful if we have diaganalized that pauli string  to the Z basis and now need to pass that into the sampled_expectation_value function.
    """

    # Get the labels from the original SparsePauliOp
    labels = [label for label, coeff in op.to_list()]

    # Replace 'X' and 'Y' with 'Z' in each label
    new_labels = ["".join("Z" if p in "XY" else p for p in label) for label in labels]

    # Create a new SparsePauliOp with the modified labels
    new_pauli_op = SparsePauliOp(new_labels, coeffs=op.coeffs)

    return new_pauli_op


def calculate_diagonalized_expectation_values(probabilities, observables):
    """
    Return the expectation values for an operator given the probabilities.

    Note that this function assumes the probabilities are from a post-diagonalized (to the computational Z basis) circuit. 
    """
    if not isinstance(probabilities, list):
        probabilities = [probabilities]
    expectation_values = list()
    for idx, op in enumerate(observables):

        expectation_value = sampled_expectation_value(
            probabilities[idx], post_diagonalization_op_converter(op)
        )
        expectation_values.append(expectation_value)

    return expectation_values

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

############### Prepare Circuits for Execution

def get_operator_for_problem(instance_filepath):
    """
    Return an operator object for the problem.

    Argument:
    instance_filepath : problem defined by instance_filepath (but should be num_qubits + radius)

    Returns:
    operator : an object encapsulating the Hamiltoian for the problem
    """
    # operator is paired hamiltonian  for each instance in the loop
    ops, coefs = common.read_paired_instance(instance_filepath)
    operator = SparsePauliOp.from_list(list(zip(ops, coefs)))
    return operator

# A dictionary of random energy filename, obtained once
random_energies_dict = None
 
def get_random_energy(instance_filepath):
    """
    Get the 'random_energy' associated with the problem
    """

    # read precomputed energies file from the random energies path
    global random_energies_dict

    # get the list of random energy filenames once
    # (DEVNOTE: probably don't want an exception here, should just return 0)
    if not random_energies_dict:
        try:
            random_energies_dict = common.get_random_energies_dict()
        except ValueError as err:
            logger.error(err)
            print("Error reading precomputed random energies json file. Please create the file by running the script 'compute_random_energies.py' in the _common/random_sampler directory")
            raise

    # get the filename from the instance_filepath and get the random energy from the dictionary
    filename = os.path.basename(instance_filepath)
    filename = filename.split(".")[0]
    random_energy = random_energies_dict[filename]

    return random_energy
    
def get_classical_solutions(instance_filepath):
    """
    Get a list of the classical solutions for this problem
    """
    # solution has list of classical solutions for each instance in the loop
    sol_file_name = instance_filepath[:-5] + ".sol"
    method_names, values = common.read_puccd_solution(sol_file_name)
    solution = list(zip(method_names, values))
    return solution

# Return the file identifier (path) for the problem at this width, radius, and instance
def get_problem_identifier(num_qubits, radius, instance_num):
    
    # if radius is given we should do same radius for max_circuits times
    if radius is not None:
        try:
            instance_filepath = common.get_instance_filepaths(num_qubits, radius)
        except ValueError as err:
            logger.error(err)
            instance_filepath = None

    # if radius is not given we should do all the radius for max_circuits times
    else:
        instance_filepath_list = common.get_instance_filepaths(num_qubits)
        try:
            if len(instance_filepath_list) >= instance_num:
                instance_filepath = instance_filepath_list[instance_num - 1]
            else:
                instance_filepath = None
        except ValueError as err:
            logger.error(err)
            instance_filepath = None

    return instance_filepath


#################################################
# EXPECTED RESULT TABLES (METHOD 1)

############### Expectation Tables Created using State Vector Simulator

# DEVNOTE: We are building these tables on-demand for now, but for larger circuits
# this will need to be pre-computed ahead of time and stored in a data file to avoid run-time delays.

# dictionary used to store pre-computed expectations, keyed by num_qubits and secret_string
# these are created at the time the circuit is created, then deleted when results are processed
expectations = {}


# Compute array of expectation values in range 0.0 to 1.0
# Use statevector_simulator to obtain exact expectation
def compute_expectation(qc, num_qubits, secret_int, backend_id="statevector_simulator", params=None):
    # ts = time.time()

    # to execute on Aer state vector simulator, need to remove measurements
    qc = qc.remove_final_measurements(inplace=False)

    if params is not None:
        qc = qc.assign_parameters(params)

    # execute statevector simulation
    sv_backend = Aer.get_backend(backend_id)
    sv_result = sv_backend.run(qc, params=params).result()

    # get the probability distribution
    counts = sv_result.get_counts()

    # print(f"... statevector expectation = {counts}")

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

        # scale probabilities to number of shots to obtain counts
        for k, v in counts.items():
            counts[k] = round(v * num_shots)

        # delete from the dictionary
        del expectations[id]

        return counts

    else:
        return None


#################################################
# RESULT DATA ANALYSIS (METHOD 1)

expected_dist = {}


# Compare the measurement results obtained with the expected measurements to determine fidelity
def analyze_and_print_result(qc, result, num_qubits, secret_int, num_shots):
    global expected_dist

    # obtain counts from the result object
    counts = result.get_counts(qc)

    # retrieve pre-computed expectation values for the circuit that just completed
    expected_dist = get_expectation(num_qubits, secret_int, num_shots)

    # if the expectation is not being calculated (only need if we want to compute fidelity)
    # assume that the expectation is the same as measured counts, yielding fidelity = 1
    if expected_dist is None:
        expected_dist = counts

    if verbose:
        print(f"For width {num_qubits}   measured: {counts}\n  expected: {expected_dist}")
    # if verbose: print(f"For width {num_qubits} problem {secret_int}\n  measured: {counts}\n  expected: {expected_dist}")

    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, expected_dist)

    # if verbose: print(f"For secret int {secret_int} fidelity: {fidelity}")

    return counts, fidelity

##### METHOD 2 function to compute application-specific figures of merit

def calculate_quality_metric(energy=None,
            fci_energy=0,
            random_energy=0, 
            precision = 4,
            num_electrons = 2):
    """
    Returns the quality metrics, namely solution quality, accuracy volume, and accuracy ratio.
    Solution quality is a value between zero and one. The other two metrics can take any value.

    Parameters
    ----------
    energy : list
        list of energies calculated for each iteration.

    fci_energy : float
        FCI energy for the problem.

    random_energy : float
        Random energy for the problem, precomputed and stored and read from json file

    precision : float
        precision factor used in solution quality calculation
        changes the behavior of the monotonic arctan function

    num_electrons : int
        number of electrons in the problem
    """
    
    _delta_energy_fci = np.absolute(np.subtract( np.array(energy), fci_energy))
    _delta_random_fci = np.absolute(np.subtract( np.array(random_energy), fci_energy))
    
    _relative_energy = np.absolute(
                            np.divide(
                                np.subtract( np.array(energy), fci_energy),
                                fci_energy)
                            )
    
    
    #scale the solution quality to 0 to 1 using arctan 
    _solution_quality = np.subtract(
                            1,
                            np.divide(
                                np.arctan(
                                    np.multiply(precision,_relative_energy)
                                    ),
                                np.pi/2)
                            )

    # define accuracy volume as the absolute energy difference between the FCI energy and the energy of the solution normalized per electron
    _accuracy_volume = np.divide(
                            np.absolute(
                                np.subtract( np.array(energy), fci_energy)
                            ),
                            num_electrons
                            )

    # define accuracy ratio as 1.0 minus the error in energy over the error in random energy:
    #       accuracy_ratio = 1.0 - abs(energy - FCI) ) / abs(random - FCI)

    _accuracy_ratio = np.subtract(1.0, np.divide(_delta_energy_fci,_delta_random_fci))
    
    return _solution_quality, _accuracy_volume, _accuracy_ratio


#################################################
# DATA SAVE FUNCTIONS

# Create a folder where the results will be saved.
# For every circuit width, metrics will be stored the moment the results are obtained
# In addition to the metrics, the parameter values obtained by the optimizer, as well as the counts
# measured for the final circuit will be stored.
def create_data_folder(save_res_to_file, detailed_save_names, backend_id):
    global parent_folder_save

    # if detailed filenames requested, use directory name with timestamp
    if detailed_save_names:
        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        parent_folder_save = os.path.join("__data", f"{backend_id}", f"run_start_{start_time_str}")

    # otherwise, just put all json files under __data/backend_id
    else:
        parent_folder_save = os.path.join("__data", f"{backend_id}")

    # create the folder if it doesn't exist already
    if save_res_to_file and not os.path.exists(parent_folder_save):
        os.makedirs(os.path.join(parent_folder_save))
        
# Function to save final iteration data to file
def store_final_iter_to_metrics_json(
    backend_id,
    num_qubits,
    radius,
    instance_num,
    num_shots,
    converged_thetas_list,
    accuracy,
    detailed_save_names,
    dict_of_inputs,
    save_final_counts,
    save_res_to_file,
    _instances=None,
):
    """
    For a given problem (specified by num_qubits and instance),
    1. For a given restart, store properties of the final minimizer iteration to metrics.circuit_metrics_final_iter, and
    2. Store various properties for all minimizer iterations for each restart to a json file.
    """
    # In order to compare with uniform random sampling, get some samples

    # Store properties of the final iteration, the converged theta values,
    # as well as the known optimal value for the current problem,
    # in metrics.circuit_metrics_final_iter. 
    metrics.store_props_final_iter(num_qubits, instance_num, "train_accuracy", accuracy)
    metrics.store_props_final_iter(num_qubits, instance_num, "converged_thetas_list", converged_thetas_list)

    # Save final iteration data to metrics.circuit_metrics_final_iter
    # This data includes final counts, cuts, etc.
    if save_res_to_file:
    
        # Save data to a json file
        dump_to_json(
            parent_folder_save,
            num_qubits,
            radius,
            instance_num,
            dict_of_inputs,
            converged_thetas_list,
            accuracy,
            save_final_counts=save_final_counts,
        )

def dump_to_json(
    parent_folder_save,
    num_qubits,
    radius,
    instance_num,
    dict_of_inputs,
    converged_thetas_list,
    accuracy,
    save_final_counts=False,
):
    """
    For a given problem (specified by number of qubits and instance_number),
    save the evolution of various properties in a json file.
    Items stored in the json file: Data from all iterations (iterations), inputs to run program ('general properties'), converged theta values ('converged_thetas_list'), computes results.
    if save_final_counts is True, then also store the distribution counts
    """

    # print(f"... saving data for width={num_qubits} radius={radius} instance={instance_num}")
    if not os.path.exists(parent_folder_save):
        os.makedirs(parent_folder_save)
    store_loc = os.path.join(parent_folder_save, "width_{}_instance_{}.json".format(num_qubits, instance_num))

    # Obtain dictionary with iterations data corresponding to given instance_num
    all_restart_ids = list(metrics.circuit_metrics[str(num_qubits)].keys())
    ids_this_restart = [r_id for r_id in all_restart_ids if int(r_id) // 1000 == instance_num]
    iterations_dict_this_restart = {r_id: metrics.circuit_metrics[str(num_qubits)][r_id] for r_id in ids_this_restart}

    # Values to be stored in json file
    dict_to_store = {"iterations": iterations_dict_this_restart}
    dict_to_store["general_properties"] = dict_of_inputs
    dict_to_store["converged_thetas_list"] = converged_thetas_list
    dict_to_store["train_accuracy"] = accuracy
    # dict_to_store['unif_dict'] = unif_dict

    # Also store the value of counts obtained for the final counts
    """
    if save_final_counts:
        dict_to_store['final_counts'] = iter_dist
        #iter_dist.get_counts()
    """

    # Now write the data fo;e
    with open(store_loc, "w") as outfile:
        json.dump(dict_to_store, outfile)

#################################################
# DATA LOAD FUNCTIONS

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
    if _gen_prop is not None:
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
    # print(list_of_files)

    # list with elements that are tuples->(width,restartInd,filename)
    width_restart_file_tuples = [
        (*get_width_restart_tuple_from_filename(fileName), fileName)
        for (ind, fileName) in enumerate(list_of_files)
        if fileName.startswith("width")
    ]

    # sort first by width, and then by restartInd
    width_restart_file_tuples = sorted(width_restart_file_tuples, key=lambda x: (x[0], x[1]))
    distinct_widths = list(set(it[0] for it in width_restart_file_tuples))
    list_of_files = [[tup[2] for tup in width_restart_file_tuples if tup[0] == width] for width in distinct_widths]

    # connot continue without at least one dataset
    if len(list_of_files) < 1:
        print("ERROR: No result files found")
        return None

    for width_files in list_of_files:
        # For each width, first load all the restart files
        for fileName in width_files:
            gen_prop = load_from_width_restart_file(folder, fileName)

        # next, do processing for the width
        method = gen_prop["method"]
        if method == 2:
            num_qubits, _ = get_width_restart_tuple_from_filename(width_files[0])
            metrics.process_circuit_metrics_2_level(num_qubits)
            metrics.finalize_group(num_qubits)

    # override device name with the backend_id if supplied by caller
    if backend_id is not None:
        metrics.set_plot_subtitle(f"Device = {backend_id}")

    return gen_prop


# # load data from a specific file

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
    with open(os.path.join(folder, fileName), "r") as json_file:
        data = json.load(json_file)
        gen_prop = data["general_properties"]
        converged_thetas_list = data["converged_thetas_list"]
        energy = data["energy"]
        if gen_prop["save_final_counts"]:
            # Distribution of measured cuts
            final_counts = data["final_counts"]

        backend_id = gen_prop.get("backend_id")
        metrics.set_plot_subtitle(f"Device = {backend_id}")

        # Update circuit metrics
        for circuit_id in data["iterations"]:
            # circuit_id = restart_ind * 1000 + minimizer_loop_ind
            for metric, value in data["iterations"][circuit_id].items():
                metrics.store_metric(num_qubits, circuit_id, metric, value)

        method = gen_prop["method"]
        if method == 2:
            metrics.store_props_final_iter(num_qubits, restart_ind, "energy", energy)
            metrics.store_props_final_iter(num_qubits, restart_ind, "converged_thetas_list", converged_thetas_list)
            if gen_prop["save_final_counts"]:
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
    pattern = "width_([0-9]+)_instance_([0-9]+).json"
    match = re.search(pattern, fileName)
    # print(match)

    # assert match is not None, f"File {fileName} found inside folder. All files inside specified folder must be named in the format 'width_int_restartInd_int.json"
    num_qubits = int(match.groups()[0])
    degree = int(match.groups()[1])
    return (num_qubits, degree)


################################################
# PLOT METHODS

def plot_results_from_data(
    num_shots=100,
    radius=0.75,
    max_iter=30,
    max_circuits=1,
    method=2,
    line_x_metrics=["iteration_count", "cumulative_exec_time"],
    line_y_metrics=["train_loss", "train_accuracy", "test_accuracy"],
    plot_layout_style="grid",
    bar_y_metrics=["average_exec_times", "accuracy_ratio_error"],
    bar_x_metrics=["num_qubits"],
    show_elapsed_times=True,
    use_logscale_for_times=False,
    score_metric=["accuracy_ratio"],
    y_metric=["num_qubits"],
    x_metric=["cumulative_exec_time", "cumulative_elapsed_time"],
    fixed_metrics={},
    num_x_bins=15,
    y_size=None,
    x_size=None,
    x_min=None,
    x_max=None,
    detailed_save_names=False,
    ansatz_type:str = 'qcnn uniform',
    **kwargs,
):
    """
    Plot results from the data contained in metrics tables.
    """

    # Add custom metric names to metrics module (in case this is run outside of run())
    add_custom_metric_names()

    # handle single string form of score metrics
    if type(score_metric) == str:
        score_metric = [score_metric]

    # for Image Recognition, objective function is always 'Energy'
    obj_str = "Energy"

    suffix = ""

    # If detailed names are desired for saving plots, put date of creation, etc.
    if detailed_save_names:
        cur_time = datetime.datetime.now()
        dt = cur_time.strftime("%Y-%m-%d_%H-%M-%S")
        suffix = f"s{num_shots}_r{radius}_mi{max_iter}_{dt}"

    suptitle = f"Benchmark Results - Image Recognition ({method}) - Qiskit"
    backend_id = metrics.get_backend_id()
    options = {"shots": num_shots, "radius": radius, "restarts": max_circuits}


    # plot all line metrics, including solution quality and accuracy ratio
    # vs iteration count and cumulative execution time
    img_metrics.plot_all_line_metrics(
        suptitle,
        line_x_metrics=line_x_metrics,
        line_y_metrics=line_y_metrics,
        plot_layout_style=plot_layout_style,
        backend_id=backend_id,
        options=options,
        method=method,
        ansatz_type=ansatz_type,
    )

    # plot all cumulative metrics, including average_execution_time and accuracy ratio
    # over number of qubits
    img_metrics.plot_all_cumulative_metrics(
        suptitle,
        bar_y_metrics=bar_y_metrics,
        bar_x_metrics=bar_x_metrics,
        show_elapsed_times=show_elapsed_times,
        use_logscale_for_times=use_logscale_for_times,
        plot_layout_style=plot_layout_style,
        backend_id=backend_id,
        options=options,
        method=method,
    )

    # # plot all area metrics
    # metrics.plot_all_area_metrics(
    #     suptitle,
    #     score_metric=score_metric,
    #     x_metric=x_metric,
    #     y_metric=y_metric,
    #     fixed_metrics=fixed_metrics,
    #     num_x_bins=num_x_bins,
    #     x_size=x_size,
    #     y_size=y_size,
    #     x_min=x_min,
    #     x_max=x_max,
    #     options=options,
    #     suffix=suffix,
    #     which_metric="solution_quality",
    # )


################################################
################################################
# RUN METHOD

MAX_QUBITS = 16

def run(
    min_qubits=2, max_qubits=4, skip_qubits=2, max_circuits=3, num_shots=100,
    method=2,
    radius=None, thetas_array=None, parameterized=False, parameter_mode=1, do_fidelities=True,
    minimizer_function=None,
    minimizer_tolerance=1e-3, max_iter=300, comfort=False,
    line_x_metrics=["iteration_count", "cumulative_exec_time", "iteration_count"],
    line_y_metrics=["train_loss", "train_accuracy", "test_accuracy"],
    bar_y_metrics=["average_exec_times", "train_accuracy", "test_accuracy"],
    bar_x_metrics=["num_qubits"],
    score_metric=["train_accuracy"],
    x_metric=["cumulative_exec_time", "cumulative_elapsed_time"],
    y_metric="num_qubits",
    fixed_metrics={},
    num_x_bins=15,
    y_size=None,
    x_size=None,
    show_results_summary=True,
    plot_results=True,
    plot_layout_style="grid",
    show_elapsed_times=True,
    use_logscale_for_times=False,
    save_res_to_file=True, save_final_counts=False, detailed_save_names=False,
    backend_id="qasm_simulator",
    provider_backend=None, hub="ibm-q", group="open", project="main",
    exec_options=None,
    _instances=None,
    ansatz_type:str = 'qcnn uniform',
    reps:int = 1,
    batch_size:int = 50,
    backend_id_train:str = 'statevector_simulator',
    test_pass_count:int = 30,
    test_size:int  = 50,
    train_size:int = 200
):
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
        Whether to use parameter objects in circuits or not. The default is False.
    parameter_mode : bool, optional
        If 1, use thetas_array of length 1, otherwise (num_qubits//2)**2, to match excitation pairs
    do_fidelities : bool, optional
        Compute circuit fidelity. The default is True.
    minimizer_function : function
        custom function used for minimizer
    minimizer_tolerance : float
        tolerance for minimizer, default is 1e-3,
    max_iter : int, optional
        Number of iterations for the minimizer routine. The default is 30.
    plot_layout_style : str, optional
        Style of plot layout, 'grid', 'stacked', or 'individual', default = 'grid'
    line_x_metrics : list or string, optional
        Which metrics are to be plotted on x-axis in line metrics plots.
    line_y_metrics : list or string, optional
        Which metrics are to be plotted on y-axis in line metrics plots.
    show_elapsed_times : bool, optional
        In execution times bar chart, include elapsed times if True
    use_logscale_for_times : bool, optional
        In execution times bar plot, use a log scale to show data
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

    thetas = []  # a default empty list of thetas

    # Update the dictionary of inputs
    dict_of_inputs = {**dict_of_inputs, **{"thetas_array": thetas, "max_circuits": max_circuits}}

    # Delete some entries from the dictionary; they may contain secrets or function pointers
    for key in ["hub", "group", "project", "provider_backend", "exec_options", "minimizer_function"]:
        dict_of_inputs.pop(key)

    global image_recognition_inputs
    image_recognition_inputs = dict_of_inputs
    
    ###########################
    # Benchmark Initializeation

    global QC_
    global circuits_done
    global minimizer_loop_index
    global opt_ts
    global unique_id

    print("Image Recognition Benchmark Program - Qiskit")

    QC_ = None

    # validate parameters
    max_qubits = max(2, max_qubits)
    max_qubits = min(MAX_QUBITS, max_qubits)
    min_qubits = min(max(2, min_qubits), max_qubits)

    # try:
    #     print("Validating user inputs...")
    #     # raise an exception if either min_qubits or max_qubits is not even
    #     if min_qubits % 2 != 0 or max_qubits % 2 != 0:
    #         raise ValueError(
    #             "min_qubits and max_qubits must be even. min_qubits = {}, max_qubits = {}".format(
    #                 min_qubits, max_qubits
    #             )
    #         )
    # except ValueError as err:
    #     # display error message and stop execution if min_qubits or max_qubits is not even
    #     logger.error(err)
    #     if min_qubits % 2 != 0:
    #         min_qubits += 1
    #     if max_qubits % 2 != 0:
    #         max_qubits -= 1
    #         max_qubits = min(max_qubits, MAX_QUBITS)
    #     print(err.args[0] + "\n Running for for values min_qubits = {}, max_qubits = {}".format(min_qubits, max_qubits))

    # don't compute exectation unless fidelity is is needed
    global do_compute_expectation
    do_compute_expectation = do_fidelities

    # save the desired parameter mode globally (for now, during dev)
    global saved_parameter_mode
    saved_parameter_mode = parameter_mode

    # given that this benchmark does every other width, set y_size default to 1.5
    if y_size is None:
        y_size = 1.5

    # Initialize metrics module with empty metrics arrays
    metrics.init_metrics()

    # Add custom metric names to metrics module
    add_custom_metric_names()

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, s_int, num_shots):
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
        metrics.store_metric(num_qubits, s_int, "solution_quality", fidelity)

    def execution_handler2(qc, result, num_qubits, s_int, num_shots):
        # Stores the results to the global saved_result variable
        global saved_result
        saved_result = result

    # Initialize execution module using the execution result handler above and specified backend_id
    # for method=2 we need to set max_jobs_active to 1, so each circuit completes before continuing
    if method == 2 or method == 3:
        ex.max_jobs_active = 1
        ex.init_execution(execution_handler2)
    else:
        ex.init_execution(execution_handler)

    # initialize the execution module with target information
    ex.set_execution_target(
        backend_id_train, provider_backend=provider_backend, hub=hub, group=group, project=project, exec_options=exec_options
    )

    # create a data folder for the results
    create_data_folder(save_res_to_file, detailed_save_names, backend_id)

    ###########################
    # Benchmark Execution Loop

    # Fetch MNIST data
    x, x_train, x_test, y, y_train, y_test = fetch_mnist_data(
            test_size=test_size, train_size=train_size, verbose=verbose)

    # dictionary to store the thetas_array for each qubit size
    thetas_array_dict = {}
    
    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    # DEVNOTE: increment by 2 for efficiency

    for num_qubits in range(min_qubits, max_qubits + 1, 2):

        np.random.seed(0)

        if method == 1:
            print(f"************\nExecuting [1] circuit for num_qubits = {num_qubits}")
        else:
            print(f"************\nExecuting [{ansatz_type}] circuit for num_qubits = {num_qubits}")

        # loop over all instance files according to max_circuits given
        # instance_num index starts from 1
        # for instance_num in range(1, max_circuits + 1):

        x_scaled, x_train_scaled, x_test_scaled = preprocess_image_data(x, x_train, x_test, num_qubits)        

        global x_batch, y_batch
        
        instance_num = 1

        # global variables to store execution and elapsed time
        global quantum_execution_time, quantum_elapsed_time
        quantum_execution_time = 0.0
        quantum_elapsed_time = 0.0   
         
        #####################
        # define the objective and the callback functions
        # NOTE: this is called twice for the SPSA optimizer
        def objective_function(thetas_array, return_accuracy=False,
                    test_pass=False, train_pass=False):
            """
            Objective function that calculates the expected energy for the given parameterized circuit

            Parameters
            ----------
            thetas_array : list
                list of theta values.
            """

            # Every circuit needs a unique id; add unique_circuit_index instead of s_int
            global minimizer_loop_index, unique_id
            unique_id = instance_num * 1000 + minimizer_loop_index

            if verbose:
                print(f"--> begin batch: instance={instance_num}, index={minimizer_loop_index}")
                
            # variables used to aggregate metrics for all terms
            result_array = []

            #####################
            # loop over each of the circuits that are generated from a batch and execute
            # create the ImageRecognition ansatz to generate a parameterized hamiltonian
            
            if verbose and comfort: print("  ", end='')
                
            ts = time.time()
            for data_point in x_batch:
            
                if verbose and comfort: print('+', end='')
                    
                qc, frmt_obs, params = ImageRecognition(
                    num_qubits=num_qubits,
                    thetas_array=thetas_array,
                    ansatz_type=ansatz_type,
                    parameterized=parameterized,
                    reps=reps,
                    x_data=data_point
                    )

                # bind parameters to circuit before execution
                # if parameterized:
                #     qc.assign_parameters(params)
            
                # submit circuit for execution on target with the current parameters
                ex.submit_circuit(qc, num_qubits, unique_id, shots=num_shots, params=params)

                # wait for circuit to complete by calling finalize  ...
                # finalize execution of group (each circuit in loop accumulates metrics)
                ex.finalize_execution(None, report_end=False)

                # after first execution and thereafter, no need for transpilation if parameterized
                # DEVNOTE: this can be removed or commented, not used currently
                if parameterized:
                    # DEVNOTE: since we gen multipl circuits in this loop, and execute.py can only
                    # cache 1 at a time, we cannot yet implement caching.  Transpile every time.
                    cached_circuits = False
                    if cached_circuits:
                        ex.set_tranpilation_flags(do_transpile_metrics=False, do_transpile_for_execute=False)
                        logger.info("  **** First execution complete, disabling transpile")

                # result array stores the multiple results we measure along different Pauli basis.
                global saved_result
                result_array.append(saved_result)

                # Aggregate execution and elapsed time for all circuits, but only when the training
                # corresponding to different feature maps
                global quantum_execution_time, quantum_elapsed_time
                if not (train_pass):
                    quantum_execution_time = (
                        quantum_execution_time
                        + metrics.get_metric(num_qubits, unique_id, "exec_time")
                    )
                    quantum_elapsed_time = (
                        quantum_elapsed_time
                        + metrics.get_metric(num_qubits, unique_id, "elapsed_time")
                    )

                else:
                    quantum_execution_time = (
                        quantum_execution_time
                        + 0.0
                    )
                    quantum_elapsed_time = (
                        quantum_elapsed_time
                        + 0.0
                    )
                    
            # end of loop over data points
            
            # store the time it took to create the circuit
            # DEVNOTE: not correct; instead, accumulate time wrapped around ImageRecognition() 
            metrics.store_metric(num_qubits, unique_id, "create_time", time.time() - ts)

            global opt_ts

            # store the new exec time and elapsed time back to metrics
            metrics.store_metric(num_qubits, unique_id, "exec_time", quantum_execution_time)
            metrics.store_metric(num_qubits, unique_id, "elapsed_time", quantum_elapsed_time)
    
            #####################
            # classical processing of results
            
            if verbose:
                if comfort: print('')
                thetas_array_round = [round(th,3) for th in thetas_array]
                print(f"  ... compute loss and accuracy for num_qubits={num_qubits}, circuit={unique_id}, parameters={params},\n  thetas_array={thetas_array_round}")
                
            # increment the minimizer loop index, the index is increased by one
            # for the group of three circuits created ( three measurement basis circuits)

            # if not verbose, print "comfort dots" (newline before the first iteration)
            if comfort and not verbose:
                if minimizer_loop_index == 1:
                    print("")
                print(".", end="")
                if verbose:
                    print("")

            # Start counting classical optimizer time here again
            tc1 = time.time()

            # compute the loss for the image data batch
            loss, accuracy = loss_function(result=result_array, y_data=y_batch,
                    num_qubits= num_qubits, formatted_observables=frmt_obs)

            # calculate std error from the variance -- identically zero if using statevector simulator
            # if backend_id.lower() != "statevector_simulator":
            #     standard_error = np.sqrt(variance/num_shots)
            # else:
            #     standard_error = 0.0

            if verbose:
                print(f"  ... loss, accuracy = {loss}, {accuracy}")

            # append the most recent accuracy value to the list
            accuracy_values.append(accuracy)
            
            # store the metrics for the current iteration
            if test_pass:
                metrics.store_metric(num_qubits, unique_id, "test_accuracy", accuracy)

            if train_pass:
                metrics.store_metric(num_qubits, unique_id, "train_loss", loss)
                metrics.store_metric(num_qubits, unique_id, "train_accuracy", accuracy)
            
            loss_this_iter.append(loss)
            accuracy_this_iter.append(accuracy)
            
            # store metrics (not needed, done above, but may need more analysis
            # metrics.store_metric(num_qubits, unique_id, "loss", loss)
            # metrics.store_metric(num_qubits, unique_id, "accuracy", accuracy)
            # metrics.store_metric(num_qubits, unique_id, "standard_error", standard_error)
            metrics.store_metric(num_qubits, unique_id, "iteration_count", minimizer_loop_index + 1)

            # store most recent metrics for export          
            # key_metrics["energy"] = loss
            # key_metrics["accuracy"] = accuracy
            # key_metrics["variance"] = variance
            # key_metrics["standard_error"] = standard_error
            # key_metrics["iteration_count"] = minimizer_loop_index
            
            if not return_accuracy:
                return loss
            else:
                return loss, accuracy
       
        ##############
        def callback_thetas_array(thetas_array):
            '''
            This function called for every iteration of optimizer
            (in case of SPSA, this is called after two calls to objective_function)
            '''
            global quantum_execution_time, quantum_elapsed_time
            global x_batch, y_batch
            global minimizer_loop_index
            
            if verbose:
                print(f"==> in callback_thetas_array, minimizer loop index {minimizer_loop_index}")
                
            ######
            # print out execution time metrics for this iteration
            if verbose:
                print(f"  ... iteration exec, elapsed time = {quantum_execution_time}, {quantum_elapsed_time}")

            # reset the quantum_execution_time and quantum_elapsed_time
            # (done after all circuits of a batch and all calls to objective_function complete)
            quantum_execution_time = 0.0
            quantum_elapsed_time = 0.0
            
            ######
            # compute mean loss and accuracy on this iteration and store it in metrics table
            loss = np.mean(loss_this_iter)
            accuracy = np.mean(accuracy_this_iter)
            
            metrics.store_metric(num_qubits, unique_id, "train_loss", loss)
            metrics.store_metric(num_qubits, unique_id, "train_accuracy", accuracy) 

            # print out loss and accuracy metrics
            if verbose:
                print(f"  ... batch {minimizer_loop_index + 1} loss: {round(loss,4)} accuracy: {round(accuracy,4)}")
                
            loss_this_iter.clear()
            accuracy_this_iter.clear()

            ######
            # whenerver minimizer_loop_index is divisible by factor,
            # save the thetas_array at this minimizer_loop_index so we can save to file later
            factor = np.ceil(max_iter/test_pass_count)
            if minimizer_loop_index % factor == 0:
                thetas_array_batch[minimizer_loop_index + 1] = thetas_array.tolist() 
                if verbose:
                    print(f"  ... saved thetas_array at batch index {minimizer_loop_index + 1}")

            # reset the backend to the training backend
            ex.set_execution_target(
                backend_id_train, provider_backend=provider_backend, hub=hub, group=group, project=project, exec_options=exec_options
                )

            # generate a random set of indices that define the next batch
            indices = np.random.choice(len(x_train_scaled), size=batch_size, replace=False)
            x_batch = x_train_scaled[indices]
            y_batch = y_train[indices]
   
            # increment the loop index
            minimizer_loop_index += 1
            
   
        ###############
        if method == 1:
        
            # create the circuit(s) for given qubit size and secret string, store time metric
            ts = time.time()

            # set x_batch and y_batch to the test data
            x_batch = x_test_scaled
            y_batch = y_test
          
            # create an intial thetas_array, given the circuit width and user input
            thetas_array_0 = get_initial_parameters(num_qubits=num_qubits, thetas_array=thetas_array, ansatz_type=ansatz_type, reps=reps)

            # create one circuit with one data point
            data_point = x_batch[0]
            # create the circuits to be tested
            qc, frmt_obs, params = ImageRecognition(
                num_qubits=num_qubits,
                thetas_array=thetas_array_0,
                ansatz_type=ansatz_type,
                parameterized=parameterized,
                reps=reps,
                x_data=data_point
                )
            """ TMI ...
            # for testing and debugging ...
            #if using parameter objects, bind before printing
            if verbose:
                print(qc.assign_parameters(params) if parameterized else qc)
            """
            # store the creation time for these circuits
            metrics.store_metric(num_qubits, instance_num, "create_time", time.time() - ts)

            # classically pre-compute and cache an array of expected measurement counts
            # for comparison against actual measured counts for fidelity calc (in analysis)

            if do_compute_expectation:
                logger.info("Computing expectation")

                # pass parameters as they are used during execution
                compute_expectation(qc, num_qubits, instance_num, params=params)

            # submit circuit for execution on target, with parameters
            ex.submit_circuit(qc, num_qubits, instance_num, shots=num_shots, params=params)
            ex.finalize_execution(None, report_end=False)
        

        ###############
        elif method == 2:
            logger.info("===============  Begin method 2 loop, enabling transpile")

            # create batch from scaled data
            indices = np.random.choice(len(x_train_scaled), size=batch_size, replace=False)
            x_batch = x_train_scaled[indices]
            y_batch = y_train[indices]

            # Array of accuracy values collected during iterations of image_recognition
            accuracy_values = []

            # create an intial thetas_array, given the circuit width and user input
            thetas_array_0 = get_initial_parameters(num_qubits=num_qubits, thetas_array=thetas_array, ansatz_type=ansatz_type, reps=reps)

            # a unique circuit index used inside the inner minimizer loop as identifier
            # Value of 0 corresponds to the 0th iteration of the minimizer
            minimizer_loop_index = 0

            # Set iteration loss and accuracy to empty lists
            loss_this_iter = []
            accuracy_this_iter = []

            # Always start by enabling transpile ...
            ex.set_tranpilation_flags(do_transpile_metrics=True, do_transpile_for_execute=True)

            # dictionary to store the thetas_array for each batch
            thetas_array_batch = {} 
            
            if verbose:
                print(f"==> Launch optimizer, batch_size={batch_size}, max_iter={max_iter}")

            # minimize loss returned from objective function to find best theta values

            # execute SPSA optimizer to minimize the objective function
            # objective function is called repeatedly with varying parameters
            # until the max_iter are run
            if minimizer_function is None:
                ret = minimizeSPSA(objective_function, x0=thetas_array_0, a=0.3, c=0.3, niter=max_iter, callback=callback_thetas_array, paired=False)

            # or, execute a custom minimizer
            else:
                ret = minimizer_function(
                    objective_function=objective_function,
                    initial_parameters=thetas_array_0.ravel(),  # note: revel not really needed for this ansatz
                    callback=callback_thetas_array,
                )

            if comfort:
                print("!") 
                
            # remove the last element of metrics arrays, since it is always zero
            metrics.pop_metric(group=num_qubits, circuit = unique_id)

            # update the thetas_array_dict with the thetas_array_batch
            thetas_array_dict[num_qubits] = thetas_array_batch

            # write the thetas_array_dict to a json file
            # NOTE: the size of this file grows each time we increment num_qubits
            write_dict_to_json(thetas_array_dict, thetas_array_path + "precomputed_thetas.json")
          
            # save the data for this qubit width, and instance number
            store_final_iter_to_metrics_json(
                backend_id=backend_id_train,
                num_qubits=num_qubits,
                radius=radius,
                instance_num=instance_num,
                num_shots=num_shots,
                converged_thetas_list=ret.x.tolist(),
                accuracy=accuracy_values[-1],
                # iter_size_dist=iter_size_dist, iter_dist=iter_dist,
                detailed_save_names=detailed_save_names,
                dict_of_inputs=dict_of_inputs,
                save_final_counts=save_final_counts,
                save_res_to_file=save_res_to_file,
                _instances=_instances,
            )

            ###### End of instance processing


        ##############
        # for method 3, need to aggregate the detail metrics appropriately for each group
        # Note that this assumes that all iterations of the circuit have completed by this point

        elif method == 3:

            # set exectution target
            ex.set_execution_target(
                backend_id, provider_backend=provider_backend, hub=hub, group=group, project=project, exec_options=exec_options
                )

            # set x_batch and y_batch to the test data
            x_batch = x_test_scaled
            y_batch = y_test

            #  read the dictionary of thetas_array from the json file
            thetas_array_dict = read_dict_from_json(thetas_array_path + "precomputed_thetas.json")

            # get the thetas_array_batch for the current qubit size
            thetas_array_batch = thetas_array_dict[str(num_qubits)]

            # the iteration list is formed from the keys of the thetas_array_batch
            iteration_list = list(thetas_array_batch.keys())

            # Array of accuracy values collected during iterations of image_recognition
            accuracy_values = []

            # Set iteration loss and accuracy to empty lists
            loss_this_iter = []
            accuracy_this_iter = []

            # iterate over the iteration_list and calculate the loss and accuracy for test data
            # use the objective function to compute loss and accuracy
            for iteration_count in iteration_list:
                minimizer_loop_index = int(iteration_count) - 1
                thetas_array = np.array(thetas_array_batch[iteration_count])
                loss, accuracy = objective_function(thetas_array, return_accuracy=True, test_pass=True)

            if comfort and not verbose:
                print("!")
                
        if method == 2 or method == 3:
            metrics.process_circuit_metrics_2_level(num_qubits)
        
        metrics.finalize_group(num_qubits)
      
         
    # Wait for some active circuits to complete; report metrics when groups complete
    ex.throttle_execution(metrics.finalize_group)

    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    # print a sample circuit
    if print_sample_circuit:
        if method == 1:
            print("Sample Circuit:")
            print(QC_ if QC_ is not None else "  ... too large!")

    # Plot metrics for all circuit sizes
    if method == 1:
        metrics.plot_metrics(f"Benchmark Results - Image Recognition ({method}) - Qiskit",
                options=dict(shots=num_shots))
                
    elif method == 2:
        if plot_results:
            plot_results_from_data(**dict_of_inputs)
       
    elif method == 3:
        if plot_results:
            plot_results_from_data(**dict_of_inputs)

    
def get_final_results():
    """
    Return the energy and dict of key metrics of the last run().
    """
    
    # find the final energy value and return it
    energy=lowest_energy_values[-1] if len(lowest_energy_values) > 0 else None
    
    return energy, key_metrics
    
###################################

# DEVNOTE: This function should be re-implemented as just the objective function
# as it is defined in the run loop above with the necessary parameters

def run_objective_function(**kwargs):
    """
    Define a function to perform one iteration of the objective function.
    These argruments are preset to single execution: method=2, max_circuits=1, max+iter=1
    """
    
    # Fix arguments required to execute of single instance
    hl_single_args = dict(

        method=2,                   # method 2 defines the objective function
        max_circuits=1,             # only one repetition
        max_iter=1,                 # maximum minimizer iterations to perform, set to 1
        
        # disable display options for line plots
        line_y_metrics=None,
        line_x_metrics=None,
        
        # disable display options for bar plots
        bar_y_metrics=None,
        bar_x_metrics=None,

        # disable display options for area plots
        score_metric=None,
        x_metric=None,
    )
    # get the num_qubits are so we can force min and max to it.
    num_qubits = kwargs.pop("num_qubits")
      
    # Run the benchmark in method 2 at just one qubit size
    run(min_qubits=num_qubits, max_qubits=num_qubits,
            **kwargs, **hl_single_args)

    # find the final energy value and return it
    energy=lowest_energy_values[-1] if len(lowest_energy_values) > 0 else None
    
    return energy, key_metrics

#################################
# MAIN

# # if main, execute method
if __name__ == "__main__":
    run(min_qubits=6, max_qubits=8, num_shots=1000, max_iter=3, method=2, test_pass_count=30)

# # %%

# run()
