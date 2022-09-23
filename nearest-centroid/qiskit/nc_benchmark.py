"""
Nearest Centroid Inner Product Estimation Benchmark Program - Qiskit
"""

import time
import sys

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import UnitaryGate

sys.path[1:1] = [ "_common", "_common/qiskit" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit" ]
import execute as ex
import metrics as metrics

np.random.seed(0)

verbose = False

# Saved circuits for display
QC_ = None
DL_ = None
iDL_ = None

############### Circuit Definition

def RBS_unitary(theta):
    return np.matrix([
        [1, 0, 0, 0],
        [0, np.cos(theta), np.sin(theta), 0],
        [0, -np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def RBS(theta):
    return UnitaryGate(RBS_unitary(theta), f"RBS({theta})")

def RBS_dagger(theta):
    return UnitaryGate(RBS_unitary(theta).H, f"RBS_dagger({theta})")

def compute_angles(num_qubits, data):
    """
    Computes angle parameters for a given input vector
    """
    # set dimension
    d = num_qubits

    # Array to hold r values
    r_values: np.ndarray = np.zeros(shape=(d-1,))
    
    # Array to hold angles
    angles: np.ndarray = np.zeros(shape=(d-1,))
        
    # Back index
    back_index_range = range(1, int(d/2) + 1)
    front_index_range = range(1, int(d/2))
    
    # Compute the last d/2 r values
    for j in back_index_range:
        two_j = 2*j
        r_index = int(d/2) + j - 1
        first_x = two_j
        second_x = two_j - 1
        
        # Pull indexes, accounting for zero vs. one-indexed array
        r_values[r_index - 1] = np.sqrt(data[first_x - 1]**2 + data[second_x - 1]**2)

    # Compute the first d/2 - 1 r values
    for j in reversed(front_index_range):
        two_j = 2*j
        r_index = j
        first_r = two_j + 1
        second_r = two_j
        
        # Pull indexes, accounting for zero vs. one-indexed array
        r_values[r_index - 1] = np.sqrt(r_values[first_r - 1]**2 + r_values[second_r - 1]**2)

    # Compute the last d/2 theta values
    for j in back_index_range:
        two_j = 2*j
        theta_index = int(d/2) + j - 1
        x_check_index = two_j
        x_index = two_j - 1
        r_index = theta_index
        
        # Run check
        if data[x_check_index - 1] >= 0:
            angles[theta_index - 1] = np.arccos(data[x_index - 1] / r_values[r_index - 1])
        else:
            angles[theta_index - 1] = (2 * np.pi) - np.arccos(data[x_index - 1] / r_values[r_index - 1])
    
    # Compute the first d/2 - 1 theta values
    for j in reversed(front_index_range):
        two_j = 2*j
        theta_index = j
        first_r = two_j
        second_r = j
        
        angles[theta_index - 1] = np.arccos(r_values[first_r - 1] / r_values[second_r - 1])
    
    return angles

def parallel_loader_recursive(offset, angle_index, level, d):
    """
    Recursively computes the necessary data to create loader circuit
    and associate the appropriate angle parameters
    """
    pairs = []
    
    # If dimension is zero, return empty qc
    if d == 1: return pairs

    # Append RBS gate
    pairs.append((angle_index, offset, offset + int(d/2)))
    
    # Compute the angle index for the next levels
    left_angle_index = angle_index + (2**level)
    right_angle_index = left_angle_index + 1
    
    # Append two more with d = d/2
    left_pairs = parallel_loader_recursive(offset, left_angle_index, level+1, int(d/2))
    right_pairs = parallel_loader_recursive(offset + int(d/2), right_angle_index, level+1, int(d/2))
    
    if len(left_pairs) > 0: pairs.extend(left_pairs)
    if len(right_pairs) > 0: pairs.extend(right_pairs)
    
    return pairs

def create_data_loader(num_qubits, data):
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr, name="DL")

    # Compute angles
    angles = compute_angles(num_qubits, data)

    # Returns gate data, list of pairs (angle_index, qubit_0, qubit_1)
    gate_data = parallel_loader_recursive(0, 0, 0, num_qubits)
    
    for gate in gate_data:
        qc.append(RBS(angles[gate[0]]), [gate[1], gate[2]])
            
    return qc

def create_inverse_data_loader(num_qubits, data):
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr, name="iDL")

    # Compute angles
    angles = compute_angles(num_qubits, data)

    # Returns gate data, list of pairs (angle_index, qubit_0, qubit_1)
    gate_data = parallel_loader_recursive(0, 0, 0, num_qubits)
    
    # Create the same circuit, except using RBS_dagger, then reverse the 
    # order of operations
    for gate in gate_data:
        qc.append(RBS_dagger(angles[gate[0]]), [gate[1], gate[2]])
            
    return qc.reverse_ops()

def NearestCentroid(num_qubits, v1, v2):
    # Define registers
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(1)

    # Define base quantum circuit
    qc = QuantumCircuit(qr, cr, name="main")

    # Initialize first qubit to |1>
    qc.x(0)

    # Construct data loader
    DL = create_data_loader(num_qubits, v1)
    qc.append(DL, qr)

    qc.barrier()

    # Construct inverse data loader
    iDL = create_inverse_data_loader(num_qubits, v2)
    qc.append(iDL, qr)

    qc.barrier()

    # Measure first qubit
    qc.measure(qr[0], 0)

    # Save circuits for display
    global QC_
    global DL_
    global iDL_

    if QC_ == None or num_qubits <= 6:
        if num_qubits < 9: QC_ = qc
    if DL_ == None or num_qubits <= 6:
        if num_qubits < 9: DL_ = DL
    if iDL_ == None or num_qubits <= 6:
        if num_qubits < 9: iDL_ = iDL

    return qc

############### Result Data Analysis

# Analyze and print measured results
# Expected result is always the secret_int, so fidelity calc is simple
def analyze_and_print_result (qc, result, num_qubits, ip, num_shots):
    # obtain counts from the result object
    counts = result.get_counts(qc)
    if verbose: print(f"For true inner product {ip} measured: {counts}")

    
    # correct distribution is measuring the probability of |1> as the 
    # square of the inner product... 
    # TODO: Figure out the exact correct distribution...
    correct_dist = {'1': int(num_shots * float(ip)**2)}

    print(counts)
    print(correct_dist)

    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)
        
    return counts, fidelity

################ Benchmark Loop

# Execute program with default parameters
def run (min_qubits=3, max_qubits=8, max_circuits=3, num_shots=100,
        method = 1,
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None):

    print("Nearest Centroid Inner Product Estimation Benchmark Program - Qiskit")

    # validate parameters (smallest circuit is 3 qubits)
    max_qubits = max(3, max_qubits)
    min_qubits = min(max(3, min_qubits), max_qubits)
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    # TODO: Update execution handler to match analyze function above and finish benchmark
    # code...
    def execution_handler(qc, result, num_qubits, ip, num_shots):  
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, ip, num_shots)
        metrics.store_metric(num_qubits, ip, 'fidelity', fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options)

    # for noiseless simulation, set noise model to be None
    # ex.set_noise_model(None)

    # TODO: Update below to generate random vectors for testing

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1):
        
        # determine number of circuits to execute for this group
        num_circuits = min(2**(num_qubits), max_circuits)
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        
        # generate max_circuits pairs of random vectors to estimate inner product
        def random_vector(): 
            v = np.random.uniform(size=(num_qubits,))
            return v / np.linalg.norm(v)

        vector_pairs = [(random_vector(), random_vector()) for _ in range(num_circuits)]

        # loop over limited # of secret strings for this
        for v_pair in vector_pairs:
            # create the circuit for given qubit size and inner product, store time metric
            ts = time.time()
            qc = NearestCentroid(num_qubits, v_pair[0], v_pair[1])
            
            # NOTE: We might need a better candidate for the circuit/group ID than the inner product itself...
            ip = np.inner(v_pair[0], v_pair[1])

            metrics.store_metric(num_qubits, ip, 'create_time', time.time()-ts)

            # collapse the sub-circuit levels used in this benchmark (for qiskit)
            qc2 = qc.decompose()

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc2, num_qubits, ip, shots=num_shots)
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
        
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    print("Sample Parallel Loader:"); print(DL_ if DL_ != None else "  ... too large!")
    print("Sample Inverse Parallel Loader:"); print(iDL_ if iDL_ != None else "  ... too large!")

    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - Nearest Centroid Inner Product Estimation - Qiskit")

# if main, execute method
if __name__ == '__main__': run()