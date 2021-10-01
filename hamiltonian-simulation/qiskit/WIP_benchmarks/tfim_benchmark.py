"""
Hamiltonian-Simulation (Transverse Field Ising Model) Benchmark Program - Qiskit
"""

import sys

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit"]
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import time
import math
import numpy as np
np.random.seed(0)
import execute as ex
import metrics as metrics
from collections import defaultdict

verbose = False

# saved circuits and subcircuits for display
QC_ = None
ZZ_ = None

############### Circuit Definition

def HamiltonianSimulation(n_spins, K, t, method):
    '''
    Construct a Qiskit circuit for Hamiltonian Simulation
    :param n_spins:The number of spins to simulate
    :param K: The Trotterization order
    :param t: duration of simulation
    :param method: whether the circuit simulates the TFIM in paramagnetic or ferromagnetic phase
    :return: return a Qiskit circuit for this Hamiltonian
    '''
    
    # strength of transverse field
    if method == 1:
        g = 20.0 # g >> 1    -> paramagnetic phase
    else:
        g = 0.1 # g << 1    -> ferromagnetic phase
    
    # allocate qubits
    qr = QuantumRegister(n_spins); cr = ClassicalRegister(n_spins); qc = QuantumCircuit(qr, cr, name="main")
    
    # define timestep based on total runtime and number of Trotter steps
    tau = t / K
    
    # initialize state to approximate eigenstate when deep into phases of TFIM
    if abs(g) > 1: # paramagnetic phase
        # start with initial state of |++...> (eigenstate in x-basis)
        for k in range(n_spins):
            qc.h(qr[k])
    if abs(g) < 1: # ferromagnetic phase
        # state with initial state of GHZ state: 1/sqrt(2) ( |00...> + |11...> )
        qc.h(qr[0])
        for k in range(1, n_spins):
            qc.cnot(qr[k-1], qr[k])

    qc.barrier()

    # loop over each trotter step, adding gates to the circuit defining the Hamiltonian
    for k in range(K):
    
        # the Pauli spin vector product
        for i in range(n_spins):
            qc.rx(2 * tau * g, qr[i])
        qc.barrier()
        
        # ZZ operation on each pair of qubits in linear chain
        for j in range(2):
            for i in range(j%2, n_spins, 2):
                qc.append(zz_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
        qc.barrier()

    # transform state back to computational basis |00000>
    if abs(g) > 1: # paramagnetic phase
        # reverse transformation from |++...> (eigenstate in x-basis)
        for k in range(n_spins):
            qc.h(qr[k])
    if abs(g) < 1: # ferromagnetic phase
        # reversed tranformation from GHZ state
        for k in reversed(range(1, n_spins)):
            qc.cnot(qr[k-1], qr[k])
        qc.h(qr[0])
    
    qc.barrier()

    # measure all the qubits used in the circuit
    for i_qubit in range(n_spins):
        qc.measure(qr[i_qubit], cr[i_qubit])

    # save smaller circuit example for display
    global QC_    
    if QC_ == None or n_spins <= 4:
        if n_spins < 9: QC_ = qc

    return qc

############### exp(ZZ) Gate Implementations

# Simple exp(ZZ) gate on q0 and q1 with angle 'tau'
def zz_gate(tau):
    qr = QuantumRegister(2); qc = QuantumCircuit(qr, name="zz_gate")
    qc.cx(qr[0], qr[1])
    qc.rz(np.pi*tau, qr[1])
    qc.cx(qr[0], qr[1])

    # save circuit example for display
    global ZZ_    
    ZZ_ = qc

    return qc

############### Result Data Analysis

# Analyze and print measured results
# Compute the quality of the result based on operator expectation for each state
def analyze_and_print_result(qc, result, num_qubits, type, num_shots):

    counts = result.get_counts(qc)
    if verbose: print(f"For type {type} measured: {counts}")

    correct_state = '0'*num_qubits

    fidelity = 0
    if correct_state in counts.keys():
        fidelity = counts[correct_state] / num_shots

    return counts, fidelity


################ Benchmark Loop

# Execute program with default parameters
def run(min_qubits=2, max_qubits=8, max_circuits=3, num_shots=100, method=1,
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main"):
        
    print("Hamiltonian-Simulation (Transverse Field Ising Model) Benchmark Program - Qiskit")
    print(f"... using circuit method {method}")
    
    # validate parameters (smallest circuit is 2 qubits)
    max_qubits = max(2, max_qubits)
    min_qubits = min(max(2, min_qubits), max_qubits)
    if min_qubits % 2 == 1: min_qubits += 1   # min_qubits must be even
    #print(f"min, max qubits = {min_qubits} {max_qubits}")
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, type, num_shots):
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, expectation_a = analyze_and_print_result(qc, result, num_qubits, type, num_shots)
        metrics.store_metric(num_qubits, type, 'fidelity', expectation_a)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project)

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for input_size in range(min_qubits, max_qubits + 1, 2):

        # determine number of circuits to execute for this group
        num_circuits = max_circuits

        num_qubits = input_size
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")

        # parameters of simulation

        t = 1  # time of simulation, 1 is chosen so that the dynamics are not completely trivial

        k = int(5*num_qubits*t)  # Trotter error.
               # A large Trotter order approximates the Hamiltonian evolution better.
               # But a large Trotter order also means the circuit is deeper.
               # For ideal or noise-less quantum circuits, k >> 1 gives perfect hamiltonian simulation.

        for circuit_id in range(num_circuits):
        
            # create the circuit for given qubit size and simulation parameters, store time metric
            ts = time.time()
            h_x = 2 * np.random.random(num_qubits) - 1  # random numbers between [-1, 1]
            h_z = 2 * np.random.random(num_qubits) - 1
            qc = HamiltonianSimulation(num_qubits, K=k, t=t, method=method)
            metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time() - ts)
            
            # collapse the sub-circuits used in this benchmark (for qiskit)
            qc2 = qc.decompose()
            
            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc2, num_qubits, circuit_id, num_shots)
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
    
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    print("\n********\nZZ ="); print(ZZ_)
        
    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - Hamiltonian Simulation ({method}) - Qiskit")


# if main, execute method
if __name__ == '__main__': run()
