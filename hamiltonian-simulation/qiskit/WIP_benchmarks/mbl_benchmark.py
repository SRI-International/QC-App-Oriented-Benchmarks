"""
Many Body Localization Benchmark Program - Qiskit
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
XX_ = None
YY_ = None
ZZ_ = None
XXYYZZ_ = None

############### Circuit Definition

def HamiltonianSimulation(n_spins, K, t, w, h_x, h_z, method=2):
    '''
    Construct a Qiskit circuit for Hamiltonian Simulation
    :param n_spins:The number of spins to simulate
    :param K: The Trotterization order
    :param t: duration of simulation
    :param method: the method used to generate hamiltonian circuit
    :return: return a Qiskit circuit for this Hamiltonian
    '''
    
    # allocate qubits
    qr = QuantumRegister(n_spins); cr = ClassicalRegister(n_spins); qc = QuantumCircuit(qr, cr, name="main")
    tau = t / K

    # start with initial state of 1010101...
    for k in range(0, n_spins, 2):
        qc.x(qr[k])

    # loop over each trotter step, adding gates to the circuit defining the hamiltonian
    for k in range(K):
    
        # the Pauli spin vector product
        [qc.rx(2 * tau * w * h_x[i], qr[i]) for i in range(n_spins)]
        [qc.rz(2 * tau * w * h_z[i], qr[i]) for i in range(n_spins)]
        qc.barrier()
        
        '''
        Method 1:
        Basic implementation of exp(i * t * (XX + YY + ZZ))
        '''
        if method == 1:

            # XX operator on each pair of qubits in linear chain
            for j in range(2):
                for i in range(j%2, n_spins - 1, 2):
                    qc.append(xx_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

            # YY operator on each pair of qubits in linear chain
            for j in range(2):
                for i in range(j%2, n_spins - 1, 2):
                    qc.append(yy_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

            # ZZ operation on each pair of qubits in linear chain
            for j in range(2):
                for i in range(j%2, n_spins - 1, 2):
                    qc.append(zz_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
        '''
        Method 2: 
        Use an optimal XXYYZZ combined operator
        See equation 1 and Figure 6 in https://arxiv.org/pdf/quant-ph/0308006.pdf
        '''
        if method == 2:

            # optimized XX + YY + ZZ operator on each pair of qubits in linear chain
            for j in range(2):
                for i in range(j % 2, n_spins, 2):
                    qc.append(xxyyzz_opt_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

        qc.barrier()

    # measure all the qubits used in the circuit
    for i_qubit in range(n_spins):
        qc.measure(qr[i_qubit], cr[i_qubit])

    # save smaller circuit example for display
    global QC_    
    if QC_ == None or n_spins <= 6:
        if n_spins < 9: QC_ = qc

    return qc

############### XX, YY, ZZ Gate Implementations

# Simple XX gate on q0 and q1 with angle 'tau'
def xx_gate(tau):
    qr = QuantumRegister(2); qc = QuantumCircuit(qr, name="xx_gate")
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416*tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    
    # save circuit example for display
    global XX_    
    XX_ = qc
    
    return qc

# Simple YY gate on q0 and q1 with angle 'tau'    
def yy_gate(tau):
    qr = QuantumRegister(2); qc = QuantumCircuit(qr, name="yy_gate")
    qc.s(qr[0])
    qc.s(qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416*tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.sdg(qr[0])
    qc.sdg(qr[1])

    # save circuit example for display
    global YY_    
    YY_ = qc

    return qc

# Simple ZZ gate on q0 and q1 with angle 'tau'
def zz_gate(tau):
    qr = QuantumRegister(2); qc = QuantumCircuit(qr, name="zz_gate")
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416*tau, qr[1])
    qc.cx(qr[0], qr[1])

    # save circuit example for display
    global ZZ_    
    ZZ_ = qc

    return qc

# Optimal combined XXYYZZ gate (with double coupling) on q0 and q1 with angle 'tau'
def xxyyzz_opt_gate(tau):
    alpha = tau; beta = tau; gamma = tau
    qr = QuantumRegister(2); qc = QuantumCircuit(qr, name="xxyyzz_opt")
    qc.rz(3.1416/2, qr[1])
    qc.cx(qr[1], qr[0])
    qc.rz(3.1416*gamma - 3.1416/2, qr[0])
    qc.ry(3.1416/2 - 3.1416*alpha, qr[1])
    qc.cx(qr[0], qr[1])
    qc.ry(3.1416*beta - 3.1416/2, qr[1])
    qc.cx(qr[1], qr[0])
    qc.rz(-3.1416/2, qr[0])

    # save circuit example for display
    global XXYYZZ_    
    XXYYZZ_ = qc

    return qc


############### Result Data Analysis

# Analyze and print measured results
# Compute the quality of the result based on operator expectation for each state
def analyze_and_print_result(qc, result, num_qubits, type, num_shots):

    counts = result.get_counts(qc)
    if verbose: print(f"For type {type} measured: {counts}")

    ################### IMBALANCE CALCULATION ONE #######################
    expectation_a = 0
    for key in counts.keys():
        # compute the operator expectation for this state
        lambda_a = sum([((-1) ** i) * ((-1) ** int(bit)) for i, bit in enumerate(key)])/num_qubits
        prob = counts[key] / num_shots  # probability of this state
        expectation_a += lambda_a * prob
    #####################################################################

    ################### IMBALANCE CALCULATION TWO #######################
    # this is the imbalance calculation explicitely defined in Sonika's notes
    prob_one = {}
    for i in range(num_qubits):
        prob_one[i] = 0
    
    for key in counts.keys():
        for i in range(num_qubits):
            if key[::-1][i] == '1':
                prob_one[i] += counts[key] / num_shots
    
    I_numer = 0
    I_denom = 0
    for i in prob_one.keys():
        I_numer += (-1)**i * prob_one[i]
        I_denom += prob_one[i]
    I = I_numer/I_denom
    #####################################################################
    
    if verbose: print(f"\tMeasured Imbalance: {I}, measured expectation_a: {expectation_a}")

    # rescaled fideltiy
    fidelity = I/0.4

    # rescaled expectation_a
    expectation_a = expectation_a/0.4

    # We expect expecation_a to give 1. We would like to plot the deviation from the true expectation.
    return counts, fidelity


################ Benchmark Loop

# Execute program with default parameters
def run(min_qubits=2, max_qubits=8, max_circuits=300, num_shots=100, method=2,
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main"):
        
    print("Many Body Localization Benchmark Program - Qiskit")
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

    ex.set_noise_model()

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for input_size in range(min_qubits, max_qubits + 1, 2):

        # determine number of circuits to execute for this group
        num_circuits = max_circuits

        num_qubits = input_size
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")

        # parameters of simulation

        w = 20  # strength of disorder

        k = 100  # Trotter error.
               # A large Trotter order approximates the Hamiltonian evolution better.
               # But a large Trotter order also means the circuit is deeper.
               # For ideal or noise-less quantum circuits, k >> 1 gives perfect hamiltonian simulation.

        t = 1.2  # time of simulation

        for circuit_id in range(num_circuits):
        
            # create the circuit for given qubit size and simulation parameters, store time metric
            ts = time.time()
            h_x = 2 * np.random.random(num_qubits) - 1  # random numbers between [-1, 1]
            h_z = 2 * np.random.random(num_qubits) - 1
            qc = HamiltonianSimulation(num_qubits, K=k, t=t, w=w, h_x= h_x, h_z=h_z, method=method)
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
   
    if method == 1:
        print("\n********\nXX, YY, ZZ =")
        print(XX_); print(YY_); print(ZZ_)
    else:
        print("\n********\nXXYYZZ =")
        print(XXYYZZ_)
        
    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - Hamiltonian Simulation ({method}) - Qiskit")


# if main, execute method
if __name__ == '__main__': run()
