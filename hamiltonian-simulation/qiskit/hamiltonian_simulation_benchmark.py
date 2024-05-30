"""
Hamiltonian-Simulation Benchmark Program - Qiskit
"""

import json
import os
import sys
import time

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import TimeEvolutionProblem, SciPyRealEvolver

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit"]
import execute as ex
import metrics as metrics

# Benchmark Name
benchmark_name = "Hamiltonian Simulation"

np.random.seed(0)

verbose = False

# saved circuits and subcircuits for display
QC_ = None
XX_ = None
YY_ = None
ZZ_ = None
XXYYZZ_ = None

# for validating the implementation of XXYYZZ operation
_use_XX_YY_ZZ_gates = False

# import precalculated data to compare against
filename = os.path.join(os.path.dirname(__file__), os.path.pardir, "_common", "precalculated_data.json")
with open(filename, 'r') as file:
    data = file.read()
precalculated_data = json.loads(data)

############### Circuit Definition

def initial_state(n_spins, method): 

    qc = QuantumCircuit(n_spins)

    if method == 1: 

        # checkerboard state, or "Neele" state 

        for k in range(0, n_spins, 2):
            qc.x([k])

    else:

        # state with initial state of GHZ state: 1/sqrt(2) ( |00...> + |11...> )
        qc.h(0)
        for k in range(1, n_spins):
            qc.cx(k-1, k)

    return qc 

def construct_TFIM_hamiltonian(n_spins):

    pauli_strings = []
    coefficients = []

    # g is hardcoded for now 
    g= .2  

    # the Pauli spin vector product
    for i in range(n_spins):
        x_term = 'I' * i + 'X' + 'I' * (n_spins - i - 1)
        pauli_strings.append(x_term)
        coefficients.append(g)

    identity_string = ['I'] * n_spins 

    # ZZ operation on each pair of qubits in linear chain
    for j in range(2):
        for i in range(j%2, n_spins-1, 2):

            zz_term = identity_string.copy()

            zz_term[i] = 'Z'
            zz_term[(i+1) % n_spins] = 'Z'

            zz_term = ''.join(zz_term)

            pauli_strings.append(zz_term)
            coefficients.append(1.0)

    # Construct SparsePauliOp
    sparse_pauli_op = SparsePauliOp.from_list(zip(pauli_strings, coefficients))
    return sparse_pauli_op


def construct_heisenberg_hamiltonian(n_spins):

    w = precalculated_data['w']  # strength of disorder
    h_x = precalculated_data['h_x'][:n_spins] # precalculated random numbers between [-1, 1]
    h_z = precalculated_data['h_z'][:n_spins]
    # Initialize lists to hold Pauli strings and coefficients
    pauli_strings = []
    coefficients = []

    # Disorder terms
    for i in range(n_spins):
        x_term = 'I' * i + 'X' + 'I' * (n_spins - i - 1)
        z_term = 'I' * i + 'Z' + 'I' * (n_spins - i - 1)
        pauli_strings.append(x_term)
        coefficients.append(w * h_x[i])
        pauli_strings.append(z_term)
        coefficients.append(w * h_z[i])

    identity_string = ['I'] * n_spins 

    # Interaction terms

    for j in range(2):
        for i in range(j % 2, n_spins - 1, 2):


            xx_term = identity_string.copy()
            yy_term = identity_string.copy() 
            zz_term = identity_string.copy()

            xx_term[i] = 'X'
            xx_term[(i+1) % n_spins] = 'X'

            yy_term[i] = 'Y'
            yy_term[(i+1) % n_spins] = 'Y'

            zz_term[i] = 'Z'
            zz_term[(i+1) % n_spins] = 'Z'

            xx_term = ''.join(xx_term)
            yy_term = ''.join(yy_term)
            zz_term = ''.join(zz_term)

            pauli_strings.append(xx_term)
            coefficients.append(1.0)
            pauli_strings.append(yy_term)
            coefficients.append(1.0)
            pauli_strings.append(zz_term)
            coefficients.append(1.0)

    # Construct SparsePauliOp
    sparse_pauli_op = SparsePauliOp.from_list(zip(pauli_strings, coefficients))

    return sparse_pauli_op

def construct_hamiltonian(n_spins, method):

    if method == 1:
        return construct_heisenberg_hamiltonian(n_spins)
    elif method == 2: 
        return construct_TFIM_hamiltonian(n_spins)
    else:
        raise ValueError("Method is not equal to 1 or 2.")

def HamiltonianSimulationExact(n_spins, t, method=1):
    """
    Returns the distribution after simulating our hamiltonian simulation circuits using a classically calculated (using SciPy) matrix evolution
    """

    hamiltonian = construct_hamiltonian(n_spins, method) 

    time_problem = TimeEvolutionProblem(hamiltonian, t, initial_state = initial_state(n_spins, method))

    result = SciPyRealEvolver(num_timesteps=1).evolve(time_problem)

    return result.evolved_state.probabilities_dict()
    

def HamiltonianSimulation(n_spins, K, t, method = 1):
    '''
    Construct a Qiskit circuit for Hamiltonian Simulation
    :param n_spins:The number of spins to simulate
    :param K: The Trotterization order
    :param t: duration of simulation
    :return: return a Qiskit circuit for this Hamiltonian
    '''
    
    num_qubits = n_spins
    secret_int = f"{K}-{t}"
    
    # allocate qubits
    qr = QuantumRegister(n_spins); cr = ClassicalRegister(n_spins)
    qc = QuantumCircuit(qr, cr, name=f"hamsim-{num_qubits}-{secret_int}")
    tau = t / K

    w = precalculated_data['w']  # strength of disorder
    h_x = precalculated_data['h_x'][:n_spins] # precalculated random numbers between [-1, 1]
    h_z = precalculated_data['h_z'][:n_spins]


    if method==1:


    # checkerboard state, or "Neele" state 
        for k in range(0, n_spins, 2):
            qc.x([k])

    # loop over each trotterustep, adding gates to the circuit defining the hamiltonian
        for k in range(K):
            # the Pauli spin vector product
            [qc.rx(2 * tau * w * h_x[i], qr[i]) for i in range(n_spins)]
            [qc.rz(2 * tau * w * h_z[i], qr[i]) for i in range(n_spins)]
            qc.barrier()
            
            # Basic implementation of exp(i * t * (XX + YY + ZZ))
            if _use_XX_YY_ZZ_gates:

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

            # Use an optimal XXYYZZ combined operator
            # See equation 1 and Figure 6 in https://arxiv.org/pdf/quant-ph/0308006.pdf
            else:
                # optimized XX + YY + ZZ operator on each pair of qubits in linear chain
                for j in range(2):
                    for i in range(j % 2, n_spins - 1, 2):
                        qc.append(xxyyzz_opt_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

            qc.barrier()

    elif method==2: 

        g=0.2 # strength of tranverse field

        # state with initial state of GHZ state: 1/sqrt(2) ( |00...> + |11...> )
        qc.h(qr[0])
        for k in range(1, n_spins):
            qc.cx(qr[k-1], qr[k])

        qc.barrier()

        ##calculate TFIM

        for k in range(K):
            # the Pauli spin vector product
            for i in range(n_spins):
                qc.rx(2 * tau * g, qr[i])
            qc.barrier()


            # ZZ operation on each pair of qubits in linear chain
            for j in range(2):
                for i in range(j%2, n_spins-1, 2):
                    qc.append(zz_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
            qc.barrier()


        # reversed tranformation from GHZ state
        for k in reversed(range(1, n_spins)):
            qc.cx(qr[k-1], qr[k])
        qc.h(qr[0])
        qc.barrier()


    else: 

        raise ValueError("Invalid method specification.")

    # measure all qubits 
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
def analyze_and_print_result(qc, result, num_qubits, type, num_shots, method, compare_to_exact_results):

    counts = result.get_counts(qc)
    if verbose: print(f"For type {type} measured: {counts}")



    # we have precalculated the correct distribution that a perfect quantum computer will return
    # it is stored in the json file we import at the top of the code

    if method == 1 and not compare_to_exact_results:
        # ideal Heisenburg Ham Sim. Circuit results
        correct_dist = precalculated_data[f"Heisenburg - Qubits{num_qubits}"]
    elif method == 1 and compare_to_exact_results:
        # Classically calculated Heisenburg Ham Sim. Results 
        correct_dist = precalculated_data[f"Exact Heisenburg - Qubits{num_qubits}"]
    elif method == 2 and not compare_to_exact_results:   
        # ideal Heisenburg Ham Sim. Circuit results
        correct_dist = precalculated_data[f"TFIM - Qubits{num_qubits}"]
    elif method == 2 and compare_to_exact_results:
        # Classically calculated TFIM Ham Sim. Results 
        correct_dist = precalculated_data[f"Exact TFIM - Qubits{num_qubits}"]
    else: 
        raise ValueError("Method 1 is not 1 or 2, or compare_to_exact_results was unexpected type.")
    

    if verbose: print(f"Correct dist: {correct_dist}")

    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)

    return counts, fidelity


################ Benchmark Loop

# Execute program with default parameters
def run(min_qubits=2, max_qubits=8, max_circuits=3, skip_qubits=1, num_shots=100,
        use_XX_YY_ZZ_gates = False,
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None,
        compare_to_exact_results = False,
        context=None, method=1):

    print(f"{benchmark_name} Benchmark Program - Qiskit")
    
    # validate parameters (smallest circuit is 2 qubits)
    max_qubits = max(2, max_qubits)
    min_qubits = min(max(2, min_qubits), max_qubits)
    if min_qubits % 2 == 1: min_qubits += 1   # min_qubits must be even
    skip_qubits = max(1, skip_qubits)
    #print(f"min, max qubits = {min_qubits} {max_qubits}")

    # create context identifier
    #if context is None: context = f"{benchmark_name} ({method}) Benchmark"
    if context is None: context = f"{benchmark_name} Benchmark"
    
    # set the flag to use an XX YY ZZ shim if given
    global _use_XX_YY_ZZ_gates
    _use_XX_YY_ZZ_gates = use_XX_YY_ZZ_gates
    if _use_XX_YY_ZZ_gates:
        print("... using unoptimized XX YY ZZ gates")
    
    ##########
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, type, num_shots):
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, expectation_a = analyze_and_print_result(qc, result, num_qubits, type, num_shots, method, compare_to_exact_results)
        metrics.store_metric(num_qubits, type, 'fidelity', expectation_a)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)

    ##########
    
    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):

        # reset random seed
        np.random.seed(0)

        # determine number of circuits to execute for this group
        num_circuits = max(1, max_circuits)
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")

        # parameters of simulation
        #### CANNOT BE MODIFIED W/O ALSO MODIFYING PRECALCULATED DATA #########
        w = precalculated_data['w']  # strength of disorder
        k = precalculated_data['k']   # Trotter error.
               # A large Trotter order approximates the Hamiltonian evolution better.
               # But a large Trotter order also means the circuit is deeper.
               # For ideal or noise-less quantum circuits, k >> 1 gives perfect hamiltonian simulation.
        t = precalculated_data['t']  # time of simulation
        #######################################################################


        # loop over only 1 circuit
        for circuit_id in range(num_circuits):
        
            #print(circuit_id)
        
            # create the circuit for given qubit size and simulation parameters, store time metric
            ts = time.time()
            h_x = precalculated_data['h_x'][:num_qubits] # precalculated random numbers between [-1, 1]
            h_z = precalculated_data['h_z'][:num_qubits]
            qc = HamiltonianSimulation(num_qubits, K=k, t=t, method=method) 
            metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time() - ts)
            qc.draw()
            
            # collapse the sub-circuits used in this benchmark (for qiskit)
            qc2 = qc.decompose()

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc2, num_qubits, circuit_id, num_shots)
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
    
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    ##########
    
    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
   
    if _use_XX_YY_ZZ_gates:
        print("\nXX, YY, ZZ =")
        print(XX_); print(YY_); print(ZZ_)
    else:
        print("\nXXYYZZ_opt =")
        print(XXYYZZ_)
       
    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - {benchmark_name} - Qiskit")


# if main, execute method
if __name__ == '__main__':

    run()
