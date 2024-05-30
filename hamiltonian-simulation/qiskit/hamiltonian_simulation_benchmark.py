"""
Hamiltonian-Simulation Benchmark Program - Qiskit

This program benchmarks Hamiltonian simulation using Qiskit. 
The central function is the `run()` method, which orchestrates the entire benchmarking process.

HamiltonianSimulation forms the trotterized circuit used in the benchmark.

HamiltonianSimulationExact runs a classical calculation that perfectly simulates hamiltonian evolution, although it does not scale well. 
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

# Saved circuits and subcircuits for display
QC_ = None
XX_ = None
YY_ = None
ZZ_ = None
XXYYZZ_ = None

# For validating the implementation of XXYYZZ operation
_use_XX_YY_ZZ_gates = False

# Import precalculated data to compare against
filename = os.path.join(os.path.dirname(__file__), os.path.pardir, "_common", "precalculated_data.json")
with open(filename, 'r') as file:
    data = file.read()
precalculated_data = json.loads(data)

############### Circuit Definition

def initial_state(n_spins: int, method: int) -> QuantumCircuit:
    """
    Initialize the quantum state.
    
    Args:
        n_spins (int): Number of spins (qubits).
        method (int): Method of initialization (1 for checkerboard state, otherwise GHZ state).

    Returns:
        QuantumCircuit: The initialized quantum circuit.
    """
    qc = QuantumCircuit(n_spins)

    if method == 1:
        # Checkerboard state, or "Neele" state
        for k in range(0, n_spins, 2):
            qc.x([k])
    else:
        # GHZ state: 1/sqrt(2) (|00...> + |11...>)
        qc.h(0)
        for k in range(1, n_spins):
            qc.cx(k-1, k)

    return qc

def construct_TFIM_hamiltonian(n_spins: int) -> SparsePauliOp:
    """
    Construct the Transverse Field Ising Model (TFIM) Hamiltonian.

    Args:
        n_spins (int): Number of spins (qubits).

    Returns:
        SparsePauliOp: The Hamiltonian represented as a sparse Pauli operator.
    """
    pauli_strings = []
    coefficients = []
    g = 0.2  # Strength of the transverse field

    # Pauli spin vector product terms
    for i in range(n_spins):
        x_term = 'I' * i + 'X' + 'I' * (n_spins - i - 1)
        pauli_strings.append(x_term)
        coefficients.append(g)

    identity_string = ['I'] * n_spins

    # ZZ operation on each pair of qubits in a linear chain
    for j in range(2):
        for i in range(j % 2, n_spins - 1, 2):
            zz_term = identity_string.copy()
            zz_term[i] = 'Z'
            zz_term[(i + 1) % n_spins] = 'Z'
            zz_term = ''.join(zz_term)
            pauli_strings.append(zz_term)
            coefficients.append(1.0)

    return SparsePauliOp.from_list(zip(pauli_strings, coefficients))

def construct_heisenberg_hamiltonian(n_spins: int) -> SparsePauliOp:
    """
    Construct the Heisenberg Hamiltonian with disorder.

    Args:
        n_spins (int): Number of spins (qubits).

    Returns:
        SparsePauliOp: The Hamiltonian represented as a sparse Pauli operator.
    """
    w = precalculated_data['w']  # Strength of disorder
    h_x = precalculated_data['h_x'][:n_spins]  # Precalculated random numbers between [-1, 1]
    h_z = precalculated_data['h_z'][:n_spins]

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
            xx_term[(i + 1) % n_spins] = 'X'

            yy_term[i] = 'Y'
            yy_term[(i + 1) % n_spins] = 'Y'

            zz_term[i] = 'Z'
            zz_term[(i + 1) % n_spins] = 'Z'

            pauli_strings.append(''.join(xx_term))
            coefficients.append(1.0)
            pauli_strings.append(''.join(yy_term))
            coefficients.append(1.0)
            pauli_strings.append(''.join(zz_term))
            coefficients.append(1.0)

    return SparsePauliOp.from_list(zip(pauli_strings, coefficients))

def construct_hamiltonian(n_spins: int, method: int) -> SparsePauliOp:
    """
    Construct the Hamiltonian based on the specified method.

    Args:
        n_spins (int): Number of spins (qubits).
        method (int): Method of Hamiltonian construction (1 for Heisenberg, 2 for TFIM).

    Returns:
        SparsePauliOp: The constructed Hamiltonian.
    """
    if method == 1:
        return construct_heisenberg_hamiltonian(n_spins)
    elif method == 2:
        return construct_TFIM_hamiltonian(n_spins)
    else:
        raise ValueError("Method is not equal to 1 or 2.")

def HamiltonianSimulationExact(n_spins: int, t: float, method: int = 1) -> dict:
    """
    Perform exact Hamiltonian simulation using classical matrix evolution.

    Args:
        n_spins (int): Number of spins (qubits).
        t (float): Duration of simulation.
        method (int): Method of Hamiltonian construction (1 for Heisenberg, 2 for TFIM).

    Returns:
        dict: The distribution of the evolved state.
    """
    hamiltonian = construct_hamiltonian(n_spins, method)
    time_problem = TimeEvolutionProblem(hamiltonian, t, initial_state=initial_state(n_spins, method))
    result = SciPyRealEvolver(num_timesteps=1).evolve(time_problem)
    return result.evolved_state.probabilities_dict()

def HamiltonianSimulation(n_spins: int, K: int, t: float, method: int = 1) -> QuantumCircuit:
    """
    Construct a Qiskit circuit for Hamiltonian simulation.

    Args:
        n_spins (int): Number of spins (qubits).
        K (int): The Trotterization order.
        t (float): Duration of simulation.
        method (int): Method of Hamiltonian construction (1 for Heisenberg, 2 for TFIM).

    Returns:
        QuantumCircuit: The constructed Qiskit circuit.
    """
    num_qubits = n_spins
    secret_int = f"{K}-{t}"
    
    # Allocate qubits
    qr = QuantumRegister(n_spins)
    cr = ClassicalRegister(n_spins)
    qc = QuantumCircuit(qr, cr, name=f"hamsim-{num_qubits}-{secret_int}")
    tau = t / K

    w = precalculated_data['w']  # Strength of disorder
    h_x = precalculated_data['h_x'][:n_spins]  # Precalculated random numbers between [-1, 1]
    h_z = precalculated_data['h_z'][:n_spins]

    if method == 1:
        # Checkerboard state, or "Neele" state
        for k in range(0, n_spins, 2):
            qc.x([k])

        # Loop over each Trotter step, adding gates to the circuit defining the Hamiltonian
        for k in range(K):
            # Pauli spin vector product
            [qc.rx(2 * tau * w * h_x[i], qr[i]) for i in range(n_spins)]
            [qc.rz(2 * tau * w * h_z[i], qr[i]) for i in range(n_spins)]
            qc.barrier()
            
            # Basic implementation of exp(i * t * (XX + YY + ZZ))
            if _use_XX_YY_ZZ_gates:
                for j in range(2):
                    for i in range(j % 2, n_spins - 1, 2):
                        qc.append(xx_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
                        qc.append(yy_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
                        qc.append(zz_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
            else:
                # Optimized XX + YY + ZZ operator on each pair of qubits in linear chain
                for j in range(2):
                    for i in range(j % 2, n_spins - 1, 2):
                        qc.append(xxyyzz_opt_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
            qc.barrier()
    elif method == 2:
        g = 0.2  # Strength of transverse field

        # GHZ state: 1/sqrt(2) (|00...> + |11...>)
        qc.h(qr[0])
        for k in range(1, n_spins):
            qc.cx(qr[k-1], qr[k])
        qc.barrier()

        # Calculate TFIM
        for k in range(K):
            for i in range(n_spins):
                qc.rx(2 * tau * g, qr[i])
            qc.barrier()

            for j in range(2):
                for i in range(j % 2, n_spins - 1, 2):
                    qc.append(zz_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
            qc.barrier()

        # Reverse transformation from GHZ state
        for k in reversed(range(1, n_spins)):
            qc.cx(qr[k-1], qr[k])
        qc.h(qr[0])
        qc.barrier()
    else:
        raise ValueError("Invalid method specification.")

    # Measure all qubits
    for i_qubit in range(n_spins):
        qc.measure(qr[i_qubit], cr[i_qubit])

    # Save smaller circuit example for display
    global QC_
    if QC_ is None or n_spins <= 6:
        if n_spins < 9:
            QC_ = qc

    return qc

############### XX, YY, ZZ Gate Implementations

def xx_gate(tau: float) -> QuantumCircuit:
    """
    Simple XX gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The XX gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="xx_gate")
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    
    global XX_
    XX_ = qc
    
    return qc

def yy_gate(tau: float) -> QuantumCircuit:
    """
    Simple YY gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The YY gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="yy_gate")
    qc.s(qr[0])
    qc.s(qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.sdg(qr[0])
    qc.sdg(qr[1])

    global YY_
    YY_ = qc

    return qc

def zz_gate(tau: float) -> QuantumCircuit:
    """
    Simple ZZ gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The ZZ gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="zz_gate")
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])

    global ZZ_
    ZZ_ = qc

    return qc

def xxyyzz_opt_gate(tau: float) -> QuantumCircuit:
    """
    Optimal combined XXYYZZ gate (with double coupling) on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The optimal combined XXYYZZ gate circuit.
    """
    alpha = tau
    beta = tau
    gamma = tau
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="xxyyzz_opt")
    qc.rz(3.1416 / 2, qr[1])
    qc.cx(qr[1], qr[0])
    qc.rz(3.1416 * gamma - 3.1416 / 2, qr[0])
    qc.ry(3.1416 / 2 - 3.1416 * alpha, qr[1])
    qc.cx(qr[0], qr[1])
    qc.ry(3.1416 * beta - 3.1416 / 2, qr[1])
    qc.cx(qr[1], qr[0])
    qc.rz(-3.1416 / 2, qr[0])

    global XXYYZZ_
    XXYYZZ_ = qc

    return qc

############### Result Data Analysis

def analyze_and_print_result(qc: QuantumCircuit, result, num_qubits: int, type: str, num_shots: int, method: int, compare_to_exact_results: bool) -> tuple:
    """
    Analyze and print the measured results. Compute the quality of the result based on operator expectation for each state.

    Args:
        qc (QuantumCircuit): The quantum circuit.
        result: The result from the execution.
        num_qubits (int): Number of qubits.
        type (str): Type of the simulation.
        num_shots (int): Number of shots.
        method (int): Method of Hamiltonian construction.
        compare_to_exact_results (bool): Whether to compare to exact results.

    Returns:
        tuple: Counts and fidelity.
    """
    counts = result.get_counts(qc)
    if verbose:
        print(f"For type {type} measured: {counts}")

    # Precalculated correct distribution
    if method == 1 and not compare_to_exact_results:
        correct_dist = precalculated_data[f"Heisenburg - Qubits{num_qubits}"]
    elif method == 1 and compare_to_exact_results:
        correct_dist = precalculated_data[f"Exact Heisenburg - Qubits{num_qubits}"]
    elif method == 2 and not compare_to_exact_results:
        correct_dist = precalculated_data[f"TFIM - Qubits{num_qubits}"]
    elif method == 2 and compare_to_exact_results:
        correct_dist = precalculated_data[f"Exact TFIM - Qubits{num_qubits}"]
    else:
        raise ValueError("Method is not 1 or 2, or compare_to_exact_results was unexpected type.")

    if verbose:
        print(f"Correct dist: {correct_dist}")

    # Use polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)
    return counts, fidelity

############### Benchmark Loop

def run(min_qubits: int = 2, max_qubits: int = 8, max_circuits: int = 3, skip_qubits: int = 1, num_shots: int = 100,
        use_XX_YY_ZZ_gates: bool = False, backend_id: str = 'qasm_simulator', provider_backend = None,
        hub: str = "ibm-q", group: str = "open", project: str = "main", exec_options = None,
        compare_to_exact_results: bool = False, context = None, method: int = 1):
    """
    Execute program with default parameters.

    Args:
        min_qubits (int): Minimum number of qubits (smallest circuit is 2 qubits).
        max_qubits (int): Maximum number of qubits.
        max_circuits (int): Maximum number of circuits to execute per group.
        skip_qubits (int): Increment of number of qubits.
        num_shots (int): Number of shots for each circuit execution.
        use_XX_YY_ZZ_gates (bool): Flag to use unoptimized XX, YY, ZZ gates.
        backend_id (str): Backend identifier for execution.
        provider_backend: Provider backend instance.
        hub (str): IBM Quantum hub.
        group (str): IBM Quantum group.
        project (str): IBM Quantum project.
        exec_options: Execution options.
        compare_to_exact_results (bool): Flag to compare results to exact simulation.
        context: Execution context.
        method (int): Method for Hamiltonian construction (1 for Heisenberg, 2 for TFIM).

    Returns:
        None
    """
    print(f"{benchmark_name} Benchmark Program - Qiskit")
    
    # Validate parameters (smallest circuit is 2 qubits)
    max_qubits = max(2, max_qubits)
    min_qubits = min(max(2, min_qubits), max_qubits)
    if min_qubits % 2 == 1: min_qubits += 1  # min_qubits must be even
    skip_qubits = max(1, skip_qubits)

    # Create context identifier
    if context is None: context = f"{benchmark_name} Benchmark"
    
    # Set the flag to use an XX YY ZZ shim if given
    global _use_XX_YY_ZZ_gates
    _use_XX_YY_ZZ_gates = use_XX_YY_ZZ_gates
    if _use_XX_YY_ZZ_gates:
        print("... using unoptimized XX YY ZZ gates")
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, type, num_shots):
        # Determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, expectation_a = analyze_and_print_result(qc, result, num_qubits, type, num_shots, method, compare_to_exact_results)
        metrics.store_metric(num_qubits, type, 'fidelity', expectation_a)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):

        # Reset random seed
        np.random.seed(0)

        # Determine number of circuits to execute for this group
        num_circuits = max(1, max_circuits)
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")

        # Parameters of simulation
        #### CANNOT BE MODIFIED W/O ALSO MODIFYING PRECALCULATED DATA #########
        w = precalculated_data['w']  # Strength of disorder
        k = precalculated_data['k']   # Trotter error.
               # A large Trotter order approximates the Hamiltonian evolution better.
               # But a large Trotter order also means the circuit is deeper.
               # For ideal or noise-less quantum circuits, k >> 1 gives perfect Hamiltonian simulation.
        t = precalculated_data['t']  # Time of simulation
        #######################################################################

        # Loop over only 1 circuit
        for circuit_id in range(num_circuits):
            ts = time.time()
            h_x = precalculated_data['h_x'][:num_qubits]  # Precalculated random numbers between [-1, 1]
            h_z = precalculated_data['h_z'][:num_qubits]
            qc = HamiltonianSimulation(num_qubits, K=k, t=t, method=method)
            metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time() - ts)
            qc.draw()
            
            # Collapse the sub-circuits used in this benchmark (for Qiskit)
            qc2 = qc.decompose()

            # Submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc2, num_qubits, circuit_id, num_shots)
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
    
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    # Print a sample circuit
    print("Sample Circuit:")
    print(QC_ if QC_ is not None else "  ... too large!")
   
    if _use_XX_YY_ZZ_gates:
        print("\nXX, YY, ZZ =")
        print(XX_)
        print(YY_)
        print(ZZ_)
    else:
        print("\nXXYYZZ_opt =")
        print(XXYYZZ_)
       
    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - {benchmark_name} - Qiskit")

if __name__ == '__main__': run()
