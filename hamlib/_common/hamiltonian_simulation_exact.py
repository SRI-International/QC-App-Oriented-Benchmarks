import sys
import os

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit"]
sys.path[1:1] = ["../qiskit"]

from qiskit import QuantumCircuit 
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import TimeEvolutionProblem, SciPyRealEvolver
from qiskit_aer import Aer
from qiskit import transpile
import numpy as np
np.random.seed(0)
import execute as ex
import metrics as metrics

from hamlib_simulation_kernel import initial_state, create_circuit

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
    g = 1  # Strength of the transverse field

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

def construct_heisenberg_hamiltonian(n_spins: int, w: int, hx: list[float], hz: list[float]) -> SparsePauliOp:
    """
    Construct the Heisenberg Hamiltonian with disorder.

    Args:
        n_spins (int): Number of spins (qubits).
        w (float): Strength of two-qubit interactions for heisenberg hamiltonian. 
        hx (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 
        hz (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 

    Returns:
        SparsePauliOp: The Hamiltonian represented as a sparse Pauli operator.
    """

    pauli_strings = []
    coefficients = []

    # Disorder terms
    for i in range(n_spins):
        x_term = 'I' * i + 'X' + 'I' * (n_spins - i - 1)
        z_term = 'I' * i + 'Z' + 'I' * (n_spins - i - 1)
        pauli_strings.append(x_term)
        coefficients.append(w * hx[i])
        pauli_strings.append(z_term)
        coefficients.append(w * hz[i])

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

def construct_hamiltonian(n_spins: int, hamiltonian: str, w: float, hx : list[float], hz: list[float]) -> SparsePauliOp:
    """
    Construct the Hamiltonian based on the specified method.

    Args:
        n_spins (int): Number of spins (qubits).
        hamiltonian (str): Which hamiltonian to run. "Heisenberg" by default but can also choose "TFIM". 
        w (float): Strength of two-qubit interactions for heisenberg hamiltonian. 
        hx (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 
        hz (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 

    Returns:
        SparsePauliOp: The constructed Hamiltonian.
    """

    hamiltonian = hamiltonian.strip().lower()

    if hamiltonian == "heisenberg":
        return construct_heisenberg_hamiltonian(n_spins, w, hx, hz)
    elif hamiltonian == "tfim":
        return construct_TFIM_hamiltonian(n_spins)
    else:
        raise ValueError("Invalid Hamiltonian specification.")

def HamiltonianSimulationExact(n_spins: int, init_state=None):
    """
    Perform exact Hamiltonian simulation using classical matrix evolution.

    Args:
        n_spins (int): Number of spins (qubits).
        t (float): Duration of simulation.
        init_state (str): The chosen initial state. By default applies the checkerboard state, but can also be set to "ghz", the GHZ state.
        hamiltonian (str): Which hamiltonian to run. "Heisenberg" by default but can also choose "TFIM". 
        w (float): Strength of two-qubit interactions for heisenberg hamiltonian. 
        hx (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 
        hz (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 

    Returns:
        dict: The distribution of the evolved state.
    """
    _, hamiltonian_sparse, _ = create_circuit(n_spins=n_spins, init_state=init_state)
    time_problem = TimeEvolutionProblem(hamiltonian_sparse, 0.2, initial_state=initial_state(n_spins, init_state))
    result = SciPyRealEvolver(num_timesteps=1).evolve(time_problem)
    
    # if verbose:
    #   print (result)
    
    return result.evolved_state.probabilities_dict()

def HamiltonianSimulationExact_Noiseless(n_spins: int, init_state=None):
    """
    Simulate a quantum Hamiltonian circuit for a specified number of spins using a noiseless quantum simulator.
    
    This function creates a quantum circuit, transpiles it for optimal execution, and runs it on a quantum simulator.
    It simulates the circuit with a specified number of shots and collects the results to compute the probability 
    distribution of measurement outcomes, normalized to represent probabilities.
    
    Args:
        n_spins (int): The number of spins (qubits) to simulate in the circuit.
    
    Returns:
        dict: A dictionary with keys representing the outcomes and values representing the probabilities of these outcomes.
    
    Note:
        This function uses the 'qasm_simulator' backend from Qiskit's Aer module, which simulates a quantum circuit 
        that measures qubits and returns a count of the measurement outcomes. The function assumes that the circuit 
        creation and the simulator are perfectly noiseless, meaning there are no errors during simulation.
    """
    qc, _, _ = create_circuit(n_spins=n_spins,init_state=init_state)
    num_shots = 100000
    backend = Aer.get_backend("qasm_simulator")
    # Transpile and run the circuits
    transpiled_qc = transpile(qc, backend, optimization_level=0)
    job = backend.run(transpiled_qc, shots=num_shots)
    result = job.result()
    counts = result.get_counts(qc)
    # Normalize probabilities for Heisenberg model circuit 
    dist = {}
    for key in counts.keys():
        prob = counts[key] / num_shots
        dist[key] = prob

    return dist