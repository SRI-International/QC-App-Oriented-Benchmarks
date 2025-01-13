'''
evolution_exact.py - Evolution Exact Functions
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

"""
This module provides functions to classically compute the value of observables during Hamiltonian evolution,
serving as a reliable reference for benchmarking estimates obtained from quantum simulations.

Note that only the first of the compute functions is in use currently and is the most efficient.
It is also independent of any quantum programming API.
The others are maintained here for reference and may be removed later.d 
"""

import copy
from math import sin, cos, pi
import time

import numpy as np
import scipy as sc

import numpy as np
from scipy.linalg import expm

# Set numpy print options to format floating point numbers
np.set_printoptions(precision=3, suppress=True)

verbose = False

##########################################################

# Define the Pauli matrices and the computational basis states for a single qubit.
# Pauli matrices as numpy arrays

I = np.array([[1, 0], [0, 1]], dtype=complex)  # Identity matrix
X = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X matrix
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # Pauli-Y matrix
Z = np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli-Z matrix

# Qubit basis states
Zero = np.array([1, 0], dtype=complex)  # |0⟩ state
One = np.array([0, 1], dtype=complex)   # |1⟩ state

# Define a function to compute the tensor product of multiple matrices or vectors.
def tensor_product(*matrices):
    """Compute the tensor product of a sequence of matrices or vectors."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

# Define a function to generate the initial state vector from a binary string, e.g., '110' for (|110⟩).
def generate_initial_state(state_str):
    """Generate the initial state from a binary string, e.g., '110' for |110⟩."""
    state = [One if bit == '1' else Zero for bit in state_str]
    return tensor_product(*state)
 
# Ensure that the initial state is a valid state vector (array)
def ensure_valid_state(initial_state, num_qubits = None): 

    # if initial_state is None or "", generate |00> state
    if initial_state is None or (isinstance(initial_state, str) and initial_state == ""):
        initial_state = np.zeros((2**num_qubits), dtype=complex)
        initial_state[0] = 1  # Set the amplitude for |00> state
    
    # if initial_state is a string turn it into a vector
    if isinstance(initial_state, str):
        if initial_state == "checkerboard" or initial_state == "neele":
            initial_state = ""
            for k in range(0, num_qubits):
                initial_state += "0" if k % 2 == 1 else "1"
                
            print(f"... initial_state (check) = {initial_state}")
        
        initial_state = generate_initial_state(initial_state)

    # ensure initial_state is a normalized complex vector
    initial_state = np.array(initial_state, dtype=complex)
    initial_state /= np.linalg.norm(initial_state)
    
    return initial_state
    
"""
Compute theoretical energies from Hamiltonian and initial state.
This version is returning an array of classically computed exact energies, one for each step of evolution over time.
"""
   
def compute_theoretical_energies(initial_state, pauli_terms, time, step_size):
    """
    Compute the theoretical energies by evolving a quantum state under a Hamiltonian using NumPy.

    Args:
        initial_state (array-like): Initial state vector as a NumPy array.
        pauli_terms (array): Array of Pauli terms in the form [(term, coeff), ...].
        time (float): Total evolution time.
        step_size (float): Time step size.

    Returns:
        exact_energies (list): Expectation values of the Hamiltonian over time.
    """
    if pauli_terms is None:
        return [None]
        
    if verbose:
        print(f"... compute_theoretical_energies({pauli_terms})")
     
    # Determine the number of qubits
    num_qubits = len(pauli_terms[0][0])  # Length of any Pauli string
    
    # create empty matrix of required size
    H_matrix = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    
    # build the matrix from the pauli terms and coefficients
    for pauli, coeff in pauli_terms:
        for i, p in enumerate(pauli):
            pass

        # Build the tensor product for the current Pauli string
        pauli_matrices = []
        for p in pauli:
            if p == 'I':
                pauli_matrices.append(I)
            elif p == 'X':
                pauli_matrices.append(X)
            elif p == 'Y':
                pauli_matrices.append(Y)
            elif p == 'Z':
                pauli_matrices.append(Z)
            else:
                raise ValueError(f"Invalid Pauli operator: {p}")

        # Compute the tensor product and add to the Hamiltonian
        term_matrix = tensor_product(*pauli_matrices)  
        H_matrix += coeff * term_matrix
    
    # ensure initial_state is a normalized complex vector
    initial_state = ensure_valid_state(initial_state, num_qubits=num_qubits)

    # Define a time mesh
    exact_times = np.arange(0, time + step_size, step_size)

    # Compute the exponential of the Hamiltonian for each time step
    exp_H = expm(-1j * step_size * H_matrix)

    # Initialize the state evolution list
    exact_evolution = [initial_state]

    # Perform the time evolution
    for t in exact_times[1:]:
        print('.', end="")
        
        # Evolve the state using matrix multiplication
        # NOTE: next line could written as next_state = exp_H @ exact_evolution[-1]
        next_state = np.dot(exp_H, exact_evolution[-1])     
        exact_evolution.append(next_state)

    # Compute the expectation values of the Hamiltonian
    exact_energies = [
        np.real(np.vdot(state, np.dot(H_matrix, state))) for state in exact_evolution
    ]

    return exact_energies


##########################################################
##########################################################
# API-DEPENDENT FUNCTIONS

try:
    from qiskit import QuantumCircuit 
    from qiskit_algorithms import TimeEvolutionProblem, SciPyRealEvolver
    from qiskit.quantum_info import Statevector

except Exception as ex:
    print("WARNING: Qiskit-dependent compute observable value functions are not available")
    
# The functions below are not currently used in the benchmarks.
# However, we retain them here for reference and possible future use.

# This function takes a SparsePauliOp and converts to StateVector for evolution
def compute_theoretical_energies_spo_sv(initial_state, H, time, step_size):

    if H is None:
        return [None]
        
    # Create the Hamiltonian matrix (array form)
    H_array = H.to_matrix()
    
    num_qubits = H.num_qubits
    
    #print(f"matrix = {H_array}")
    
    # ensure initial_state is a normalized complex vector
    initial_state = ensure_valid_state(initial_state, num_qubits=num_qubits)

    # need to convert to Statevector so the evolve() function can be used
    initial_state = Statevector(initial_state)
    
    #print(f"... initial state = {initial_state}")

    # We define a slightly denser time mesh
    exact_times = np.arange(0, time+step_size, step_size)
    
    # We compute the exact evolution using the exp
    exact_evolution = [initial_state]
    exp_H = sc.linalg.expm(-1j * step_size * H_array)
    for time in exact_times[1:]:
        print('.', end="")
        exact_evolution.append(exact_evolution[-1].evolve(exp_H))

    # Having the exact state vectors, we compute the exact evolution of our operators’ expectation values.
    exact_energies = np.real([sv.expectation_value(H) for sv in exact_evolution])
    
    return exact_energies

##################################
    
# This function takes a SparsePauliOp and implicitly converts to matrix as needed
def compute_theoretical_energies_spo_mat(initial_state, H, time, step_size):
    """
    Compute the theoretical energies by evolving a quantum state under a Hamiltonian using NumPy.

    Args:
        initial_state (array-like): Initial state vector as a NumPy array.
        H (np.ndarray): Hamiltonian matrix (Hermitian).
        time (float): Total evolution time.
        step_size (float): Time step size.

    Returns:
        list (float): Expectation values of the Hamiltonian over time.
    """
    if H is None:
        return [None]
        
    #print(f"... in cte2, H = \n{H}")
    #print(f"... in cte2, H_matrix = \n{H.to_matrix()}")

    # Ensure initial_state is a normalized complex vector
    initial_state = np.array(initial_state, dtype=complex)
    initial_state /= np.linalg.norm(initial_state)

    #print(f"... initial state = {initial_state}")

    # Define a time mesh
    exact_times = np.arange(0, time + step_size, step_size)

    # Compute the exponential of the Hamiltonian for each time step
    #print(f"... in cte2, -1j * step_size * H = \n{-1j * step_size * H}")
    exp_H = expm(-1j * step_size * H)
    
    #print(f"... in cte2, exp_H = {exp_H}")

    # Initialize the state evolution list
    exact_evolution = [initial_state]

    # Perform the time evolution
    for t in exact_times[1:]:
        print('.', end="")
        
        # Evolve the state using matrix multiplication
        # NOTE: next line could written as next_state = exp_H @ exact_evolution[-1]
        next_state = np.dot(exp_H, exact_evolution[-1])     
        exact_evolution.append(next_state)

    # Compute the expectation values of the Hamiltonian
    exact_energies = [
        np.real(np.vdot(state, np.dot(H, state))) for state in exact_evolution
    ]

    return exact_energies
 
##################################
   
# This function takes a SparsePauliOp and computes observable value from the start at each time step
# It is much slower than the similar but faster compute_theoretical_energies() currently in use

def compute_theoretical_energies_spo_slow(initial_state, H, time, step_size):
    """
    Compute the theoretical energies by evolving a quantum state under a Hamiltonian using NumPy.

    Args:
        initial_state (array-like): Initial state vector as a NumPy array.
        H (np.ndarray): Hamiltonian matrix (Hermitian).
        time (float): Total evolution time.
        step_size (float): Time step size.

    Returns:
        exact_energies (list): Expectation values of the Hamiltonian over time.
    """
    if H is None:
        return [None]

    # Ensure initial_state is a normalized complex vector
    initial_state = np.array(initial_state, dtype=complex)
    initial_state /= np.linalg.norm(initial_state)

    #print(f"... initial state = {initial_state}")

    # Define a time mesh
    exact_times = np.arange(0, time + step_size, step_size)

    # Compute the exponential of the Hamiltonian for each time step
    #exp_H = expm(-1j * step_size * H)

    # Initialize the state evolution list
    exact_evolution = [initial_state]

    # Perform the time evolution
    for t in exact_times[1:]:
        print('.', end="")
        # Evolve the state using matrix multiplication
        #next_state = np.dot(exp_H, exact_evolution[-1])
        U = expm(-1j * H * t)
        next_state = U @ initial_state
        exact_evolution.append(next_state)

    # Compute the expectation values of the Hamiltonian
    exact_energies = [
        np.real(np.vdot(state, np.dot(H, state))) for state in exact_evolution
    ]

    return exact_energies
    
####################################

# This function takes a SparsePauliOp and uses the SciPyRealEvolver to compute the observable value
def compute_theoretical_energies_spo_scipy(initial_state, H, time, step_size):
    """
    Compute the theoretical energies by evolving a quantum state under a Hamiltonian using NumPy.

    Args:
        initial_state (array-like): Initial state vector as a NumPy array.
        H (np.ndarray): Hamiltonian matrix (Hermitian).
        time (float): Total evolution time.
        step_size (float): Time step size.

    Returns:
        exact_energies (list): Expectation values of the Hamiltonian over time.
    """
    if H is None:
        return [None]

    # Ensure initial_state is a normalized complex vector
    initial_state = np.array(initial_state, dtype=complex)
    initial_state /= np.linalg.norm(initial_state)

    #print(f"... initial state = {initial_state}")

    # Define a time mesh
    exact_times = np.arange(0, time + step_size, step_size)
    
    qc_initial = QuantumCircuit(10)
    
    # Initialize the circuit with the given state vector
    qc_initial.initialize(initial_state, qc_initial.qubits)
    
    #time_problem = TimeEvolutionProblem(hamiltonian_op, time, initial_state=qc_initial)
    #time_problem = TimeEvolutionProblem(hamiltonian_op, time, initial_state=initial_state)
    
    # We compute the exact evolution using the exp
    exact_evolution = [Statevector(initial_state)]
     
    #exp_H = sc.linalg.expm(-1j * step_size * H_array)
    for time in exact_times[1:]:
        print('.', end="")
        #exact_evolution.append(exact_evolution[-1].evolve(exp_H))
        
        # Evolve the state using SciPyRealEvolver
        #next_state = np.dot(exp_H, exact_evolution[-1])
        time_problem = TimeEvolutionProblem(H, time, initial_state=qc_initial)
        next_state = SciPyRealEvolver(num_timesteps=1).evolve(time_problem).evolved_state
        #print(next_state)
        #print(f"... exp = {next_state.expectation_value(H)}")
        exact_evolution.append(next_state)

    # Having the exact state vectors, we compute the exact evolution of our operators’ expectation values.
    #exact_energies = np.real([sv.expectation_value(H) for sv in exact_evolution])
    exact_energies = []
    for state in exact_evolution:
        #print(f"... state = {state}")
        exact_energies.append(state.expectation_value(H))
        
    return exact_energies
    
# this is taken from the (deprecated) hamiltonian_exact.py file, for reference
def compute_theoretical_energy_scipy(qc_initial, n_spins: int, hamiltonian_op = None, time: float = 1.0):
    """
    Perform exact Hamiltonian simulation using classical matrix evolution.

    Args:
        n_spins (int): Number of spins (qubits).
        hamiltonian_op (Object): A hamiltonian operator.
        time (float): Duration of simulation, default = 1.0
        init_state (str): The chosen initial state. By default applies the checkerboard state, but can also be set to "ghz", the GHZ state.

    Returns:
        dict: The distribution of the evolved state.
    """
    time_problem = TimeEvolutionProblem(hamiltonian_op, time, initial_state=qc_initial)
    result = SciPyRealEvolver(num_timesteps=1).evolve(time_problem)
    
    exp_value = result.evolved_state.expectation_value(H)
    
    return exp_value
    
    
    
