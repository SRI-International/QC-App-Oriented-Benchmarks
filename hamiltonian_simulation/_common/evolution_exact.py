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
import time as timefuns

import numpy as np
import scipy as sc

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csr_matrix, identity, kron

# Set numpy print options to format floating point numbers
np.set_printoptions(precision=3, suppress=True)

verbose = False

##########################################################
# LINEAR ALGEBRA FUNCTIONS

# Define the Pauli matrices and the computational basis states for a single qubit.
# Pauli matrices as numpy arrays

# Qubit basis states
Zero = csr_matrix([[1], [0]], dtype=complex)  # |0⟩ state as a sparse column vector
One = csr_matrix([[0], [1]], dtype=complex)  # |1⟩ state as a sparse column vector

# Predefine sparse Pauli matrices
I = identity(2, format='csr', dtype=complex)
X = csr_matrix([[0, 1], [1, 0]], dtype=complex)
Y = csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
Z = csr_matrix([[1, 0], [0, -1]], dtype=complex)

# Pauli matrix map for convenience
PAULI_MAP = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

def tensor_product(*matrices):
    """Efficient sparse tensor product."""
    result = matrices[0]
    for m in matrices[1:]:
        result = kron(result, m, format='csr')
    return result


##########################################################
# INITIAL STATE FUNCTIONS
    
def generate_initial_state(state_str):
    """
    Generate the initial state as a sparse vector from a binary string, e.g., '110' for |110⟩.
    
    Args:
        state_str (str): A binary string representing the desired initial state (e.g., '110').

    Returns:
        csr_matrix: A sparse vector representing the initial quantum state.
    """
    # Map '0' to |0⟩ (Zero) and '1' to |1⟩ (One)
    state = [One if bit == '1' else Zero for bit in state_str]
    # Compute the tensor product of the qubits
    return tensor_product(*state)
    
# Ensure that the initial state is a valid state vector (array)
def ensure_valid_state(initial_state, num_qubits = None, reverse = False): 
    # "reverse" only applies to the checkerboard"
    
    # if initial_state is None or "", generate |00> state
    if initial_state is None or (isinstance(initial_state, str) and initial_state == ""):
        initial_state = np.zeros((2**num_qubits), dtype=complex)
        initial_state[0] = 1  # Set the amplitude for |00> state
    
    # if initial_state is a string turn it into a vector
    elif isinstance(initial_state, str):
        if initial_state == "checkerboard" or initial_state == "neele":
            initial_state = ""
            for k in range(0, num_qubits):
                if not reverse:
                    initial_state += "0" if k % 2 == 1 else "1"
                else:
                    initial_state += "1" if k % 2 == 1 else "0"
                    
            initial_state = generate_initial_state(initial_state)
         
            # Convert to dense 1D array for general use; DEVNOTE: could use sparse state later
            initial_state = initial_state.toarray().flatten()
            # dense_state = initial_state.toarray()
            # initial_state = dense_state
            
        elif set(initial_state).issubset({'0', '1'}):
            initial_state = generate_initial_state(initial_state)
            initial_state = initial_state.toarray().flatten()
    
        elif initial_state == "ghz":
            initial_state = np.zeros((2**num_qubits), dtype=complex)
            initial_state[0] = 1/np.sqrt(2)
            initial_state[-1] = 1/np.sqrt(2)

        else:
            raise ValueError(f"Invalid initial state: {initial_state}")
            
    # if initial_state is a list, assume it is a state vector passed in, ready to use
    elif isinstance(initial_state, np.ndarray) or isinstance(initial_state, list):
        pass
        
    else:
        raise ValueError(f"Invalid initial state: {initial_state}")
    
    # ensure initial_state is a normalized complex vector
    initial_state = np.array(initial_state, dtype=complex)
    initial_state /= np.linalg.norm(initial_state)
    
    return initial_state

    
##########################################################
# HAMILTONIAN MATRIX FUNCTIONS
 
def build_hamiltonian(num_qubits, pauli_terms):
    """
    Build the Hamiltonian matrix.
    
    Args:
        num_qubits (int): Number of qubits.
        pauli_terms (list): List of tuples [(pauli_string, coefficient)].

    Returns:
        csr_matrix: The Hamiltonian as a matrix.
    """  

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
        
    return H_matrix


def build_sparse_hamiltonian(num_qubits, pauli_terms):
    """
    Build the Hamiltonian as a sparse matrix.
    
    Args:
        num_qubits (int): Number of qubits.
        pauli_terms (list): List of tuples [(pauli_string, coefficient)].

    Returns:
        csr_matrix: The Hamiltonian as a sparse matrix.
    """
    # Start with an empty sparse matrix
    H_sparse = csr_matrix((2**num_qubits, 2**num_qubits), dtype=complex)

    # Build the Hamiltonian
    for pauli, coeff in pauli_terms:
        # Build the tensor product for the current Pauli string
        pauli_matrices = [PAULI_MAP[p] for p in pauli]
        term_matrix = tensor_product(*pauli_matrices)

        # Add the term to the Hamiltonian
        H_sparse += coeff * term_matrix

    return H_sparse
  

##########################################################
# COMPUTE EXACT EXPECTATION FUNCTIONS 
    
"""
Compute theoretical energies from Hamiltonian and initial state.
This version is returning an array of classically computed exact energies, one for each step of evolution over time.

NOTE: there is some issue with the bit order.  The checkerboard init_state needs to be reversed.
250125: this needs to be invistigated.  The older scipy_spo version works the other way.
We need to also check on the |00> and other string states to see if they should be reversed also.
"""
   
def compute_expectations_exact(initial_state, pauli_terms, time, step_size):
    """
    Compute the theoretical energies by evolving a quantum state under a Hamiltonian using NumPy.

    Args:
        initial_state (array-like): Initial state vector as a NumPy array.
        pauli_terms (array): Array of Pauli terms in the form [(term, coeff), ...].
        time (float): Total evolution time.
        step_size (float): Time step size.

    Returns:
        exact_expectations (list): Expectation values of the Hamiltonian over time.
    """
    if pauli_terms is None:
        return [None]
        
    if verbose:
        print(f"... compute_expectations_exact({pauli_terms})")
     
    # Determine the number of qubits
    num_qubits = len(pauli_terms[0][0])  # Length of any Pauli string

    ts = timefuns.time()
    
    # Build the sparse Hamiltonian
    H_matrix = build_sparse_hamiltonian(num_qubits, pauli_terms)
    
    matrix_time = round(timefuns.time()-ts, 3)
    if verbose:
        print(f"... matrix creation time = {matrix_time} sec")
    
    # ensure initial_state is a normalized complex vector
    initial_state = ensure_valid_state(initial_state, num_qubits=num_qubits, reverse=True)

    initial_time = round(timefuns.time()-ts, 3)
    if verbose:
        print(f"... matrix and initial state time = {initial_time} sec")

    # Define a time array
    exact_times = np.arange(0, time + step_size, step_size)
     
    # Assume H_matrix and initial_state are defined
    H_matrix_sparse = csr_matrix(H_matrix)  # Convert H to sparse format if large
    
    # Compute the exponential of the Hamiltonian for each time step  
    exact_evolution = expm_multiply(
            -1j * step_size * H_matrix_sparse, initial_state,
            start = 0,
            stop = time + step_size,
            num = exact_times.size
            )

    # Compute the expectation values of the Hamiltonian at multiple steps
    exact_expectations = []
    for evolved_state in exact_evolution:
    
        # Compute expectation value
        H_psi = H_matrix_sparse @ evolved_state  # Sparse matrix-vector product
        exact_expectation = np.real(np.vdot(evolved_state, H_psi))  # Inner product 
    
        exact_expectations.append(exact_expectation)
        
    return exact_expectations

"""
Compute theoretical energy from Hamiltonian and initial state.
This version is returning a single classically computed exact energy, and the asociated distribution.
"""
   
def compute_expectation_exact(initial_state, pauli_terms, time):
    """
    Compute the theoretical energy by evolving a quantum state under a Hamiltonian using NumPy.

    Args:
        initial_state (array-like): Initial state vector as a NumPy array.
        pauli_terms (array): Array of Pauli terms in the form [(term, coeff), ...].
        time (float): Total evolution time.

    Returns:
        exact_expectation (float): Expectation value of the Hamiltonian over given time.
        probability_distribution (dict): Probability distribution of the Hamiltonian at the given time.
    """
    if pauli_terms is None:
        return [None]
        
    if verbose:
        print(f"... compute_expectation_exact({pauli_terms})")
     
    # Determine the number of qubits
    num_qubits = len(pauli_terms[0][0])  # Length of any Pauli string
    
    ts = timefuns.time()
    
    # Build the sparse Hamiltonian
    H_matrix = build_sparse_hamiltonian(num_qubits, pauli_terms)
    
    matrix_time = round(timefuns.time()-ts, 3)
    if verbose:
        print(f"... matrix creation time = {matrix_time} sec")
    
    # ensure initial_state is a normalized complex vector
    initial_state = ensure_valid_state(initial_state, num_qubits=num_qubits, reverse=True)

    initial_time = round(timefuns.time()-ts, 3)
    if verbose:
        print(f"... matrix and initial state time = {initial_time} sec")
    
    # Compute the exponential of the Hamiltonian for each time step  
    # Assume H_matrix and initial_state are defined
    H_matrix_sparse = H_matrix  # this is a sparse matrix
    
    # evolve the initial state by this matrix
    evolved_state = expm_multiply(-1j * time * H_matrix_sparse, initial_state)
    
    # Compute expectation value
    H_psi = H_matrix_sparse @ evolved_state  # Sparse matrix-vector product
    exact_expectation = np.real(np.vdot(evolved_state, H_psi))  # Inner product 
    
    # Compute the probabilities as the squared magnitudes of the state vector
    probabilities = np.abs(evolved_state)**2
    
    # Create a dictionary keyed by bitstrings
    probability_distribution = {
        format(i, f'0{num_qubits}b'): prob for i, prob in enumerate(probabilities)
    }
    
    # values in distribution items are arrays of len 1; collapse to single values
    for bitstring, count in probability_distribution.items():
        probability_distribution[bitstring] = count
        
    return exact_expectation, probability_distribution


##########################################################
##########################################################
# API-DEPENDENT FUNCTIONS

try:
    from qiskit import QuantumCircuit 
    from qiskit.quantum_info import Statevector

except Exception as ex:
    print("WARNING: Qiskit-dependent exact evolution functions are not available")
    
# The functions below are not currently used in the benchmarks.
# However, we retain them here for reference and possible future use.

# This function takes a SparsePauliOp and converts to StateVector for evolution
def compute_expectations_exact_spo_sv(initial_state, H, time, step_size):

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
    exact_expectations = np.real([sv.expectation_value(H) for sv in exact_evolution])
    
    return exact_expectations

##################################
    
# This function takes a SparsePauliOp and implicitly converts to matrix as needed
def compute_expectations_exact_spo_mat(initial_state, H, time, step_size):
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
    exact_expectations = [
        np.real(np.vdot(state, np.dot(H, state))) for state in exact_evolution
    ]

    return exact_expectations
 
##################################
   
# This function takes a SparsePauliOp and computes observable value from the start at each time step
# It is much slower than the similar but faster compute_expectations_exact() currently in use

def compute_expectations_exact_spo_slow(initial_state, H, time, step_size):
    """
    Compute the theoretical energies by evolving a quantum state under a Hamiltonian using NumPy.

    Args:
        initial_state (array-like): Initial state vector as a NumPy array.
        H (np.ndarray): Hamiltonian matrix (Hermitian).
        time (float): Total evolution time.
        step_size (float): Time step size.

    Returns:
        exact_expectations (list): Expectation values of the Hamiltonian over time.
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
    exact_expectations = [
        np.real(np.vdot(state, np.dot(H, state))) for state in exact_evolution
    ]

    return exact_expectations
    