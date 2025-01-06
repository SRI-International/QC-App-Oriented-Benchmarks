'''
evolution_exact.py - Evolution Exact Functions
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

"""
This module provides functions to classically compute the value of observables during Hamiltonian evolution,
serving as a reliable reference for benchmarking estimates obtained from quantum simulations.
"""

import copy
from math import sin, cos, pi
import time

import numpy as np
import scipy as sc

import numpy as np
from scipy.linalg import expm

from qiskit.quantum_info import Statevector

# Set numpy print options to format floating point numbers
np.set_printoptions(precision=3, suppress=True)

verbose = False
   
"""
Compute theoretical energies from Hamiltonian and initial state.
This version is returning an array of classically computed exact energies, one for each step of evolution over time.
"""
def compute_theoretical_energies(initial_state, H, time, step_size):

    if H is None:
        return [None]
        
    # Create the Hamiltonian matrix (array form)
    H_array = H.to_matrix()
    
    #print(f"matrix = {H_array}")

    # need to convert to Statevector so the evolve() function can be used
    initial_state = Statevector(initial_state)
    
    #print(f"... initial state = {initial_state}")
    
    # use this if string is passed for initialization
    #initial_state = Statevector.from_label("001100")

    # We define a slightly denser time mesh
    exact_times = np.arange(0, time+step_size, step_size)
    
    # We compute the exact evolution using the exp
    exact_evolution = [initial_state]
    exp_H = sc.linalg.expm(-1j * step_size * H_array)
    for time in exact_times[1:]:
        print('.', end="")
        exact_evolution.append(exact_evolution[-1].evolve(exp_H))

    # Having the exact state vectors, we compute the exact evolution of our operators’ expectation values.
    exact_energy = np.real([sv.expectation_value(H) for sv in exact_evolution])
    
    return exact_energy, exact_times
    

def compute_theoretical_energies2(initial_state, H, time, step_size):
    """
    Compute the theoretical energies by evolving a quantum state under a Hamiltonian using NumPy.

    Args:
        initial_state (array-like): Initial state vector as a NumPy array.
        H (np.ndarray): Hamiltonian matrix (Hermitian).
        time (float): Total evolution time.
        step_size (float): Time step size.

    Returns:
        tuple: (exact_energy, exact_times)
            - exact_energy (list): Expectation values of the Hamiltonian over time.
            - exact_times (np.ndarray): Time steps for the evolution.
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
    exact_energy = [
        np.real(np.vdot(state, np.dot(H, state))) for state in exact_evolution
    ]

    return exact_energy, exact_times

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
    
      
def compute_theoretical_energies22(initial_state, H, time, step_size):
    """
    Compute the theoretical energies by evolving a quantum state under a Hamiltonian using NumPy.

    Args:
        initial_state (array-like): Initial state vector as a NumPy array.
        H (np.ndarray): Hamiltonian matrix (Hermitian).
        time (float): Total evolution time.
        step_size (float): Time step size.

    Returns:
        tuple: (exact_energy, exact_times)
            - exact_energy (list): Expectation values of the Hamiltonian over time.
            - exact_times (np.ndarray): Time steps for the evolution.
    """
    if H is None:
        return [None]
        
    #print(f"... in cte22, H = ")
     
    # Determine the number of qubits
    num_qubits = len(H[0][0])  # Length of any Pauli string
    #print(f"... in cte22, num_qubits = {num_qubits}")
    
    #for pauli, coeff in H:
        #print(f"  {pauli}: {coeff}")
    
    H_matrix = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    #print(f"... initial matrix = \n{H_matrix}")
    
    for pauli, coeff in H:
        #print(f"  {pauli}: {coeff}")
        for i, p in enumerate(pauli):
            #print(f"    {i}: {p}")
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
        #print(f"    Term matrix:\n{term_matrix}")
        
        H_matrix += coeff * term_matrix
        
    #print(f"... actual matrix = \n{H_matrix}")
    
    # Ensure initial_state is a normalized complex vector
    initial_state = np.array(initial_state, dtype=complex)
    initial_state /= np.linalg.norm(initial_state)

    #print(f"... initial state = {initial_state}")

    # Define a time mesh
    exact_times = np.arange(0, time + step_size, step_size)

    # Compute the exponential of the Hamiltonian for each time step
    #print(f"... in cte2, -1j * step_size * H_matrix = {-1j * step_size * H_matrix}")
    exp_H = expm(-1j * step_size * H_matrix)
    
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
    exact_energy = [
        np.real(np.vdot(state, np.dot(H_matrix, state))) for state in exact_evolution
    ]

    return exact_energy, exact_times
    
##########################################################
   

def compute_theoretical_energies2x(initial_state, H, time, step_size):
    """
    Compute the theoretical energies by evolving a quantum state under a Hamiltonian using NumPy.

    Args:
        initial_state (array-like): Initial state vector as a NumPy array.
        H (np.ndarray): Hamiltonian matrix (Hermitian).
        time (float): Total evolution time.
        step_size (float): Time step size.

    Returns:
        tuple: (exact_energy, exact_times)
            - exact_energy (list): Expectation values of the Hamiltonian over time.
            - exact_times (np.ndarray): Time steps for the evolution.
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
    exact_energy = [
        np.real(np.vdot(state, np.dot(H, state))) for state in exact_evolution
    ]

    return exact_energy, exact_times
    

from qiskit import QuantumCircuit 
from qiskit_algorithms import TimeEvolutionProblem, SciPyRealEvolver


def compute_theoretical_energies3(initial_state, H, time, step_size):
    """
    Compute the theoretical energies by evolving a quantum state under a Hamiltonian using NumPy.

    Args:
        initial_state (array-like): Initial state vector as a NumPy array.
        H (np.ndarray): Hamiltonian matrix (Hermitian).
        time (float): Total evolution time.
        step_size (float): Time step size.

    Returns:
        tuple: (exact_energy, exact_times)
            - exact_energy (list): Expectation values of the Hamiltonian over time.
            - exact_times (np.ndarray): Time steps for the evolution.
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
    #exact_energy = np.real([sv.expectation_value(H) for sv in exact_evolution])
    exact_energy = []
    for state in exact_evolution:
        #print(f"... state = {state}")
        exact_energy.append(state.expectation_value(H))
        
    return exact_energy, exact_times
    
    

# this is taken from the hamiltonian_exact.py file
def compute_theoretical_energy3(qc_initial, n_spins: int, hamiltonian_op = None, time: float = 1.0):
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
    
    
    
