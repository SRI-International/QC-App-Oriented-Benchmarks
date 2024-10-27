'''
Observable Computation Function Library - Qiskit Version
(C) Quantum Economic Development Consortium (QED-C) 2024.

This module includes helper funtions for computing observables from a Hamiltonian.
'''

from qiskit.quantum_info import Pauli, SparsePauliOp
from itertools import combinations
import numpy as np

'''
import numpy as np
import copy

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.primitives import Estimator

from qiskit.quantum_info import Operator, Pauli

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
'''

verbose = False

noise_model = None

# ===========================================
# CONVERT TO SPARSE PAULI OP

def convert_to_sparse_pauli_op(pauli_terms):
    """
    Convert an array of (coefficient, pauli string) tuples into a SparsePauliOp.
    
    Args:
    pauli_terms (list): List of tuples, each containing (coeff, pauli)
    
    Returns:
    SparsePauliOp: Qiskit SparsePauliOp representation of the Hamiltonian
    """
         
    if (pauli_terms is None):
        return None
    
    coeffs = []
    paulis = []

    for coeff, pauli_string in pauli_terms:
        coeffs.append(coeff)
        paulis.append(pauli_string)
                
    #print(paulis)
    #print(coeffs)
    
    return SparsePauliOp(paulis, coeffs)
    
# ===========================================
# SWAP COEFF and PAULI STRING IN PAULI_LIST
  
def swap_pauli_list(pauli_terms):
    """
    Swap the (pauli string, coefficient) components of each pauli term in the given list.
    
    Args:
    pauli_terms (list): List of tuples, each containing (pauli, coeff)
    
    Returns:
    pauli_terms (list): A copy of the list of tuples, containing swapped (coeff, pauli)
    """
    
    if (pauli_terms is None):
        return None
    
    new_pauli_terms = []    
    for pauli_term in pauli_terms:
        new_pauli_term = (pauli_term[1], pauli_term[0])
        new_pauli_terms.append(new_pauli_term)
        
    return new_pauli_terms

# ===========================================
# COMMUTING GROUP FUNCTIONS

#Extract Qubit-wise Commuting Groups from a Hamiltonian

def pauli_commutes(pauli1, pauli2):
    """
    Check if two Pauli operators commute using their string representations.
    
    Args:
    pauli1 (SparsePauliOp): First Pauli operator.
    pauli2 (SparsePauliOp): Second Pauli operator.
    
    Returns:
    bool: True if pauli1 and pauli2 commute, False otherwise.
    """
    pauli_str1 = pauli1.paulis[0].to_label()
    pauli_str2 = pauli2.paulis[0].to_label()
    
    for i in range(len(pauli_str1)):
        if pauli_str1[i] != 'I' and pauli_str2[i] != 'I' and pauli_str1[i] != pauli_str2[i]:
            return False
    return True

def build_commutativity_graph(pauli_list):
    """
    Build a commutativity graph where nodes represent Pauli operators,
    and edges represent commutation relations between them.
    
    Args:
    pauli_list (list of SparsePauliOp): List of Pauli operators.
    
    Returns:
    np.ndarray: Adjacency matrix of the commutativity graph.
    """
    num_terms = len(pauli_list)
    adjacency_matrix = np.zeros((num_terms, num_terms), dtype=bool)
    
    for i, j in combinations(range(num_terms), 2):
        if pauli_commutes(pauli_list[i], pauli_list[j]):
            adjacency_matrix[i, j] = True
            adjacency_matrix[j, i] = True
    
    return adjacency_matrix

# Function to group commuting terms - this version does not produce qubit-wise groups
# The third and fith Pauli strings do not commute. The adjacency matrix indicates that they do not.
# So, why do they appear in the same group? This code is buggy

def group_commuting_terms(pauli_list):
    """
    Group commuting Pauli terms.
    
    Args:
    pauli_list (list): List of tuples where each tuple contains a Pauli string and a coefficient.
    
    Returns:
    list: List of groups where each group is a list of commuting Pauli terms.
    """
    # Convert the list of Pauli strings to SparsePauliOp objects
    paulis = [SparsePauliOp.from_list([(p, 1)]) for p, coeff in pauli_list]

    print(paulis)
    
    # Build the commutativity graph
    adjacency_matrix = build_commutativity_graph(paulis)

    print(adjacency_matrix)
    
    # Group commuting terms
    groups = []
    ungrouped_indices = set(range(len(paulis)))
    
    while ungrouped_indices:
        current_group = []
        i = ungrouped_indices.pop()
        current_group.append(pauli_list[i])
        
        for j in ungrouped_indices.copy():
            if adjacency_matrix[i, j]:
                current_group.append(pauli_list[j])
                ungrouped_indices.remove(j)
        
        groups.append(current_group)
    
    return groups

# Function to group commuting terms into qubit-wise commuting groups
def group_commuting_terms_2(pauli_list):
    """
    Group commuting Pauli terms.
    
    Args:
    pauli_list (list): List of tuples where each tuple contains a Pauli string and a coefficient.
    
    Returns:
    list: List of groups where each group is a list of commuting Pauli terms.
    """
    # Convert the list of Pauli strings to SparsePauliOp objects
    paulis = [SparsePauliOp.from_list([(p, 1)]) for p, coeff in pauli_list]

    print(paulis)
    
    # Build the commutativity graph
    adjacency_matrix = build_commutativity_graph(paulis)

    print(adjacency_matrix)
    
    # Group commuting terms
    groups = []
    ungrouped_indices = set(range(len(paulis)))

    # loop over all the terms, looking for those that can be grouped
    while ungrouped_indices:
        current_group = []
        current_group_indices = []

        # make the first item in the ungrouped list into the new current group
        i = ungrouped_indices.pop()
        current_group.append(pauli_list[i])
        current_group_indices.append(i)
        #print(f"\n... visit ungrouped_index: {i} {pauli_list[i]}")

        # we need to check that the candidate term commutes with all the terms in the current group
        for j in ungrouped_indices.copy():
            #print(f"  ... compare to ungrouped_index: {j} {pauli_list[j]}")

            commuting = True
            for k in current_group_indices:
                #print(f"    ... checking against: {k} {pauli_list[k]}")
                if not adjacency_matrix[k, j]:
                    commuting = False
                    print(f"    ... conflict, do not add to this group")
                    break

            if commuting:
                #print(f"    ... commutes, add to this grouop")
                current_group.append(pauli_list[j])
                current_group_indices.append(j)
                ungrouped_indices.remove(j)
        
        groups.append(current_group)
    
    return groups




# ===========================================
# Execution Functions

# Here we define a function to perform the execution and return an array of measurements.
# DEVNOTE: params argument not currently used, as the circuit is created with values defined; we don't use Parameter yet

# Execute a quantum circuit with the specified parameters on the given backend system
def execute_circuit(qc, backend, num_shots, params):
     
    # Execute the quantum circuit to obtain the probability distribution associated with the current parameters
    result = backend.run(qc, shots=num_shots, noise_model=noise_model).result()
    
    # get the measurement counts from the result object
    counts = result.get_counts(qc)
    
    # For the statevector simulator, need to scale counts to num_shots
    if str(backend) == 'statevector_simulator':
        for k, v in counts.items():
            counts[k] = round(v * num_shots)
    
    return counts
    
# Define a function to compute the energy associated with the current set of paramters and 
# one term of the Hamiltonian for the problem.
# Also define a function to append basis rotations to the ansatz circuit for one term of the Hamiltonian operator.
# The expecation_value function performs a computation of the energy of a single Pauli term of a Hamiltonian
# from the measurements obtained from execution of the parameterized circuit.

# Compute the expectation value from measurement results with respect to a single Pauli operator 
def expectation_value(counts, nshots, pPauli):
    
    """
    counts: Measured bistring counts. e.g. {'01':500, '10':500}
    nshots: Total number of shots
    Pauli:  The Pauli operator e.g. "XX"
    """
    
    # initialize expectation value
    exp_val = 0.
    
    # loop over measurement results
    for measurement in counts:
        # local parity
        loc_parity = 1.
        # loop over qubits
        for pauli_ind, pauli in enumerate(reversed(pPauli)):
            # skip identity
            if pauli == 'I':
                continue
            # parity
            loc_parity *= (-2 * float(measurement[::-1][pauli_ind]) + 1) # this turns 0 -> 1, 1 -> -1
        
        # accumulate expectation value
        exp_val += loc_parity * counts[measurement]
    
    # normalization
    exp_val /= nshots
    
    return exp_val



# Append basis rotations to the ansatz circuit for one term of the Hamiltonian operator
def append_hamiltonian_term_to_circuit(qc, params, pauli):

    # determine number of qubits from length of the pauli (this might need to improve)
    nqubit = len(pauli)

    # append the basis rotations as needed to apply the pauli operator
    is_diag = True     # whether this term is diagonal
    for ii, p in enumerate(pauli):     
        target_qubit = nqubit - ii - 1 
        if (p == "X"):
            is_diag = False
            qc.h(target_qubit)
        elif (p == "Y"):
            qc.sdg(target_qubit)
            qc.h(target_qubit)
            is_diag = False
            
# =========================================================================================
# ESTIMATE EXPECTATION CALUE

# Estimate Expectation Value for Circuit with Hamiltonian

# Function to estimate expectation value for an array of weighted Pauli strings
def estimate_expectation(backend, qc, H_terms, num_shots=10000):
    
    # Function to estimate expectation value of a Pauli string
    def estimate_expectation_term(backend, qc, pauli_string, num_shots=10000):

        # Make a clone of the original circuit since we append gates
        # (may not be a reliable way to clone)
        qc = qc.copy()

        # append the gates for the current Pauli string
        append_hamiltonian_term_to_circuit(qc, None, pauli_string)
    
        # Add measure gates
        qc.measure_all()
        
        if verbose: print(f"... circuit with Pauli {pauli_string} =\n{qc}")

        # execute the circuit on the backend to obtain counts
        counts = execute_circuit(qc, backend, num_shots, None)
        if verbose: print(f"... counts = {counts}")

        # from the counts and pauli_string, compute the expectation
        expectation = expectation_value(counts, num_shots, pauli_string)
        
        return expectation
    
    # Measure energy
    total_energy = 0
    for coeff, pauli_string in H_terms: 
        exp_val = estimate_expectation_term(backend, qc, pauli_string)
        total_energy += coeff * exp_val
        if verbose: print(f"... exp value for pauli term = ({coeff}, {pauli_string}), exp = {exp_val}")

    return total_energy
    
# Function to estimate expectation value for an array of weighted Pauli strings
def estimate_expectation2(backend, qc, H_terms_multiple, num_shots=10000):
    
    # Function to estimate expectation value of a Pauli string
    def estimate_expectation_term(backend, qc, pauli_string, num_shots=10000):

        # Make a clone of the original circuit since we append gates
        # (may not be a reliable way to clone)
        qc = qc.copy()

        # append the gates for the current Pauli string
        append_hamiltonian_term_to_circuit(qc, None, pauli_string)
    
        # Add measure gates
        qc.measure_all()
        
        if verbose: print(f"... circuit with Pauli {pauli_string} =\n{qc}")

        # execute the circuit on the backend to obtain counts
        counts = execute_circuit(qc, backend, num_shots, None)
        if verbose: print(f"... counts = {counts}")

        # from the counts and pauli_string, compute the expectation
        expectation = expectation_value(counts, num_shots, pauli_string)
        
        return expectation
    
    # Storage for observables
    observables_store = []
    # Storage of observables in dictionary format in a list
    H_observables = []
    
    # Initialize the expectation values to 0.
    for i in range(len(H_terms_multiple)):
        observables_store.append(0)
    
    # Make dictionaries of Observables and store it in a list.
    for j in range(len(H_terms_multiple)):
        H_observables.append({pauli_term: coeff for coeff, pauli_term in H_terms_multiple[j]})
    
    #Iterate through each terms in Hamiltonian, which is the first element in H_observables.
    for pauli_string, coeff in H_observables[0].items():  
        
        exp_val = estimate_expectation_term(backend, qc, pauli_string)
              
        for i in range(1, len(H_observables)):
            if pauli_string in H_observables[i]:
                observables_store[i] += H_observables[i][pauli_string] * exp_val

        
        observables_store[0] += coeff * exp_val
            
        if verbose: print(f"... exp value for pauli term = ({coeff}, {pauli_string}), exp = {exp_val}")

    return observables_store

# Estimate Expectation Value for Circuit with Hamiltonian -- using Estimator Class 

# Function to estimate expectation value for an array of weighted Pauli strings, using Estimator class
def estimate_expectation_with_estimator(backend, qc, H_terms, num_shots=10000):

    #print(f"... in estimate_expectation_with_estimator()")

    # Convert to SparsePauliOp
    H_op = convert_to_sparse_pauli_op(H_terms)

    # Create an Estimator
    estimator = Estimator()
    
    # Measure energy
    job = estimator.run(qc, H_op)
    result = job.result()
    
    measured_energy = result.values[0]
    
    return measured_energy
    