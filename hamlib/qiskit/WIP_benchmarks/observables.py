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

    if verbose: print(paulis)
    
    # Build the commutativity graph
    adjacency_matrix = build_commutativity_graph(paulis)

    if verbose: print(adjacency_matrix)
    
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
    paulis = [SparsePauliOp.from_list([(p, coeff)]) for p, coeff in pauli_list]

    if verbose: print(paulis)
    
    # Build the commutativity graph
    adjacency_matrix = build_commutativity_graph(paulis)

    if verbose: print(adjacency_matrix)
    
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
                    if verbose: print(f"    ... conflict, do not add to this group")
                    break

            if commuting:
                #print(f"    ... commutes, add to this grouop")
                current_group.append(pauli_list[j])
                current_group_indices.append(j)
                ungrouped_indices.remove(j)
        
        groups.append(current_group)
    
    return groups



# ===========================================
# EXECUTION FUNCTIONS

from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
from qiskit.primitives import Estimator
import numpy as np

# Initialize the backend and the simulator
backend = Aer.get_backend('qasm_simulator')

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

# =========================================================================================
# CIRCUIT CREATION FUNCTIONS

def create_circuits_for_hamiltonian(num_qubits, qc, ham_terms, use_commuting_groups=True):
    """
    Creates quantum circuits for a Hamiltonian, with optional optimization using commuting groups.
    Note: this version of the function creates only the rotation portion of the circuit.
    
    Args:
        num_qubits (int): The number of qubits in the circuit.
        qc (QuantumCircuit): The circuit to which we will append rotation gates.
        ham_terms (list of tuples): The Hamiltonian represented as a list of tuples, 
                                    where each tuple contains a Pauli string and a coefficient.
        use_commuting_groups (bool): If True, groups commuting terms to optimize the circuit creation.

    Returns:
        list of tuples: A list where each element is a tuple (QuantumCircuit, group or [(term, coeff)]).
    """
    
    if not use_commuting_groups:
        print("\n******** creating circuits from Hamiltonian:")
        circuits = create_circuits_for_ham_terms(num_qubits, qc, ham_terms)
    else:
        print("\n******** creating commuting groups for the Hamiltonian and circuits from the groups:")
        groups = group_commuting_terms_2(ham_terms)
        for i, group in enumerate(groups):
            print(f"Group {i+1}:")
            for pauli, coeff in group:
                print(f"  {pauli}: {coeff}")
        circuits = create_circuits_for_grouped_terms(num_qubits, qc, groups)

    print(f"\n... constructed {len(circuits)} circuits for this Hamiltonian.")
    return circuits

def create_circuits_for_ham_terms(num_qubits, qc, ham):
    """
    Creates quantum circuits for measuring terms in a raw Hamiltonian.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        qc (QuantumCircuit): The circuit to which we will append rotation gates.
        ham (list of tuples): The Hamiltonian represented as a list of tuples, 
                              where each tuple contains a Pauli string and a coefficient.

    Returns:
        list of tuples: A list where each element is a tuple (QuantumCircuit, [(term, coeff)]).
    """
    circuits = []

    for term, coeff in ham:
        print(f"  ... {term}, {coeff}")
        
        if qc is None:
            qc = QuantumCircuit(num_qubits)
            
        # Make a clone of the original circuit since we append gates
        qc2 = qc.copy()

        append_hamiltonian_term_to_circuit(qc2, None, term)

        qc2.measure_all()
        circuits.append((qc2, [(term, coeff)]))

    return circuits

def create_circuits_for_grouped_terms(num_qubits, qc, groups):
    """
    Creates quantum circuits for groups of commuting terms in a Hamiltonian.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        qc (QuantumCircuit): The circuit to which we will append rotation gates.
        groups (list of list of tuples): A list of groups, where each group is a list of tuples 
                                         (term, coeff) representing commuting Hamiltonian terms.

    Returns:
        list of tuples: A list where each element is a tuple (QuantumCircuit, group).
    """
    circuits = []
    for group in groups:
    
        if qc is None:
            qc = QuantumCircuit(num_qubits)
            
        # Make a clone of the original circuit since we append gates
        qc2 = qc.copy()

        # Merge Pauli terms into a single string to create one circuit per group
        merged_paulis = ['I'] * num_qubits
        for term, coeff in group:
            for i, pauli in enumerate(term):
                if pauli != "I":
                    merged_paulis[i] = pauli

        merged_term = "".join(merged_paulis)

        append_hamiltonian_term_to_circuit(qc2, None, merged_term)

        qc2.measure_all()
        circuits.append((qc2, group))

    return circuits

def append_hamiltonian_term_to_circuit(qc, params, pauli):
    """
    Appends basis rotations to a quantum circuit for a given term of the Hamiltonian operator.

    Args:
        qc (QuantumCircuit): The quantum circuit to which the operations are appended.
        params (None): Unused parameter, reserved for potential future use.
        pauli (str): A string representing the Pauli operator (e.g., 'X', 'Y', 'Z', 'I') for each qubit.

    Returns:
        None
    """
    is_diag = True  # Tracks if the term is diagonal (currently unused)
    for i, p in enumerate(pauli):
        if p == "X":
            is_diag = False
            qc.h(i)
        elif p == "Y":
            qc.sdg(i)
            qc.h(i)
            is_diag = False

###################################################################
# CALCULATE EXPECTATION VALUE 

# Calculate expectation value, or total_energy, from execution results
# This function operates on tuples of (circuit, group)
#def calculate_expectation(num_qubits, results, circuits, is_commuting=False):
def calculate_expectation(num_qubits, results, circuits):
    total_energy = 0
    
    # loop over each group, to accumulate observables in the the terms of the group
    for (qc, group), result in zip(circuits, results.get_counts()):
    
        counts = result
        num_shots = sum(counts.values())
        ###print(f"... got num_shots = {num_shots}: counts = {counts}")

        # process each term in the current group
        for term, coeff in group:
            ###print(f"--> for term: {term}, {coeff}")
            
            # Calculate the expectation value for each term
            exp_val = get_expectation_term(term, counts)
            
            total_energy += coeff * exp_val
            ###print(f"  ******* exp_val = {exp_val} {coeff * exp_val}")
                
    return total_energy
    
# =========================================================================================
# ESTIMATE EXPECTATION VALUE   

# Estimate Expectation Value for Circuit with Hamiltonian

# Function to estimate expectation value for a list of weighted Pauli strings
def estimate_expectation(backend, qc, H_terms, num_shots=10000):
    
    # Measure energy
    total_energy = 0
    
    # Iterate through terms of the first Hamiltonian and accumulate expectation for energy
    for coeff, pauli_string in H_terms: 
    
        exp_val = estimate_expectation_term(backend, qc, pauli_string, num_shots=num_shots)
        total_energy += coeff * exp_val
        if verbose: print(f"... exp value for pauli term = ({coeff}, {pauli_string}), exp = {exp_val}")

    return total_energy
    
# Function to estimate expectation value for a list of weighted Pauli strings
# Note: This version accepts a list of Pauli term lists,
# the first of which is the primary energy Hamiltonian. 
# Other entries in the list are collections of terms for observables that are subsets of the primary.
# The expectation value for these are calculated using the same measurment results as the primary.
def estimate_expectation_multiple(backend, qc, H_terms_multiple, num_shots=10000):
    
    # Storage for observables
    observables_store = []
    # Storage of observables in dictionary format in a list
    H_observables = []
    
    # Initialize the expectation value of each Observable to 0.
    for i in range(len(H_terms_multiple)):
        observables_store.append(0)
    
    # For each Observable, make a dictionary of its terms, keyed by pauli_string
    for j in range(len(H_terms_multiple)):
        H_observables.append({pauli_term: coeff for coeff, pauli_term in H_terms_multiple[j]})
    
    # Iterate through terms of the first Hamiltonian and accumulate expectation for Observables
    for pauli_string, coeff in H_observables[0].items():
        
        # accumulate expectation for the primary Hamiltonian (first in list)
        exp_val = estimate_expectation_term(backend, qc, pauli_string, num_shots=num_shots)
        observables_store[0] += coeff * exp_val
        
        # check the remaining observables; if this pauli_string is in a term, accumulate the value
        for i in range(1, len(H_observables)):
            if pauli_string in H_observables[i]:
                observables_store[i] += H_observables[i][pauli_string] * exp_val
  
        if verbose: print(f"... exp value for pauli term = ({coeff}, {pauli_string}), exp = {exp_val}")

    return observables_store

# Define a function to compute the energy associated with the current set of paramters and 
# one term of the Hamiltonian for the problem.
# The expecation_value function performs a computation of the energy of a single Pauli term of a Hamiltonian
# from the measurements obtained from execution of the parameterized circuit.

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
    #expectation = expectation_value(counts, num_shots, pauli_string)
    expectation = get_expectation_term(pauli_string, counts)
    
    return expectation

"""
# Compute the expectation value from measurement results with respect to a single Pauli operator 
def expectation_value(counts, nshots, pPauli):
   
    # initialize expectation value
    exp_val = 0.
    
    # loop over measurement results
    for measurement in counts:
        # local parity
        loc_parity = 1.
        
        # loop over bits of the pauli string
        for pauli_ind, pauli in enumerate(pPauli):
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

# Calculate the expectation value for each term
"""
"""
# Compute the expectation value from measurement results with respect to a single Pauli operator 
def get_expectation_term(term, counts):
    exp_val = 0
    total_counts = sum(counts.values()) # total number of shots
    num_qubits = len(term)              # Total number of qubits

    # loop over measurement results
    for bitstring, count in counts.items():
        parity = 1.0  # Initialize parity for this bitstring

        # loop over each element of the pauli term
        for qubit_index, pauli in enumerate(term):
        
            # skip identity
            if pauli == 'I':
                continue
                
            # Map qubit index to bitstring index (little-endian) and get bit value
            bit_index = num_qubits - 1 - qubit_index
            bit_value = int(bitstring[bit_index])

            # Map bit_value to eigenvalue: 0 -> +1, 1 -> -1
            eigenvalue = 1 - 2 * bit_value
            parity *= eigenvalue

        exp_val += parity * count

    # Normalize by total number of shots to get the expectation value
    return exp_val / total_counts
"""

def get_expectation_term(term, counts):
    """
    Computes the expectation value of a measurement outcome with respect to a single Pauli operator.

    Args:
        term (str): A string representing a Pauli operator (e.g., 'XXI', 'ZZI', 'III'), 
                    where each character corresponds to the Pauli operator ('X', 'Y', 'Z', or 'I') 
                    acting on a specific qubit.
        counts (dict): A dictionary containing measurement results as keys (bitstrings) and 
                       their corresponding counts as values.

    Returns:
        float: The expectation value of the measurement results with respect to the specified Pauli term.

    Example:
        term = "ZZI"
        counts = {"000": 500, "011": 300, "101": 200}
        result = get_expectation_term(term, counts)
    """
    exp_val = 0
    total_counts = sum(counts.values())  # Total number of measurement shots
    num_qubits = len(term)  # Total number of qubits in the system

    # Loop over all measurement results
    for bitstring, count in counts.items():
        parity = 1.0  # Initialize parity for the current bitstring

        # Iterate over each qubit and its corresponding Pauli operator
        for qubit_index, pauli in enumerate(term):
            if pauli == 'I':  # Skip identity operators, as they do not affect the parity
                continue

            # Map qubit index to bitstring index (little-endian order) and extract bit value
            bit_index = num_qubits - 1 - qubit_index
            bit_value = int(bitstring[bit_index])

            # Map bit value (0 or 1) to eigenvalue (+1 or -1)
            eigenvalue = 1 - 2 * bit_value
            parity *= eigenvalue  # Update parity based on the eigenvalue

        exp_val += parity * count  # Weighted sum of parities based on counts

    # Normalize by the total number of measurement shots
    return exp_val / total_counts

    
####################################################################################
           
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
  