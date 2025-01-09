'''
Observable Computation Function Library - Qiskit Version
(C) Quantum Economic Development Consortium (QED-C) 2024.

This module includes helper funtions for computing observables from a Hamiltonian.
'''

from itertools import combinations
import numpy as np
import copy

from qiskit.quantum_info import SparsePauliOp

verbose = False
verbose_circuits = False

verbose_time = False

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

# Function to group commuting terms into qubit-wise commuting groups
def group_commuting_terms(pauli_list):
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

    #if verbose: print(adjacency_matrix)
    
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
                    #if verbose: print(f"    ... conflict, do not add to this group")
                    break

            if commuting:
                #print(f"    ... commutes, add to this grouop")
                current_group.append(pauli_list[j])
                current_group_indices.append(j)
                ungrouped_indices.remove(j)
        
        groups.append(current_group)
    
    return groups

# =========================================================================================
# PAULI TERM OPTIMIZATION FUNCTIONS

def group_pauli_terms_for_execution(
        num_qubits: int,
        pauli_terms: list,
        use_commuting_groups: bool = True
    ):
    """
    Groups Pauli terms for quantum execution, optionally combining commuting terms into groups.

    This function prepares Pauli terms for use in quantum computations. It can either group 
    commuting terms together to optimize execution or treat each term as its own group. This 
    step is typically performed prior to creating quantum circuits for execution.

    Args:
        num_qubits (int): The number of qubits in the quantum system.
        pauli_terms (list): A list of tuples representing the Hamiltonian Pauli terms.
            Each tuple contains a Pauli string (str) and a coefficient (float/complex).
        use_commuting_groups (bool, optional): Whether to group commuting terms together.
            Defaults to True. If False, each term is placed in its own group.

    Returns:
        tuple: A tuple containing:
            - pauli_term_groups (list): A list of groups, where each group is a list of 
              Pauli terms (tuples) that are either commuting or individual terms.
            - pauli_str_list (list): A list of merged Pauli strings (one for each group),
              or individual strings if `use_commuting_groups` is False.

    Notes:
        - When `use_commuting_groups` is True, terms are grouped using `group_commuting_terms()`, 
          and each group's Pauli strings are merged using `merge_pauli_terms()`.
        - When `use_commuting_groups` is False, each term is treated as its own group, ensuring 
          consistency for expectation function submission.
    """
    if not use_commuting_groups:
        if verbose:
            print("\n******** creating circuits from Hamiltonian pauli terms:")
            for term in pauli_terms:
                print(term)
        
        # create an array of groups, with one term in each group
        # (for consisency in submitting to expectation functions)
        pauli_term_groups = [[pauli_term] for pauli_term in pauli_terms]
        
        # get a list of just the pauli strings
        pauli_str_list, _ = zip(*pauli_terms)
  
    else:
        if verbose:
            print("\n******** creating commuting groups for the Hamiltonian and circuits from the groups:")
        
        # create an array of groups, by combining commuting terms
        pauli_term_groups = group_commuting_terms(pauli_terms)
        
        if verbose:
            print("... created pauli_term_groups:")
            for i, group in enumerate(pauli_term_groups):
                print(f"Group {i+1}:")
                for pauli, coeff in group:
                    print(f"  {pauli}: {coeff}")
        
        # for each group, create a merged pauli string from all the terms in the group
        pauli_str_list = []
        for group in pauli_term_groups:
            merged_pauli_str = merge_pauli_terms(group, num_qubits)
            pauli_str_list.append(merged_pauli_str)
            
    return pauli_term_groups, pauli_str_list
    

def merge_pauli_terms(group: list, num_qubits: int):
    """
    Merges a group of Pauli terms into a single Pauli string.

    Combines the terms in a group by taking the non-identity ("I") operators 
    at each qubit position. The result is a single Pauli string that represents
    the merged terms for a group.

    Args:
        group (list): A list of tuples, where each tuple contains a Pauli string 
                      and its coefficient.
        num_qubits (int): The number of qubits in the system.

    Returns:
        str: The merged Pauli string.
    """
    merged_paulis = ['I'] * num_qubits
    for term, coeff in group:
        for i, pauli in enumerate(term):
            if pauli != "I":
                merged_paulis[i] = pauli

    merged_term = "".join(merged_paulis)
    return merged_term


# ####################################################################################
# ESTIMATE EXPECTATION VALUE 

# The code in this section is work-in-progress, building on code in the test notebooks

# Estimate expectation, advanced version, similar to Estimator, but customizable

def estimate_expectation_value(backend, qc, pauli_terms, use_commuting_groups=True, num_shots=10000):
    """
    Estimates the expectation value of a Hamiltonian using a quantum backend.

    This function computes the total energy of a Hamiltonian represented by Pauli terms
    through quantum execution. It groups the Pauli terms (optionally by commuting groups),
    generates circuits, executes them, and calculates the resulting expectation values.

    Args:
        backend: The quantum backend to execute the circuits on.
        qc: The quantum circuit used as a starting point for measurement circuits.
        pauli_terms (list): A list of tuples representing the Hamiltonian Pauli terms,
            where each tuple contains a Pauli string (str) and a coefficient (float/complex).
        use_commuting_groups (bool, optional): Whether to group commuting terms together 
            to optimize execution. Defaults to True.
        num_shots (int, optional): The number of shots to use for execution. Defaults to 10,000.

    Returns:
        tuple: A tuple containing:
            - total_energy (float): The computed total energy of the Hamiltonian.
            - term_contributions (dict): A dictionary with individual Pauli terms as keys
              and their respective contributions to the total energy as values.

    Notes:
        - The function leverages `group_pauli_terms_for_execution` to organize the terms
          and `create_circuits_for_pauli_terms` to generate measurement circuits.
        - Execution times for various stages (circuit creation, transpilation, execution,
          and expectation calculation) are printed if `verbose_time` is enabled.
        - Verbose output includes detailed circuit and group information for debugging.

    """
    num_qubits = qc.num_qubits

    # Create circuits from the Hamiltonian
    ts0 = time.time()  
    
    # group Pauli terms for quantum execution, optionally combining commuting terms into groups.
    pauli_term_groups, pauli_str_list = group_pauli_terms_for_execution(
            num_qubits, pauli_terms, use_commuting_groups)
    
    # generate an array of circuits, one for each pauli_string in list
    circuits = create_circuits_for_pauli_terms(qc, num_qubits, pauli_str_list)
       
    ts1 = time.time()
  
    if verbose:
        print(f"\n... constructed {len(circuits)} circuits for this Hamiltonian.")
    if verbose_circuits:
        for circuit, group in list(zip(circuits, pauli_term_groups)):
            print(group)
            print(circuit)

    # Execute all of the circuits to obtain array of result objects
    ts2 = time.time()
    results = execute_circuits(circuits, backend, num_shots, params=None)
    
    # Compute the total energy for the Hamiltonian
    ts3 = time.time()
    total_energy, term_contributions = calculate_expectation(num_qubits, results, circuits, pauli_term_groups)
    ts4 = time.time()
    
    if verbose_time:
        print(f"... circuit creation time = {round(ts1 - ts0, 3)}")
        print(f"... execution time = {round(ts3 - ts2, 3)}")
        print(f"... expectation time = {round(ts4 - ts3, 3)}") 
        print(f"... total elapsed time = {round((ts4 - ts2) + (ts1 - ts0), 3)}")
        print("")
        
    #print(f"Total Energy: {total_energy}")#print("")
    #print(f"Term Contributions: {term_contributions}")
     
    return total_energy, term_contributions


def calculate_expectation(num_qubits, results, circuits, pauli_term_groups):
    """
    Calculates the total expectation value (energy) from measurement results and provided circuits.

    This function processes measurement results for a set of quantum circuits, each corresponding to
    a group of Pauli terms, to compute the expectation value of a Hamiltonian. Optionally, it can store
    the contribution of each term in a dictionary.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        results (Result): Results object containing measurement counts from circuit execution.
        circuits (list): List of quantum circuits corresponding to the Pauli term groups.
        pauli_term_groups (list): Groups of Pauli terms as tuples of (pauli, coeff).
        term_contributions (dict, optional): Dictionary to store the contribution of each term.

    Returns:
        float: The total expectation value of the Hamiltonian.
    """
    total_exp = 0
    term_contributions = {}
    
    # bundle the circuits with the corresponding sets of terms (one or multiple)
    circuits = list(zip(circuits, pauli_term_groups))
    #for circuit in circuits: print(circuit)
    
    # Loop over each circuit and its corresponding measurement results
    if len(circuits) > 1:
        for (qc, group), result in zip(circuits, results.get_counts()):
            counts = result

            # Process each Pauli term in the current group
            for term, coeff in group:
                exp_val = get_expectation_term(term, counts)
                total_exp += coeff * exp_val
                
                # save the contribution from each term
                term_contributions[term] = exp_val
                    
    # results object has different structure when only one circuit, process specially here
    else:
        counts = results.get_counts()
        group = circuits[0][1]

        # Process each Pauli term in the current group
        for term, coeff in group:
            exp_val = get_expectation_term(term, counts)
            total_exp += coeff * exp_val
            
            # if dict provided, save the contribution from each term
            if term_contributions is not None:
                term_contributions[term] = exp_val

    return total_exp, term_contributions

def calculate_expectation_from_contributions(ham_terms, term_contributions):
    """
    Computes the total expectation value from precomputed term contributions.

    Args:
        ham_terms (list of tuples): A list of Pauli terms with coefficients, where each element is 
                                    a tuple of the form (Pauli term, coefficient).
        term_contributions (dict): A dictionary mapping Pauli terms to their corresponding 
                                   expectation values. Missing terms are assumed to have a value of zero.

    Returns:
        float: The total expectation value for the Hamiltonian.

    Note:
        If `term_contributions` is None, the function returns 0 and logs a warning for missing terms.
    """
    total_exp = 0
    
    if term_contributions is None:
        return total_exp

    # Process each Pauli term in the current group
    for term, coeff in ham_terms:
        exp_val = term_contributions.get(term)
        
        if exp_val is None:
            exp_val = 0
            print(f"WARN: term not found in term_contributions: {term}")
            
        total_exp += coeff * exp_val
            
    return total_exp

# ####################################
# EXPECTATION VALUE SUPPORT FUNCTIONS
   
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
# HIGH-LEVEL ESTIMATION FUNCTIONS
           
def estimate_expectation_with_estimator(backend, qc, H_terms, num_shots=10000):
    """
    Estimates the expectation value of a quantum circuit and Hamiltonian using the `Estimator` class.

    Args:
        backend (Backend): The quantum backend used for the estimation (currently unused in this function).
        qc (QuantumCircuit): The parameterized quantum circuit representing the ansatz.
        H_terms (list of tuples): The Hamiltonian as a list of (coefficient, Pauli string) tuples.
        num_shots (int, optional): The number of shots for repeated measurements (not used with `Estimator`). Default is 10,000.

    Returns:
        float: The calculated expectation value (measured energy) of the Hamiltonian.

    Notes:
        - Converts `H_terms` to a `SparsePauliOp` using `convert_to_sparse_pauli_op`.
        - Uses the `Estimator` class for efficient, non-shot-based computation.
    """
    # Convert Hamiltonian terms to SparsePauliOp
    H_op = convert_to_sparse_pauli_op(H_terms)

    # Create an Estimator instance
    estimator = Estimator()

    # Use the estimator to compute the expectation value
    job = estimator.run(qc, H_op)
    result = job.result()

    # Extract the measured energy (expectation value)
    measured_energy = result.values[0]

    return measured_energy


   
####################################################################################
# FROM OBSERVABLES GENERALIZED

import numpy as np
import copy
from math import sin, cos, pi
import time

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.primitives import Estimator

from qiskit.quantum_info import Operator, Pauli
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PauliEvolutionGate

import scipy as sc

# Set numpy print options to format floating point numbers
np.set_printoptions(precision=3, suppress=True)
   

"""
Define Pauli Evolution Circuit¶
This function is used to create a circuit, given an array of Pauli terms, that performs Trotterized state evolution for time t.
"""

def create_pauli_evolution_circuit(pauli_terms, time=1.0):
    """
    Create a QuantumCircuit with PauliEvolution gate from Pauli terms.
    
    Args:
    pauli_terms (list): List of tuples, each containing (coefficient, Pauli string)
    time (float): Evolution time (default is 1.0)
    
    Returns:
    QuantumCircuit: Circuit with PauliEvolution gate
    """
    
    # Determine the number of qubits
    num_qubits = len(pauli_terms[0][1])  # Length of any Pauli string
    
    # Convert to SparsePauliOp
    sparse_pauli_op = convert_to_sparse_pauli_op(pauli_terms)
    
    # Create the PauliEvolutionGate
    evo_gate = PauliEvolutionGate(sparse_pauli_op, time=time)
    
    # Create a quantum circuit and apply the evolution gate
    qc = QuantumCircuit(num_qubits)
    qc.append(evo_gate, range(num_qubits))
    
    return qc
    
"""
Create Quantum Test Evolution Circuit
Here, we create a circuit that will be measured and that will have its energy computed against a specific Hamiltonian. We start with an initial state and apply quantum Hamiltonian evolution to it. The resulting state will be used for testing in subsequent cells.

We create it using a generated quantum circuit to perform the evolution.
"""

def create_quantum_test_circuit(initial_state, H_terms, step, step_size):

    initial_state = normalize(np.array(initial_state))
    
    n_qubits = len(H_terms[0][1])
    qc = QuantumCircuit(n_qubits)

    # Initialize the circuit with the given state vector
    qc.initialize(initial_state, qc.qubits)
    
    qc_ev = create_pauli_evolution_circuit(H_terms, time = step_size)
    
    if verbose: print(f"... evolution circuit = \n{qc_ev}")

    # Need to decompose here, so we do not have references to PauliEvolution gates, which cannot be copied
    qc_ev = qc_ev.decompose().decompose()

    # use compose, instead of append, so that the copy used in expectation computation can function correctly
    for k in range(step):
        qc.compose(qc_ev, inplace=True)
    
    if verbose: print(f"... after compose, saved circuit = \n{qc}")
    
    return qc

def normalize(array):
    # Calculate the sum of squares of the elements
    sum_of_squares = np.sum(np.square(array))
    # Calculate the normalization factor
    normalization_factor = np.sqrt(sum_of_squares)
    # Normalize the array
    normalized_array = array / normalization_factor
    return normalized_array


######################## KERNEL FUNCTIONS

# These functions all belong down in the kernel file

def create_circuits_for_pauli_terms(qc: QuantumCircuit, num_qubits: int, pauli_str_list: list):
    """
    Creates quantum circuits for measuring terms in a raw Hamiltonian.
    If a circuit is passed in, a copy is made, otherwise and empty circuit is created for each pauli string.
    Then, the rotations and the measurements are appended to the circuit and all the circuits are returned in a list.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        qc (QuantumCircuit): The circuit to which we will append rotation gates.
        pauli_str_list (list of tuples): The Hamiltonian represented as a list of tuples, 
                              where each tuple contains a Pauli string and a coefficient.

    Returns:
        list of tuples: A list where each element is a tuple (QuantumCircuit, [(term, coeff)]).
    """
    circuits = []

    for pauli_str in pauli_str_list:
        if verbose: print(f"  ... create_circuits_for_pauli_term: {pauli_str}")
        
        # append the rotations specific to each pauli and the measurements
        qc2 = append_measurement_circuit_for_term(qc, num_qubits, pauli_str)
    
        circuits.append(qc2)
        
    return circuits
   
def append_measurement_circuit_for_term(qc: QuantumCircuit, num_qubits: int, term: str):
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
    # if no circuit passed in, create one
    if qc is None:
        qc = QuantumCircuit(num_qubits)
         
    # Make a clone of the original circuit since we append gates
    qc2 = qc.copy()

    append_hamiltonian_term_to_circuit(qc2, None, term)

    qc2.measure_all()

    return qc2
          
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


# ===========================================
# EXECUTION FUNCTIONS

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.primitives import Estimator
import numpy as np

# Initialize the backend and the simulator
backend = Aer.get_backend('qasm_simulator')
#backend = Aer.get_backend('statevector_simulator')
noise_model = None

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

# Execute a list of quantum circuits with the specified parameters on the given backend system
def execute_circuits(circuits, backend, num_shots: int, params=None):

    results = backend.run(circuits, shots=num_shots).result()
    
    return results
