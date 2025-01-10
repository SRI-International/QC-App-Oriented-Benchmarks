'''
Observable Computation Function Library - Qiskit Version
(C) Quantum Economic Development Consortium (QED-C) 2024.

This module includes helper funtions for computing observables from a Hamiltonian.
'''

from itertools import combinations
import numpy as np
import copy
import sys
import time

from qiskit.quantum_info import SparsePauliOp

sys.path[1:1] = ["..//qiskit"]
import hamlib_simulation_kernel as kernel

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

# Extract Qubit-wise Commuting Groups from a Hamiltonian

# Function to group commuting terms into qubit-wise commuting groups
def group_commuting_terms(pauli_terms):
    """
    Group commuting Pauli terms.
    
    Args:
    pauli_terms (list): List of tuples where each tuple contains a Pauli string and a coefficient.
    
    Returns:
    list: List of groups where each group is a list of commuting Pauli terms.
    """
    if verbose:
        print(f"... group_commuting_terms({pauli_terms})")
    
    # Build the commutativity graph
    adjacency_matrix = build_commutativity_graph(pauli_terms)
    
    # Group commuting terms
    groups = []
    ungrouped_indices = set(range(len(pauli_terms)))

    # loop over all the terms, looking for those that can be grouped
    while ungrouped_indices:
        current_group = []
        current_group_indices = []

        # make the first item in the ungrouped list into the new current group
        i = ungrouped_indices.pop()
        current_group.append(pauli_terms[i])
        current_group_indices.append(i)
        #print(f"\n... visit ungrouped_index: {i} {pauli_terms[i]}")

        # we need to check that the candidate term commutes with all the terms in the current group
        for j in ungrouped_indices.copy():
            #print(f"  ... compare to ungrouped_index: {j} {pauli_terms[j]}")

            commuting = True
            for k in current_group_indices:
                #print(f"    ... checking against: {k} {pauli_terms[k]}")
                if not adjacency_matrix[k, j]:
                    commuting = False
                    #if verbose: print(f"    ... conflict, do not add to this group")
                    break

            if commuting:
                #print(f"    ... commutes, add to this grouop")
                current_group.append(pauli_terms[j])
                current_group_indices.append(j)
                ungrouped_indices.remove(j)
        
        groups.append(current_group)
    
    return groups

def build_commutativity_graph(pauli_terms):
    """
    Build a commutativity graph where nodes represent Pauli operators,
    and edges represent commutation relations between them.
    
    Args:
    pauli_terms (list): List of tuples where each tuple contains a Pauli string and a coefficient.
    
    Returns:
    np.ndarray: Adjacency matrix of the commutativity graph.
    """
    num_terms = len(pauli_terms)
    adjacency_matrix = np.zeros((num_terms, num_terms), dtype=bool)
    
    for i, j in combinations(range(num_terms), 2):
        pauli_str1 = pauli_terms[i][0]
        pauli_str2 = pauli_terms[j][0]
        
        if pauli_commutes(pauli_str1, pauli_str2):
            adjacency_matrix[i, j] = True
            adjacency_matrix[j, i] = True
    
    #if verbose: print(adjacency_matrix)
    
    return adjacency_matrix

def pauli_commutes(pauli_str1, pauli_str2):
    """
    Check if two Pauli operators commute using their string representations.
    
    Args:
    pauli_str1 (str): Pauli string of first Pauli operator.
    pauli_str2 (str): Pauli string of second Pauli operator.
    
    Returns:
    bool: True if pauli1 and pauli2 commute, False otherwise.
    """
    for i in range(len(pauli_str1)):
        if pauli_str1[i] != 'I' and pauli_str2[i] != 'I' and pauli_str1[i] != pauli_str2[i]:
            return False
    return True


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
# CALCULATE EXPECTATION VALUE FUNCTIONS

# These functions compute expectation values from either measurement results or 
# from the term contributions returned from the measurement function.

def calculate_expectation_from_measurements(num_qubits, results, pauli_term_groups):
    """
    Calculates the total expectation value (energy) from measurement results and provided pauli_term_groups.

    This function processes measurement results for a set of quantum circuits, each corresponding to
    a group of Pauli terms, to compute the expectation value of a Hamiltonian. Optionally, it can store
    the contribution of each term in a dictionary.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        results (Result): Results object containing measurement counts from circuit execution.
        pauli_term_groups (list): Groups of Pauli terms as tuples of (pauli, coeff).

    Returns:
        tuple: A tuple containing:
            - total_energy (float): The computed total energy of the Hamiltonian.
            - term_contributions (dict): A dictionary with individual Pauli terms as keys
              and their respective contributions to the total energy as values.
    """
    total_exp = 0
    term_contributions = {}
   
    # Loop over each group and its corresponding measurement results
    if len(pauli_term_groups) > 1:
        for group, result in zip(pauli_term_groups, results.get_counts()):
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
        group = pauli_term_groups[0]

        # Process each Pauli term in the current group
        for term, coeff in group:
            exp_val = get_expectation_term(term, counts)
            total_exp += coeff * exp_val
            
            # if dict provided, save the contribution from each term
            term_contributions[term] = exp_val

    return total_exp, term_contributions
    
def calculate_expectation_from_contributions(term_contributions: dict, pauli_terms: list):
    """
    Computes the total expectation value from precomputed term contributions.

    Args:    
        term_contributions (dict): A dictionary mapping Pauli terms to their corresponding 
                                   expectation values. Missing terms are assumed to have a value of zero.                            
        pauli_terms (list of tuples): A list of Pauli terms with coefficients, where each element is 
                                    a tuple of the form (Pauli term, coefficient).

    Returns:
        float: The total expectation value for the Hamiltonian.

    Note:
        If `term_contributions` is None, the function returns 0 and logs a warning for missing terms.
    """
    total_exp = 0
    
    if term_contributions is None:
        return total_exp

    # Process each Pauli term in the current group
    for term, coeff in pauli_terms:
        exp_val = term_contributions.get(term)
        
        if exp_val is None:
            exp_val = 0
            print(f"WARN: term not found in term_contributions: {term}")
            
        total_exp += coeff * exp_val
            
    return total_exp

# ####################################
# EXPECTATION VALUE SUPPORT FUNCTIONS
 
# This function may also be used from custom expectation functions that use diagonalization, etc.
 
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

# Estimate expectation, advanced version, similar to Estimator, but customizable
# The functions below rely on access to a quantum kernel to create the circuits and
# an execution module to execute the circuits, both of which are dependent on the target API.

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
    circuits = kernel.create_circuits_for_pauli_terms(qc, num_qubits, pauli_str_list)
       
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
    total_energy, term_contributions = calculate_expectation_from_measurements(
                                            num_qubits, results, pauli_term_groups)
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
