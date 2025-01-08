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
verbose_circuits = False

verbose_time = True

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


# ===========================================
# EXECUTION FUNCTIONS

from qiskit import QuantumCircuit, transpile, assemble
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


# =========================================================================================
# CIRCUIT CREATION FUNCTIONS

def create_circuits_for_hamiltonian(
        qc: QuantumCircuit,
        num_qubits: int,
        pauli_terms: list,
        use_commuting_groups: bool = True
    ):
    """
    Creates quantum circuits for a Hamiltonian, with optional optimization using commuting groups.
    Note: this version of the function creates only the rotation portion of the circuit.
    
    Args:
        num_qubits (int): The number of qubits in the circuit.
        qc (QuantumCircuit): The circuit to which we will append rotation gates.
        pauli_terms (list of tuples): The Hamiltonian represented as a list of tuples, 
                                    where each tuple contains a Pauli string and a coefficient.
        use_commuting_groups (bool): If True, groups commuting terms to optimize the circuit creation.

    Returns:
        list of tuples: A list where each element is a tuple (QuantumCircuit, group or [(term, coeff)]).
    """
    
    if not use_commuting_groups:
        if verbose: print("\n******** creating circuits from Hamiltonian pauli terms:")
        for term in pauli_terms:
            print(term)
            
        circuits = create_circuits_for_pauli_terms(qc, num_qubits, pauli_terms)
    else:
        if verbose: print("\n******** creating commuting groups for the Hamiltonian and circuits from the groups:")
        pauli_term_groups = group_commuting_terms(pauli_terms)
        
        print("... created pauli_term_groups:")
        for i, group in enumerate(pauli_term_groups):
            print(f"Group {i+1}:")
            for pauli, coeff in group:
                print(f"  {pauli}: {coeff}")
                
        circuits = create_circuits_for_grouped_terms(qc, num_qubits, pauli_term_groups)

    if verbose: print(f"\n... constructed {len(circuits)} circuits for this Hamiltonian.")
    return circuits
    
def create_circuits_for_pauli_terms(qc: QuantumCircuit, num_qubits: int, pauli_terms: list):
    """
    Creates quantum circuits for measuring terms in a raw Hamiltonian.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        qc (QuantumCircuit): The circuit to which we will append rotation gates.
        pauli_terms (list of tuples): The Hamiltonian represented as a list of tuples, 
                              where each tuple contains a Pauli string and a coefficient.

    Returns:
        list of tuples: A list where each element is a tuple (QuantumCircuit, [(term, coeff)]).
    """
    circuits = []

    for term, coeff in pauli_terms:
        if verbose: print(f"  ... create_circuits_for_pauli_term: {term}, {coeff}")
        
        # append the rotations specific to each pauli of this term and the measurements
        qc2 = append_measurement_circuit_for_term(qc, num_qubits, term)
    
        circuits.append((qc2, [(term, coeff)]))

    return circuits

def create_circuits_for_grouped_terms(qc: QuantumCircuit, num_qubits: int, pauli_term_groups: list):
    """
    Creates quantum circuits for groups of commuting terms in a Hamiltonian.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        qc (QuantumCircuit): The circuit to which we will append rotation gates.
        pauli_term_groups (list of list of tuples): A list of groups, where each group is a list of tuples 
                                         (term, coeff) representing commuting Hamiltonian terms.

    Returns:
        list of tuples: A list where each element is a tuple (QuantumCircuit, group).
    """
    circuits = []
    for group in pauli_term_groups:
    
        merged_term = merge_pauli_terms(group, num_qubits)
      
        # append the rotations specific to each pauli of this term and the measurements
        qc2 = append_measurement_circuit_for_term(qc, num_qubits, merged_term)

        circuits.append((qc2, group))

    return circuits

def merge_pauli_terms(group: list, num_qubits: int):
    """
    Merge Pauli terms into a single string to create one circuit per group
    """
    merged_paulis = ['I'] * num_qubits
    for term, coeff in group:
        for i, pauli in enumerate(term):
            if pauli != "I":
                merged_paulis[i] = pauli

    merged_term = "".join(merged_paulis)
    return merged_term

##### KERNEL FUNCTIONS
    
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


# =========================================================================================
# CALCULATE EXPECTATION VALUE (GROUPS)

def calculate_expectation(num_qubits, results, circuits, term_contributions=None):
    """
    Calculates the total energy (expectation value) from measurement results and provided circuits.

    This function operates on a list of tuples, where each tuple contains a fully formed circuit
    (with rotations prior to measurements at the end) and the group of Pauli terms from which it was created.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        results (Result): The results object containing measurement counts from circuit execution.
        circuits (list of tuples): A list where each element is a tuple of the form (QuantumCircuit, group),
                                   where `group` is a list of (Pauli term, coefficient).
        term_contributions (dict): Optional dictionary in which to place the contribution value of each term.

    Returns:
        float: The total energy (expectation value) for the Hamiltonian.

    Example:
        circuits = [(qc1, group1), (qc2, group2)]
        results = execute(circuits, backend)
        energy = calculate_expectation(3, results, circuits)
    """
    total_exp = 0

    # Loop over each circuit and its corresponding measurement results
    if len(circuits) > 1:
        for (qc, group), result in zip(circuits, results.get_counts()):
            counts = result

            # Process each Pauli term in the current group
            for term, coeff in group:
                exp_val = get_expectation_term(term, counts)
                total_exp += coeff * exp_val
                
                # if dict provided, save the contribution from each term
                if term_contributions is not None:
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

    return total_exp

def calculate_expectation_from_contributions(ham_terms, term_contributions):
    """
    Computes the total expectation value from precomputed term contributions.

    This function uses a Hamiltonian represented as a list of Pauli terms with coefficients and combines
    them with the provided expectation values for each term to calculate the total expectation value.

    Args:
        ham_terms (list of tuples): A list where each element is a tuple of the form (Pauli term, coefficient).
                                    Each Pauli term is expected to match the keys in `term_contributions`.
        term_contributions (dict): A dictionary mapping Pauli terms to their corresponding expectation values.
                                   If a term is missing, its contribution is assumed to be zero, and a warning is logged.

    Returns:
        float: The total expectation value for the Hamiltonian.

    Note:
        If `term_contributions` is None, the function returns 0. Logs a warning for each missing term.

    Example:
        ham_terms = [("XX", 0.5), ("ZZ", -0.3)]
        term_contributions = {"XX": 0.8, "ZZ": 0.6}
        energy = calculate_expectation_from_contributions(ham_terms, term_contributions)
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

# =========================================================================================
# ESTIMATE EXPECTATION VALUE

def estimate_expectation(backend, qc, H_terms, num_shots=10000):
    """
    Estimates the expectation value for a Hamiltonian given a parameterized quantum circuit.

    This function computes the expectation value for a list of weighted Pauli strings 
    by executing a quantum circuit on a backend and measuring the results.

    Args:
        backend (Backend): The quantum backend to execute the circuit (e.g., simulator or quantum device).
        qc (QuantumCircuit): The parameterized quantum circuit.
        H_terms (list of tuples): The Hamiltonian represented as a list of (coefficient, Pauli string) tuples.
        num_shots (int, optional): The number of shots (repeated measurements) to perform. Default is 10,000.

    Returns:
        float: The total energy (expectation value) of the Hamiltonian.

    Example:
        backend = Aer.get_backend('qasm_simulator')
        qc = QuantumCircuit(3)
        H_terms = [(0.5, "XXI"), (1.0, "ZZI")]
        energy = estimate_expectation(backend, qc, H_terms, num_shots=10000)
    """
    total_energy = 0

    # Iterate through each term in the Hamiltonian
    for coeff, pauli_string in H_terms:
        exp_val = estimate_expectation_term(backend, qc, pauli_string, num_shots=num_shots)
        total_energy += coeff * exp_val

        if verbose:
            print(f"... exp value for pauli term = ({coeff}, {pauli_string}), exp = {exp_val}")

    return total_energy


# =========================================================================================
# ESTIMATE EXPECTATION VALUE FOR MULTIPLE OBSERVABLES

def estimate_expectation_multiple(backend, qc, H_terms_multiple, num_shots=10000):
    """
    Estimates the expectation values for a primary Hamiltonian and additional observables.

    This function calculates the expectation values for a list of Pauli term collections, 
    where the first collection represents the primary Hamiltonian, and subsequent collections 
    represent additional observables. The same measurement results are used for all observables.

    Args:
        backend (Backend): The quantum backend to execute the circuit (e.g., simulator or quantum device).
        qc (QuantumCircuit): The parameterized quantum circuit.
        H_terms_multiple (list of lists of tuples): A list of Hamiltonian representations, where each 
                                                    representation is a list of (coefficient, Pauli string) tuples.
        num_shots (int, optional): The number of shots (repeated measurements) to perform. Default is 10,000.

    Returns:
        list of float: A list of expectation values, where the first value corresponds to the primary Hamiltonian,
                       and subsequent values correspond to additional observables.

    Example:
        backend = Aer.get_backend('qasm_simulator')
        qc = QuantumCircuit(3)
        H_terms_multiple = [
            [(0.5, "XXI"), (1.0, "ZZI")],  # Primary Hamiltonian
            [(0.2, "XXI")],                # Observable 1
            [(0.3, "ZZI")]                 # Observable 2
        ]
        expectations = estimate_expectation_multiple(backend, qc, H_terms_multiple, num_shots=10000)
    """
    # Storage for observables
    observables_store = []
    H_observables = []

    # Initialize expectation values for each observable
    for _ in range(len(H_terms_multiple)):
        observables_store.append(0)

    # Convert each observable's terms into a dictionary for easier access
    for terms in H_terms_multiple:
        H_observables.append({pauli_term: coeff for coeff, pauli_term in terms})

    # Iterate through terms of the primary Hamiltonian
    for pauli_string, coeff in H_observables[0].items():
        exp_val = estimate_expectation_term(backend, qc, pauli_string, num_shots=num_shots)

        # Accumulate expectation for the primary Hamiltonian
        observables_store[0] += coeff * exp_val

        # Accumulate expectation for additional observables
        for i in range(1, len(H_observables)):
            if pauli_string in H_observables[i]:
                observables_store[i] += H_observables[i][pauli_string] * exp_val

        if verbose:
            print(f"... exp value for pauli term = ({coeff}, {pauli_string}), exp = {exp_val}")

    return observables_store

# =========================================================================================
# EXPECTATION VALUE SUPPORT FUNCTIONS
   
def estimate_expectation_term(backend, qc, pauli_string, num_shots=10000):
    """
    Estimates the expectation value of a given Pauli string for a quantum circuit.

    This function computes the energy contribution of a single Pauli term of a Hamiltonian
    by executing a parameterized quantum circuit, measuring the results, and calculating
    the expectation value from the measurement outcomes.

    Args:
        backend (Backend): The quantum backend to execute the circuit (e.g., a simulator or quantum device).
        qc (QuantumCircuit): The quantum circuit representing the parameterized ansatz.
        pauli_string (str): The Pauli string (e.g., 'XXI', 'ZIZ') defining the term of the Hamiltonian.
        num_shots (int, optional): The number of shots (repeated measurements) to perform. Default is 10,000.

    Returns:
        float: The expectation value of the specified Pauli string.

    Notes:
        - The circuit is cloned to avoid modifying the original quantum circuit.
        - Gates corresponding to the Pauli string are appended to the circuit before measurement.
        - The function relies on `append_hamiltonian_term_to_circuit`, `execute_circuit`, 
          and `get_expectation_term` for appending gates, executing the circuit, and computing 
          the expectation value, respectively.

    Dependencies:
        This function uses the following helper functions:
        - `append_hamiltonian_term_to_circuit`: Appends gates for the Pauli string.
        - `execute_circuit`: Executes the circuit on the given backend and returns measurement counts.
        - `get_expectation_term`: Computes the expectation value from measurement counts.
    """
    # Clone the original circuit to avoid modifications
    qc = qc.copy()

    # Append gates for the Pauli string
    append_hamiltonian_term_to_circuit(qc, None, pauli_string)

    # Add measurement gates
    qc.measure_all()
    
    if verbose:
        print(f"... circuit with Pauli {pauli_string} =\n{qc}")

    # Execute the circuit on the backend and obtain measurement counts
    counts = execute_circuit(qc, backend, num_shots, None)
    if verbose:
        print(f"... counts = {counts}")

    # Compute the expectation value from the counts
    expectation = get_expectation_term(pauli_string, counts)
    
    return expectation

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
# HIGH-LEVEL FUNCTIONS
           
def estimate_expectation_with_estimator(backend, qc, H_terms, num_shots=10000):
    """
    Estimates the expectation value for a quantum circuit and Hamiltonian using the `Estimator` class.

    This function calculates the expectation value of a Hamiltonian represented by an array of weighted Pauli strings
    by converting it into a `SparsePauliOp` and using the `Estimator` class for efficient computation.

    Args:
        backend (Backend): The quantum backend used for the estimation (not currently utilized in this function).
        qc (QuantumCircuit): The parameterized quantum circuit representing the ansatz.
        H_terms (list of tuples): The Hamiltonian represented as a list of (coefficient, Pauli string) tuples.
        num_shots (int, optional): The number of shots (repeated measurements) to perform. Default is 10,000. 
                                   (Not currently used in this function, as `Estimator` does not require it.)

    Returns:
        float: The measured energy (expectation value) of the Hamiltonian.

    Example:
        backend = Aer.get_backend('qasm_simulator')
        qc = QuantumCircuit(3)
        H_terms = [(0.5, "XXI"), (1.0, "ZZI")]
        energy = estimate_expectation_with_estimator(backend, qc, H_terms)

    Notes:
        - The Hamiltonian terms are converted to a `SparsePauliOp` using `convert_to_sparse_pauli_op`.
        - The `Estimator` class is used to calculate the expectation value efficiently.
        - This method does not use shot-based sampling, and `num_shots` is included only for consistency with
          other estimation functions.

    Dependencies:
        This function requires:
        - `convert_to_sparse_pauli_op`: Converts the Hamiltonian terms into a `SparsePauliOp` object.
        - `Estimator`: A class used to calculate expectation values.

    Debugging:
        Uncomment the print statement to trace function entry.
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

# =========================================================================================
# ESTIMATE EXPECTATION VALUE - NEW 

# The code in this section is work-in-progress, building on code in the test notebooks

# Estimate expectation, advanced version, similar to Estimator, but customizable

def estimate_expectation_plus(backend, qc, ham_terms, use_commuting_groups=True, num_shots=10000):
    
    num_qubits = qc.num_qubits

    # Create circuits from the Hamiltonian
    ts0 = time.time()
    circuits = create_circuits_for_hamiltonian(qc, num_qubits, ham_terms, use_commuting_groups)
    ts1 = time.time()
    
    if verbose_circuits:
        for circuit in circuits:
            print(circuit)
            print(circuit[0])  
    
    # Compile and execute the circuits
    ts2 = time.time()
    transpiled_circuits = transpile([circuit for circuit, group in circuits], backend)

    # Execute all of the circuits to obtain array of result objects
    ts3 = time.time()
    results = backend.run(transpiled_circuits).result()
    
    """ debugging when single circuit
    print(f"... results = {results}")
    # Loop over each circuit and its corresponding measurement results
    if len(circuits) > 1:
        for (qc, group), result in zip(circuits, results.get_counts()):
            counts = result
            print(counts, flush=True)
    else:
        counts = results.get_counts()
        print(counts, flush=True)
    """
    
    # Compute the total energy for the Hamiltonian
    ts4 = time.time()
    term_contributions = {}
    total_energy = calculate_expectation(num_qubits, results, circuits,
                                    term_contributions=term_contributions)
    ts5 = time.time()
    
    if verbose_time:
        print(f"... circuit creation time = {round(ts1 - ts0, 3)}")
        print(f"... transpilation time = {round(ts3 - ts2, 3)}")
        print(f"... execution time = {round(ts4 - ts3, 3)}")
        print(f"... expectation time = {round(ts5 - ts4, 3)}") 
        print(f"... total time = {round((ts5 - ts2) + (ts1 - ts0), 3)}")
        print("")
        
    #print(f"Total Energy: {total_energy}")#print("")
    #print(f"Term Contributions: {term_contributions}")
     
    return total_energy, term_contributions

   
####################################################################################
# FROM OBSERVABLES GENERALIZED

import numpy as np
import copy
from math import sin, cos, pi
import time

from qiskit import QuantumCircuit, transpile
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
