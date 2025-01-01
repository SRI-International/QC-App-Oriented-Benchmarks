'''
Hamiltonian Simulation Benchmark Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

'''
There are multiple Hamiltonians and three methods defined for this kernel.
The Hamiltonian name is specified in the "hamiltonian" argument.
The "method" argument indicates the type of fidelity comparison that will be done. 
In this case, method 3 is used to create a mirror circuit for scalability.
'''

from typing import Union, List, Tuple, Dict

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate


verbose = False

# global vars that specify the current hdf5 filename, dataset template, and parameters set by the user
dataset_name_template = ""
filename = ""

global_U = None
global_enc = None
global_ratio = None
global_rinst = None
global_h = None
global_pbc_val = None
        
# Saved circuits and subcircuits for display
QC_ = None
QCI_ = None
HAM_ = None
EVO_ = None
INV_ = None

import hamlib_utils

from hamlib_utils import (
    process_hamiltonian_file,
    create_full_filenames,
    construct_dataset_name,
    process_hamlib_data
)

def initialize():
    global global_U, global_enc, global_ratio, global_rinst, global_h, global_pbc_val

    # Initialize default parameters in HamLib kernel module
    global_U = None
    global_enc = None
    global_ratio = None
    global_rinst = None
    global_h = None
    global_pbc_val = None
    
    hamlib_utils.active_hamiltonian_datasets = None


def set_default_parameter_values(filename):
    """
    Set defaults for the parameters that are relevant to the specific Hamiltonian
    """
    
    global global_U, global_enc, global_ratio, global_rinst, global_h, global_pbc_val

    if filename == 'tfim.hdf5' or filename == 'heis.hdf5':
        if global_h == None:
            global_h = 0.1
        if global_pbc_val == None:
            global_pbc_val = 'nonpbc'
        global_U = None
        global_enc = None
        global_ratio = None
        global_rinst = None
    elif filename == 'random_max3sat-hams.hdf5':
        if global_ratio == None:
            global_ratio = 2
        if global_rinst == None:
            global_rinst = '00'
        global_U = None
        global_enc = None
        global_h = None
        global_pbc_val = None
    elif filename == 'FH_D-1.hdf5':
        if global_U == None:
            global_U = 0
        if global_enc == None:
            global_enc = 'bk'
        if global_pbc_val == None:
            global_pbc_val = 'nonpbc'
        global_ratio = None
        global_rinst = None
        global_h = None
    elif filename == 'BH_D-1_d-4.hdf5':
        if global_U == None:
            global_U = 2
        if global_enc == None:
            global_enc = 'gray'
        if global_pbc_val == None:
            global_pbc_val = 'nonpbc'
        global_ratio = None
        global_rinst = None
        global_h = None
    else:
        print("No such hamiltonian is available.")

def get_params_from_globals(hamiltonian_name):
    """
    This function is provided to enable backwards compatilibility of an earlier approach
    to specifying the parameters to be used for selecting a Hamiltonian from HamLib.
    
    NOTE: This function is used internally and will be removed in the near future.
    Once removed, the setting of parameters via the global variables will no longer work.
    
    Returns:
        dict: A dictiionary of Hamiltonian parameters and their values.
    """
    params = {}
    
    if hamiltonian_name == "TFIM" or hamiltonian_name == "Heisenberg":
        if global_h != None:
            params["h"] = global_h
        if global_pbc_val != None:
            params["1D-grid"] = global_pbc_val

    elif hamiltonian_name == "Max3Sat":
        if global_ratio != None:
            params["ratio"] = global_ratio
        if global_rinst != None:
            params["rinst"] = global_rinst

    elif hamiltonian_name == "Fermi-Hubbard-1D":
        if global_U != None:
            params["U"] = global_U
        if global_enc != None:
            params["enc"] = global_enc
        if global_pbc_val != None:
            params["1D-grid"] = global_pbc_val

    elif hamiltonian_name == "Bose-Hubbard-1D":
        if global_U != None:
            params["U"] = global_U
        if global_enc != None:
            params["enc"] = global_enc
        if global_pbc_val != None:
            params["1D-grid"] = global_pbc_val
    
    if verbose:
        print(f"... get_params_from_globals() ==> {params}")
        
    return params
    

#####################################################################################
# PUBLIC API FUNCTIONS

def get_hamlib_sparsepaulilist(
    hamiltonian_name: str, 
    n_spins: int,
):
    """
    Return a quantum Hamiltonian as a sparse Pauli list given the Hamiltonian name,
    the number of qubits, and an associated set of parameter values.
    From the number of qubits and parameters, the specific dataset is selected and processed.

    Steps:
        1. Determine the dataset that matches the given arguments.
        2. Extract Hamiltonian data from an HDF5 file.
        3. Process the data to obtain a SparsePauliList.

    Returns:
        tuple: A tuple containing the Hamiltonian as a SparsePauliOp and the number of qubits required.
    """
    ##print(f"****************** get_hamlib_sparsepaulilist({hamiltonian_name}, {n_spins})")
    
    get_hamiltonian_info(hamiltonian_name=hamiltonian_name)
    
    parsed_pauli_list, num_qubits = get_hamlib_sparsepaulilist_current(n_spins)
	
    return parsed_pauli_list, num_qubits
	
def get_hamlib_sparsepaulilist_current(
    n_spins: int,
):
    """
    Return the quantum operator associated with the current HamLib hdf5 filename and dataset name.

    Steps:
        1. Extract Hamiltonian data from an HDF5 file.
        2. Process the data to obtain a SparsePauliOp and determine the number of qubits.

    Returns:
        tuple: A tuple containing the Hamiltonian as a SparsePauliOp and the number of qubits required.
    """
    global filename
    
    ##print(f"****************** get_hamlib_sparsepaulilist_current({n_spins})")

    # Replace placeholders with actual n_qubits value: n_spins (and other params)
    dataset_name = get_current_dataset_name(n_spins, True)

    if verbose:
        print(f"Trying dataset: {dataset_name}")  # Debug print
    
    data = process_hamiltonian_file(filename, dataset_name)
    
    # print(f"Using dataset: {dataset_name}")
    # print("Raw Hamiltonian Data: ", data)
     
    parsed_pauli_list = None
    num_qubits = 0
    
    if data is not None:
        
        # get the Hamiltonian operator as SparsePauliList and its size from the data       
        parsed_pauli_list, num_qubits = process_hamlib_data(data)
            
    return parsed_pauli_list, num_qubits
  

def get_valid_qubits(min_qubits, max_qubits, skip_qubits):
    """
    Get an array of valid qubits within the specified range, removing duplicates.

    Returns:
        list: A list of valid qubits.
    """
    global dataset_name_template, filename

    # Create an array with the given min, max, and skip values
    qubit_candidates = list(range(min_qubits, max_qubits + 1, skip_qubits))
    valid_qubits_set = set()  # Use a set to avoid duplicates

    ### print(f"************ dataset_name_template (0) = {dataset_name_template}")
    for qubits in qubit_candidates:
        initial_n_spins = qubits // 2 if "{n_qubits/2}" in dataset_name_template else qubits
        n_spins = initial_n_spins

        # print(f"Starting check for qubits = {qubits}, initial n_spins = {n_spins}")

        found_valid_dataset = False

        while n_spins <= max_qubits:
            dataset_name = get_current_dataset_name(n_spins, False)
            ### print(f"************ {n_spins}: dataset_name_template = {dataset_name_template}")

            if verbose:
                print(f"Checking dataset: {dataset_name}")
                
            data = process_hamiltonian_file(filename, dataset_name)
            
            if data is not None:
                # print(f"Valid dataset found for n_spins = {n_spins}")
                if "{n_qubits/2}" in dataset_name_template:
                    valid_qubits_set.add(n_spins * 2)  # Add the original qubits value
                else:
                    valid_qubits_set.add(qubits)
                found_valid_dataset = True
                break
            else:
                # print(f"Dataset not available for n_spins = {n_spins}. Trying next value...")
                n_spins += 1
                if n_spins >= (qubits + skip_qubits) // 2 if "{n_qubits/2}" in dataset_name_template else (qubits + skip_qubits):
                    print(f"No valid dataset found for qubits = {qubits}")
                    break

        if found_valid_dataset:
            continue  # Move to the next candidate in the original skip sequence

    valid_qubits = list(valid_qubits_set)  # Convert set to list to remove duplicates
    valid_qubits.sort()  # Sorting the qubits for consistent order
    
    if verbose:
        print(f"Final valid qubits: {valid_qubits}")
        
    return valid_qubits  


#####################################################################################
# INTERNAL SUPPORTING FUNCTIONS

# get key infomation about the selected Hamiltonian
# DEVNOTE: Error handling here can be improved by simply returning False or raising exception
def get_hamiltonian_info(hamiltonian_name=None):
    global filename, dataset_name_template
    try:
        filename = create_full_filenames(hamiltonian_name)
        dataset_name_template = construct_dataset_name(filename)
    except ValueError:
        print(f"ERROR: cannot load HamLib data for Hamiltonian: {hamiltonian_name}")
        return
    
    if dataset_name_template == "File key not found in data":
        print(f"ERROR: cannot load HamLib data for Hamiltonian: {hamiltonian_name}")
        return
    
    # Set default parameter values for the hamiltonians
    set_default_parameter_values(filename)
    
# Get the actual dataset name by applying parameters to the dataset_name_template
def get_current_dataset_name(n_spins, div_by_2):
    global dataset_name_template
    
    dataset_name_template = dataset_name_template.replace("{ratio}", str(global_ratio)).replace("{rinst}", str(global_rinst))
    dataset_name_template = dataset_name_template.replace("{h}", str(global_h)).replace("{pbc_val}", str(global_pbc_val))
    dataset_name_template = dataset_name_template.replace("{U}", str(global_U)).replace("{enc}", str(global_enc))
    
    # DEVNOTE: problem here ... other code depends on the dataset_name_template being modifed,
    # but not the n_qubits variable ... this needs to be looked at.
    if div_by_2:
        dataset_name = dataset_name_template.replace("{n_qubits}", str(n_spins)).replace("{n_qubits/2}", str(n_spins // 2))
    else:
        dataset_name = dataset_name_template.replace("{n_qubits}", str(n_spins)).replace("{n_qubits/2}", str(n_spins))
    #print(f"*****================ dataset_name_template = {dataset_name_template}")
    #print(f"*****============= dataset_name = {dataset_name}")
    
    return dataset_name
    
 
#####################################################################################
# KERNEL UTILITY FUNCTIONS

def get_hamlib_sparsepauliop(
    hamiltonian_name: str,
    n_spins: int,
):
    """
    Return the quantum operator associated with the given Hamiltonian and parameters.

    Steps:
        1. Determine the dataset that matches the given arguments.
        2. Extract Hamiltonian data from an HDF5 file.
        3. Process the data to obtain a SparsePauliOp and determine the number of qubits.

    Returns:
        tuple: A tuple containing the Hamiltonian as a SparsePauliOp and the number of qubits required.
    """

    if verbose:
        print(f"... get_hamlib_sparsepauliop({hamiltonian_name}, {n_spins})")
    
    # get the list of Pauli terms for the given Hamiltonian
    parsed_pauli_list, num_qubits = get_hamlib_sparsepaulilist(hamiltonian_name, n_spins)
	
    # convert the SparsePauliList to a SparsePauliOp object
    ham_op = ensure_sparse_pauli_op(parsed_pauli_list, num_qubits)
	
    return ham_op, num_qubits
    
# DEVNOTE: this function should not be needed or called externally.  
# However, it is being used below by the create_circuit function. 
# The create_circuit function should just accept the ham terms as an arg
def get_hamlib_operator(
    n_spins: int,
):
    """
    Return the quantum operator associated with the current HamLib hdf5 filename and dataset name.

    Steps:
        1. Extract Hamiltonian data from an HDF5 file.
        2. Process the data to obtain a SparsePauliOp and determine the number of qubits.

    Returns:
        tuple: A tuple containing the Hamiltonian as a SparsePauliOp and the number of qubits required.
    """

    #print(f"****************** get_hamlib_operator({n_spins})")
    
    # get the list of Pauli terms for the currently specified Hamiltonian
    parsed_pauli_list, num_qubits = get_hamlib_sparsepaulilist_current(n_spins)
	
    # convert the SparsePauliList to a SparsePauliOp object
    ham_op = ensure_sparse_pauli_op(parsed_pauli_list, num_qubits)
	
    return ham_op, num_qubits
 
# DEVNOTE: this should not be called from the outside
def process_data(data):
    """
    Process the given data to construct a Hamiltonian in the form of a SparsePauliOp and determine the number of qubits.

    Args:
        data (str or bytes): The Hamiltonian data to be processed. Can be a string or bytes.

    Returns:
        tuple: A tuple containing the Hamiltonian as a SparsePauliOp and the number of qubits.
        
    NOTE: this function os provided for backwards compatility, as other benchmarks are using it.
    """
    
    parsed_pauli_list, num_qubits = process_hamlib_data(data)
    
    hamiltonian = ensure_sparse_pauli_op(parsed_pauli_list, num_qubits)
    return hamiltonian, num_qubits 


#####################################################################################
# CONVERSION FUNCTIONS   

def convert_simple_to_sparse_pauli_op(simple_terms: List[Tuple[str, complex]]) -> SparsePauliOp:
    """
    Converts a list of simple Pauli terms to a SparsePauliOp.
    Args:
        simple_terms: List of tuples where each tuple contains a Pauli string and a coefficient.
    Returns:
        A SparsePauliOp object.
    """
    return SparsePauliOp.from_list(simple_terms)

# This version does NOT account for num_qubits
def convert_sparse_to_sparse_pauli_op(sparse_terms: List[Tuple[Dict[int, str], complex]]) -> SparsePauliOp:
    """
    Converts a list of sparse Pauli terms to a SparsePauliOp.
    Args:
        sparse_terms: List of tuples where each tuple contains a dict of qubit indices and Pauli operators,
                      and a coefficient.
    Returns:
        A SparsePauliOp object.
    """
    # Convert sparse terms to simple terms for SparsePauliOp conversion
    simple_terms = []
    for term, coeff in sparse_terms:
        max_qubit = max(term.keys(), default=-1)
        pauli_string = ['I'] * (max_qubit + 1)  # Create an identity string of appropriate length
        for qubit, pauli in term.items():
            pauli_string[qubit] = pauli
        simple_terms.append(("".join(pauli_string), coeff))
    
    return SparsePauliOp.from_list(simple_terms)

# This version DOES account for num_qubits
def convert_sparse_pauli_terms_to_sparse_pauliop(sparse_pauli_terms, num_qubits):
    """
    Construct a SparsePauliOp from a list of sparse Pauli terms and the number of qubits.

    Args:
        sparse_pauli_terms (list): A list of tuples, where each tuple contains a dictionary representing the Pauli operators and 
                      their corresponding qubit indices, and a complex coefficient.
        num_qubits (int): The total number of qubits.

    Returns:
        SparsePauliOp: The Hamiltonian represented as a SparsePauliOp.
    """
    pauli_list = []
    
    for pauli_dict, coefficient in sparse_pauli_terms:
        label = ['I'] * num_qubits  # Start with identity on all qubits
        for qubit, pauli_op in pauli_dict.items():
            label[qubit] = pauli_op
        label = ''.join(label)
        pauli_list.append((label, coefficient))
    
    hamiltonian = SparsePauliOp.from_list(pauli_list, num_qubits=num_qubits)
    return hamiltonian
    
def ensure_sparse_pauli_op(
    input_data: Union[
        List[Tuple[str, complex]],
        List[Tuple[Dict[int, str], complex]],
        SparsePauliOp
    ],
    num_qubits: int = 0
) -> SparsePauliOp:
    """
    Processes the input data, which can be one of:
    - An array of simple tuples (List[Tuple[str, complex]]).
    - An array of sparse tuples (List[Tuple[Dict[int, str], complex]]).
    - A SparsePauliOp object.
    
    If the input is an array of simple tuples, it is converted to a SparsePauliOp.
    If the input is an array of sparse tuples, it is converted to a SparsePauliOp via an intermediate step.
    If it is already a SparsePauliOp, it is returned as-is.
    
    Args:
        input_data: Input data to process.
    
    Returns:
        A SparsePauliOp object.
    """
    if isinstance(input_data, SparsePauliOp):
        # If already SparsePauliOp, return it directly
        return input_data
    elif isinstance(input_data, list):
        # Check if the first element indicates a simple or sparse format
        if all(isinstance(term[0], str) for term in input_data):
            return convert_simple_to_sparse_pauli_op(input_data)
        elif all(isinstance(term[0], dict) for term in input_data):
            #return convert_sparse_to_sparse_pauli_op(input_data)
            return convert_sparse_pauli_terms_to_sparse_pauliop(input_data, num_qubits)
        else:
            raise ValueError("Inconsistent format in the input list.")
    else:
        raise TypeError("Input must be a list of tuples or a SparsePauliOp.")

  
#####################################################################################
# KERNEL FUNCTIONS

def create_trotter_steps(num_trotter_steps, evo, operator, circuit):
    """
    Appends Trotter steps to a quantum circuit based on the given evolution operator.

    This function iteratively applies an evolution operator to the quantum circuit
    over a specified number of Trotter steps. A barrier is added at the end to 
    prevent gate reordering across this sequence by optimization algorithms.

    Args:
        num_trotter_steps (int): The number of Trotter steps to append to the circuit.
        evo (QuantumGate): The quantum gate representing the evolution operator.
        operator (QuantumOperator): The operator specifying the qubits the evolution 
                                    operator acts upon.
        circuit (QuantumCircuit): The quantum circuit to which the Trotter steps are 
                                  appended.

    Returns:
        QuantumCircuit: The quantum circuit with the added Trotter steps and a barrier.
    """
    for _ in range (num_trotter_steps):
        circuit.append(evo, range(operator.num_qubits))
    circuit.barrier()
    return circuit
    
    
def create_circuit_from_op(
    #ham_op: SparsePauliOp = None,
    ham_op: Union[
        List[Tuple[str, complex]],
        List[Tuple[Dict[int, str], complex]],
        SparsePauliOp
    ] = None,
    num_qubits: int = 0,
    time: float = 1,
    num_trotter_steps: int = 5,
    method: int = 1,
    use_inverse_flag: bool = False,
    init_state: str = None,
    random_pauli_flag: bool = False,
    random_init_flag: bool = False,
    append_measurements: bool = True
):
    """
    Create a quantum circuit based on the Hamiltonian data from an HDF5 file.

    Steps:
        1. Extract Hamiltonian data from an HDF5 file.
        2. Process the data to obtain a SparsePauliOp and determine the number of qubits.
        3. Build a quantum circuit with an initial state and an evolution gate based on the Hamiltonian.
        4. Measure all qubits and print the circuit details.

    Returns:
        tuple: A tuple containing the constructed QuantumCircuit and the Hamiltonian as a SparsePauliOp.
    """
    global QCI_, INV_

    if ham_op is None or num_qubits == 0:
        # print(f"Dataset not available for num_qubits = {num_qubits}.")
        return None, None, None
    
    # print("Number of qubits:", num_qubits)
    if verbose:
        print(f"... Evolution operator = {ham_op}")
    
    # convert from any form to SparsePauliOp
    ham_op = ensure_sparse_pauli_op(ham_op, num_qubits)

    # Build the evolution gate
    # label = "e\u2071\u1D34\u1D57"    # superscripted, but doesn't look good
    evo_label = "e^-iHt"
    evo = PauliEvolutionGate(ham_op, time=time/num_trotter_steps, label=evo_label)

    # Plug it into a circuit
    circuit = QuantumCircuit(ham_op.num_qubits)
    circuit_without_initial_state = QuantumCircuit(ham_op.num_qubits)
    
    # first create and append the initial_state
    # init_state = "checkerboard"
    i_state = initial_state(num_qubits, init_state)
    circuit.append(i_state, range(ham_op.num_qubits))
    circuit.barrier()
    
    if num_qubits <= 6:
        QCI_ = i_state
    
    # Append K trotter steps
    circuit = create_trotter_steps(num_trotter_steps,
            evo if not use_inverse_flag else evo.inverse(),
            ham_op,
            circuit)
            
    circuit_without_initial_state = create_trotter_steps(num_trotter_steps,
            evo if not use_inverse_flag else evo.inverse(),
            ham_op,
            circuit_without_initial_state)

    # Append K Trotter steps of inverse, if method 3
    inv = None
    bitstring = None

    if method == 3: 

        # if not adding random Paulis, just create simple inverse Trotter steps
        if not random_pauli_flag:
            inv = evo.inverse()
            inv.name = "e^iHt"
            circuit = create_trotter_steps(num_trotter_steps, inv, ham_op, circuit)
            if num_qubits <= 6:
                INV_ = inv
 
        # if adding Paulis, do that here, with code from pyGSTi
        else:
            from pygsti_mirror import convert_to_mirror_circuit
           
            # if random init flag state is set, then discard the inital state input and use a completely (harr) random one
            if random_init_flag:
                circuit, bitstring = convert_to_mirror_circuit(circuit_without_initial_state, random_pauli = True, init_state=None)
            else: 
                init_state = initial_state(num_qubits, init_state)
                circuit, bitstring = convert_to_mirror_circuit(circuit_without_initial_state, random_pauli = True, init_state=init_state)

    # convert_to_mirror_circuit adds its own measurement gates    
    if not (random_pauli_flag and method == 3):
        if append_measurements:
            circuit.measure_all()

    return circuit, bitstring, evo if not use_inverse_flag else evo.inverse()

def create_circuit(
    n_spins: int,
    time: float = 1,
    num_trotter_steps: int = 5,
    method: int = 1,
    use_inverse_flag: bool = False,
    init_state: str = None,
    random_pauli_flag: bool = False,
    random_init_flag: bool = False,
    append_measurements: bool = True
):
    """
    Create a quantum circuit based on the Hamiltonian data from an HDF5 file.

    Steps:
        1. Extract Hamiltonian data from an HDF5 file.
        2. Process the data to obtain a SparsePauliOp and determine the number of qubits.
        3. Build a quantum circuit with an initial state and an evolution gate based on the Hamiltonian.
        4. Measure all qubits and print the circuit details.

    Returns:
        tuple: A tuple containing the constructed QuantumCircuit and the Hamiltonian as a SparsePauliOp.
    """
    global QCI_, INV_

    ham_op, num_qubits = get_hamlib_operator(n_spins)
    if ham_op is not None:
    
        # print("Number of qubits:", num_qubits)
        if verbose:
            print(f"... Evolution operator = {ham_op}")

        # Build the evolution gate
        # label = "e\u2071\u1D34\u1D57"    # superscripted, but doesn't look good
        evo_label = "e^-iHt"
        evo = PauliEvolutionGate(ham_op, time=time/num_trotter_steps, label=evo_label)

        # Plug it into a circuit
        circuit = QuantumCircuit(ham_op.num_qubits)
        circuit_without_initial_state = QuantumCircuit(ham_op.num_qubits)
        
        # first create and append the initial_state
        # init_state = "checkerboard"
        i_state = initial_state(num_qubits, init_state)
        circuit.append(i_state, range(ham_op.num_qubits))
        circuit.barrier()
        
        if n_spins <= 6:
            QCI_ = i_state
        
        # Append K trotter steps
        circuit = create_trotter_steps(num_trotter_steps,
                evo if not use_inverse_flag else evo.inverse(),
                ham_op,
                circuit)
                
        circuit_without_initial_state = create_trotter_steps(num_trotter_steps,
                evo if not use_inverse_flag else evo.inverse(),
                ham_op,
                circuit_without_initial_state)

        # Append K Trotter steps of inverse, if method 3
        inv = None
        bitstring = None

        if method == 3: 
    
            # if not adding random Paulis, just create simple inverse Trotter steps
            if not random_pauli_flag:
                inv = evo.inverse()
                inv.name = "e^iHt"
                circuit = create_trotter_steps(num_trotter_steps, inv, ham_op, circuit)
                if n_spins <= 6:
                    INV_ = inv
     
            # if adding Paulis, do that here, with code from pyGSTi
            else:
                from pygsti_mirror import convert_to_mirror_circuit
               
                # if random init flag state is set, then discard the inital state input and use a completely (harr) random one
                if random_init_flag:
                    circuit, bitstring = convert_to_mirror_circuit(circuit_without_initial_state, random_pauli = True, init_state=None)
                else: 
                    init_state = initial_state(n_spins, init_state)
                    circuit, bitstring = convert_to_mirror_circuit(circuit_without_initial_state, random_pauli = True, init_state=init_state)

        # convert_to_mirror_circuit adds its own measurement gates    
        if not (random_pauli_flag and method == 3):
            if append_measurements:
                circuit.measure_all()
    
        return circuit, bitstring, ham_op, evo if not use_inverse_flag else evo.inverse()

    else:
        # print(f"Dataset not available for n_spins = {n_spins}.")
        return None, None, None, None



############### Initial Circuit Definition

def initial_state(n_spins: int, init_state: str = "checker") -> QuantumCircuit:
    """
    Initialize the quantum state.
    
    Args:
        n_spins (int): Number of spins (qubits).
        init_state (str): The chosen initial state. By default applies the checkerboard state, but can also be set to "ghz", the GHZ state.

    Returns:
        QuantumCircuit: The initialized quantum circuit.
    """
    qc = QuantumCircuit(n_spins)

    init_state = init_state.strip().lower()
    
    if init_state == "checkerboard" or init_state == "neele":
        # Checkerboard state, or "Neele" state
        qc.name = "Neele"
        for k in range(0, n_spins, 2):
            qc.x([k])
    elif init_state.strip().lower() == "ghz":
        # GHZ state: 1/sqrt(2) (|00...> + |11...>)
        qc.name = "GHZ"
        qc.h(0)
        for k in range(1, n_spins):
            qc.cx(k-1, k)

    return qc

############### Hamiltonian Circuit Definition

def HamiltonianSimulation(
            num_qubits: int = 0,
            #ham_op: SparsePauliOp = None, 
            ham_op: Union[
                List[Tuple[str, complex]],
                List[Tuple[Dict[int, str], complex]],
                SparsePauliOp
            ] = None,
            K: int = 5, t: float = 1.0,
            init_state = None,
            method: int = 1,
            use_inverse_flag: bool = False,
            random_pauli_flag = False,
            random_init_flag = False,
            append_measurements = True,
        ) -> QuantumCircuit:
    """
    Construct a Qiskit circuit for Hamiltonian simulation.

    Args:
        num_qubits (int): Number of qubits.
        ham_op (Union): Term, Sparse Term, or SparsePauliOp representation of the Hamiltonian. 
        K (int): The Trotterization order.
        t (float): Duration of simulation.
        method (int): Type of comparison for fidelity
        random_pauli_flag (bool): Insert random Pauli gates if method 3

    Returns:
        QuantumCircuit: The constructed Qiskit circuit.
    """
    circuit_id = f"{K}-{t}"

    # Allocate qubits
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(qr, cr, name=f"hamsim-{num_qubits}-{circuit_id}")
    
    # create the quantum circuit for this Hamiltonian, along with the correct pauli bstring,
    # the operator and trotter evolution circuit
    qc, bitstring, evo = create_circuit_from_op(
        ham_op=ham_op,
        num_qubits=num_qubits,
        time=t,
        method=method,
        use_inverse_flag=use_inverse_flag,
        init_state=init_state,
        num_trotter_steps=K,
        random_pauli_flag=random_pauli_flag,
        random_init_flag=random_init_flag,
        append_measurements=append_measurements
        )

    # Save smaller circuit example for display
    global QC_, HAM_, EVO_, INV_
    if num_qubits <= 6:
        QC_ = qc
        HAM_ = ham_op
        EVO_ = evo
        #INV_ = inv
            
    # Collapse the sub-circuits used in this benchmark (for Qiskit)
    qc2 = qc.decompose().decompose()

    # return both the circuit created, the bitstring, and the Hamiltonian operator
    # if random_pauli_flag is false or method isn't 3, bitstring will be None
    return qc2, bitstring    

    
############### Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw(hamiltonian: str = "hamlib", method: int = 1):
                          
    # Print a sample circuit
    print("Sample Circuit:")
    if QC_ is not None:
        print(f"  H = {HAM_}")
        print(QC_)

        if QCI_ is not None:
            print(f"  Initial State {QCI_.name}:")
            print(QCI_)
            
        # create a small circuit, just to display this evolution subciruit structure
        print("  Evolution Operator (e^-iHt) =")
        qctt = QuantumCircuit(QC_.num_qubits)
        qctt.append(EVO_, range(QC_.num_qubits))
        print(transpile(qctt, optimization_level=3))
        
        # create a small circuit, just to display this inverse evolution subcircuit structure        
        if INV_ is not None:                       
            print("  Inverse Evolution Operator (e^iHt) = Inverse of Above Circuit")
            try:
                qctt = QuantumCircuit(QC_.num_qubits)
                qctt.append(INV_, range(QC_.num_qubits))
                print(transpile(qctt, optimization_level=3))
            except:
                print(f"  WARNING: cannot display inverse circuit.")
    else:
        print("  ... circuit too large!")

