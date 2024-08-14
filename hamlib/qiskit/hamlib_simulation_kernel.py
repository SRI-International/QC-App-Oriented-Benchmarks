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

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
import h5py
import re
import os
import requests
import zipfile
import json
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate

verbose = False

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


from hamlib_utils import (
    process_hamiltonian_file,
    needs_normalization,
    normalize_data_format,
    parse_hamiltonian_to_sparsepauliop,
    determine_qubit_count,
)

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


def process_data(data):
    """
    Process the given data to construct a Hamiltonian in the form of a SparsePauliOp and determine the number of qubits.

    Args:
        data (str or bytes): The Hamiltonian data to be processed. Can be a string or bytes.

    Returns:
        tuple: A tuple containing the Hamiltonian as a SparsePauliOp and the number of qubits.
    """
    if needs_normalization(data) == "Yes":
        data = normalize_data_format(data)
    parsed_pauli_list = parse_hamiltonian_to_sparsepauliop(data)
    num_qubits = determine_qubit_count(parsed_pauli_list)
    hamiltonian = sparse_pauliop(parsed_pauli_list, num_qubits)
    return hamiltonian, num_qubits


def sparse_pauliop(terms, num_qubits):
    """
    Construct a SparsePauliOp from a list of Pauli terms and the number of qubits.

    Args:
        terms (list): A list of tuples, where each tuple contains a dictionary representing the Pauli operators and 
                      their corresponding qubit indices, and a complex coefficient.
        num_qubits (int): The total number of qubits.

    Returns:
        SparsePauliOp: The Hamiltonian represented as a SparsePauliOp.
    """
    pauli_list = []
    
    for pauli_dict, coefficient in terms:
        label = ['I'] * num_qubits  # Start with identity on all qubits
        for qubit, pauli_op in pauli_dict.items():
            label[qubit] = pauli_op
        label = ''.join(label)
        pauli_list.append((label, coefficient))
    
    hamiltonian = SparsePauliOp.from_list(pauli_list, num_qubits=num_qubits)
    return hamiltonian

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

    for qubits in qubit_candidates:
        initial_n_spins = qubits // 2 if "{n_qubits/2}" in dataset_name_template else qubits
        n_spins = initial_n_spins

        # print(f"Starting check for qubits = {qubits}, initial n_spins = {n_spins}")

        found_valid_dataset = False

        while n_spins <= max_qubits:
            dataset_name_template = dataset_name_template.replace("{ratio}", str(global_ratio)).replace("{rinst}", str(global_rinst))
            dataset_name_template = dataset_name_template.replace("{h}", str(global_h)).replace("{pbc_val}", str(global_pbc_val))
            dataset_name_template = dataset_name_template.replace("{U}", str(global_U)).replace("{enc}", str(global_enc))
            dataset_name = dataset_name_template.replace("{n_qubits}", str(n_spins)).replace("{n_qubits/2}", str(n_spins))
            # print(f"Checking dataset: {dataset_name}")

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

# In hamiltonian_simulation_kernel.py

dataset_name_template = ""
filename = ""

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

def create_circuit(
    n_spins: int,
    time: float = 1,
    num_trotter_steps: int = 5,
    method: int = 1,
    use_inverse_flag: bool = False,
    init_state: str = None,
    random_pauli_flag: bool = False,
    random_init_flag: bool = False 
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
    global dataset_name_template, filename
    global global_h, global_pbc_val
    global global_U, global_enc
    global global_ratio, global_rinst
    global QCI_, INV_

    # Replace placeholders with actual n_qubits value: n_spins
    dataset_name_template = dataset_name_template.replace("{ratio}", str(global_ratio)).replace("{rinst}", str(global_rinst))
    dataset_name_template = dataset_name_template.replace("{h}", str(global_h)).replace("{pbc_val}", str(global_pbc_val))
    dataset_name_template = dataset_name_template.replace("{U}", str(global_U)).replace("{enc}", str(global_enc))
    dataset_name = dataset_name_template.replace("{n_qubits}", str(n_spins)).replace("{n_qubits/2}", str(n_spins // 2))

    if verbose:
        print(f"Trying dataset: {dataset_name}")  # Debug print

    data = process_hamiltonian_file(filename, dataset_name)
    if data is not None:
        # print(f"Using dataset: {dataset_name}")
        # print("Raw Hamiltonian Data: ", data)
        
        # get the Hamiltonian operator as SparsePauliOp and its size from the data
        ham_op, num_qubits = process_data(data)

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
            n_spins: int,
            hamiltonian: str, 
            K: int = 5, t: float = 1.0,
            init_state = None,
            method: int = 1,
            use_inverse_flag: bool = False,
            random_pauli_flag = False,
            random_init_flag = False
        ) -> QuantumCircuit:
    """
    Construct a Qiskit circuit for Hamiltonian simulation.

    Args:
        n_spins (int): Number of spins (qubits).
        hamiltonian (str): Which hamiltonian to run. "heisenberg" by default but can also choose "TFIM". 
        K (int): The Trotterization order.
        t (float): Duration of simulation.
        method (int): Type of comparison for fidelity
        random_pauli_flag (bool): Insert random Pauli gates if method 3

    Returns:
        QuantumCircuit: The constructed Qiskit circuit.
    """
    num_qubits = n_spins
    secret_int = f"{K}-{t}"

    # Allocate qubits
    qr = QuantumRegister(n_spins)
    cr = ClassicalRegister(n_spins)
    qc = QuantumCircuit(qr, cr, name=f"hamsim-{num_qubits}-{secret_int}")

    hamiltonian = hamiltonian.strip().lower()
    
    # create the quantum circuit for this Hamiltonian, along with the correct pauli bstring,
    # the operator and trotter evolution circuit
    qc, bitstring, ham_op, evo = create_circuit(
        n_spins=n_spins,
        time=t,
        method=method,
        use_inverse_flag=use_inverse_flag,
        init_state=init_state,
        num_trotter_steps=K,
        random_pauli_flag=random_pauli_flag,
        random_init_flag=random_init_flag
        )

    # Save smaller circuit example for display
    global QC_, HAM_, EVO_, INV_
    if n_spins <= 6:
        QC_ = qc
        HAM_ = ham_op
        EVO_ = evo
        #INV_ = inv
            
    # Collapse the sub-circuits used in this benchmark (for Qiskit)
    qc2 = qc.decompose().decompose()

    # return both the circuit created, the bitstring, and the Hamiltonian operator
    # if random_pauli_flag is false or method isn't 3, bitstring will be None
    return qc2, bitstring, ham_op
        

    
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


