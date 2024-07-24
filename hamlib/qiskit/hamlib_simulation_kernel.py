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

from pygsti_mirror import convert_to_mirror_circuit


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
    time: float = 0.2,
    num_trotter_steps: int = 5,
    method=1,
    init_state=None,
    random_pauli_flag=False,
    random_pauli_bitstring = None 
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
        hamiltonian, num_qubits = process_data(data)

        # print("Number of qubits:", num_qubits)
        if verbose:
            print(f"... Evolution operator = {hamiltonian}")

        operator = hamiltonian  # Use the SparsePauliOp object directly
        # print (operator)

        # Build the evolution gate
        # label = "e\u2071\u1D34\u1D57"    # superscripted, but doesn't look good
        evo_label = "e^-iHt"
        evo = PauliEvolutionGate(operator, time=time/num_trotter_steps, label=evo_label)

        # Plug it into a circuit
        circuit = QuantumCircuit(operator.num_qubits)
        circuit_without_initial_state = QuantumCircuit(operator.num_qubits)
        
        # first insert the initial_state
        # init_state = "checkerboard"
        i_state = initial_state(num_qubits, init_state)
        circuit.append(i_state, range(operator.num_qubits))
        circuit.barrier()
        
        if n_spins <= 6:
            QCI_ = i_state
        
        # Append K trotter steps
        circuit = create_trotter_steps(num_trotter_steps, evo, operator, circuit)
        circuit_without_initial_state = create_trotter_steps(num_trotter_steps, evo, operator, circuit_without_initial_state)

        # if method 3 with random pauli flag, we use a random initial state rather than the supplied one
        inv = None
        if method == 3 and not random_pauli_flag:
            INV_ = inv = evo.inverse()
            inv.name = "e^iHt"
            circuit = create_trotter_steps(num_trotter_steps, inv, operator, circuit)

        elif method == 3 and random_pauli_flag:
            circuits, bitstrings = convert_to_mirror_circuit(circuit_without_initial_state, random_pauli_bitstring)

            return circuits, bitstrings, hamiltonian, evo

        #if not random_pauli_flag:
        circuit.measure_all()

        return circuit, hamiltonian, evo

    else:
        # print(f"Dataset not available for n_spins = {n_spins}.")
        return None, None, None


############### Circuit Definition

def initial_state(n_spins: int, initial_state: str = "checker") -> QuantumCircuit:
    """
    Initialize the quantum state.
    
    Args:
        n_spins (int): Number of spins (qubits).
        initial_state (str): The chosen initial state. By default applies the checkerboard state, but can also be set to "ghz", the GHZ state.

    Returns:
        QuantumCircuit: The initialized quantum circuit.
    """
    qc = QuantumCircuit(n_spins)

    if initial_state.strip().lower() == "checkerboard" or initial_state.strip().lower() == "neele":
        # Checkerboard state, or "Neele" state
        qc.name = "Neele"
        for k in range(0, n_spins, 2):
            qc.x([k])
    elif initial_state.strip().lower() == "ghz":
        # GHZ state: 1/sqrt(2) (|00...> + |11...>)
        qc.name = "GHZ"
        qc.h(0)
        for k in range(1, n_spins):
            qc.cx(k-1, k)

    return qc


def HamiltonianSimulation(n_spins: int, K: int, t: float,
            hamiltonian: str, w: float, hx: list[float], hz: list[float],
            use_XX_YY_ZZ_gates: bool = False, init_state=None,
            method: int = 1, random_pauli_flag = False) -> QuantumCircuit:
    """
    Construct a Qiskit circuit for Hamiltonian simulation.

    Args:
        n_spins (int): Number of spins (qubits).
        K (int): The Trotterization order.
        t (float): Duration of simulation.
        hamiltonian (str): Which hamiltonian to run. "heisenberg" by default but can also choose "TFIM". 
        w (float): Strength of two-qubit interactions for heisenberg hamiltonian. 
        hx (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 
        hz (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 

    Returns:
        QuantumCircuit: The constructed Qiskit circuit.
    """
    num_qubits = n_spins
    secret_int = f"{K}-{t}"

    # Allocate qubits
    qr = QuantumRegister(n_spins)
    cr = ClassicalRegister(n_spins)
    qc = QuantumCircuit(qr, cr, name=f"hamsim-{num_qubits}-{secret_int}")
    tau = t / K

    h_x = hx[:n_spins]
    h_z = hz[:n_spins]

    hamiltonian = hamiltonian.strip().lower()

    if method == 3 and random_pauli_flag: 
        qcs, bss, ham_op, evo = create_circuit(
            n_spins=n_spins,
            time=t,
            method=method,
            init_state=init_state,
            num_trotter_steps=K,
        )
        qc2s = [qc.decompose().decompose() for qc in qcs]
        # Save smaller circuit example for display

        global QC_, HAM_, EVO_, INV_
        if n_spins <= 6:
            # just show the first random pauli circuit
            QC_ = qc2s[0]
            HAM_ = ham_op
            EVO_ = evo
            #INV_ = inv

        return qc2s, bss
    else: 
        qc, ham_op, evo = create_circuit(
            n_spins=n_spins,
            time=t,
            method=method,
            init_state=init_state,
            num_trotter_steps=K,
        )

        # Collapse the sub-circuits used in this benchmark (for Qiskit)
        qc2 = qc.decompose().decompose()
        # Save smaller circuit example for display
        global QC_, HAM_, EVO_, INV_
        if n_spins <= 6:
            QC_ = qc
            HAM_ = ham_op
            EVO_ = evo
            #INV_ = inv

        return qc2
            
    
    
############### Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw(hamiltonian: str = "hamlib", use_XX_YY_ZZ_gates: bool = False, method: int = 1):
                          
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
            '''  DEVNOTE: This fails on some systems with an error about mismatch of Q and C widths
            '''
            try:
                qctt = QuantumCircuit(QC_.num_qubits)
                qctt.append(INV_, range(QC_.num_qubits))
                print(transpile(qctt, optimization_level=3))
            except:
                print(f"  WARNING: cannot display inverse circuit.")
            
    
    else:
        print("  ... circuit too large!")


    
