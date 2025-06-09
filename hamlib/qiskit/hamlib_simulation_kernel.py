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

# Saved circuits and subcircuits for display
QC_ = None
QCI_ = None
HAM_ = None
EVO_ = None
INV_ = None

    
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
    if verbose:
        print(f"  ... ensure_sparse_pauli_op({input_data}, {num_qubits})")
        
    if input_data is None or num_qubits < 1:
        #return SparsePauliOp.from_list([], num_qubits=num_qubits)
        return None     # this is a better indicator of invalid terms, for now
        
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
# KERNEL FUNCTIONS (EXTRACTED FROM OBSERVABLES)

# The next 6 functions below were copied from the observables module.
# Merging these into the kernel is a work-in-progress

import numpy as np
import copy
from math import sin, cos, pi
import time

from qiskit import QuantumCircuit
from qiskit_aer import Aer
#from qiskit.primitives import Estimator
from qiskit_ibm_runtime import Estimator

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
    num_qubits = qc.num_qubits
    is_diag = True  # Tracks if the term is diagonal (currently unused)
    for i, p in enumerate(pauli):
        ii = num_qubits - i -1
        if p == "X":
            is_diag = False
            qc.h(ii)
        elif p == "Y":
            qc.sdg(ii)
            qc.h(ii)
            is_diag = False

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
    
    
#####################################################################################
# KERNEL FUNCTIONS
 
################ Create Trotterized Circuit with Initial State   
   
def create_circuit_from_op(
    num_qubits: int = 0,
    ham_op: Union[
        List[Tuple[str, complex]],
        List[Tuple[Dict[int, str], complex]],
        SparsePauliOp
    ] = None, 
    time: float = 1,
    num_trotter_steps: int = 5,
    method: int = 1,
    use_inverse_flag: bool = False,
    init_state: str = None,
    append_measurements: bool = True
):
    """
    Create a quantum circuit based on the given Hamiltonian data.

    Steps:
        1. Extract Hamiltonian data from an HDF5 file.
        2. Process the data to obtain a SparsePauliOp and determine the number of qubits.
        3. Build a quantum circuit with an initial state and an evolution gate based on the Hamiltonian.
        4. Measure all qubits and print the circuit details.

    Returns:
        tuple: A tuple containing the constructed QuantumCircuit and the Hamiltonian as a SparsePauliOp.
    """
    global QCI_, INV_
    
    time_step = time/num_trotter_steps if num_trotter_steps > 0 else 0.0

    # Build the evolution gate
    # label = "e\u2071\u1D34\u1D57"    # superscripted, but doesn't look good
    evo_label = "e^-iHt"
    
    #Check if ham_op is a list of lists
    if isinstance(ham_op, list) and isinstance(ham_op[0], list):
        # construct the evo circuit from the groups
        evo = []
        for ham_op_group in ham_op:
            # convert from any form to SparsePauliOp
            ham_op_group = ensure_sparse_pauli_op(ham_op_group, num_qubits)
            evo += [PauliEvolutionGate(ham_op_group, time=time_step, label=evo_label)]    
    else:
        # convert from any form to SparsePauliOp
        ham_op = ensure_sparse_pauli_op(ham_op, num_qubits)
        evo = [PauliEvolutionGate(ham_op, time=time_step, label=evo_label)]
        
    # Plug it into a circuit
    circuit = QuantumCircuit(num_qubits)
    
    # first create and append the initial_state
    i_state = None
    if init_state is not None:
        i_state = initial_state(num_qubits, init_state)
        circuit.append(i_state, range(num_qubits))
        circuit.barrier()
    
    if num_qubits <= 6:
        QCI_ = i_state
    
    # Append K trotter steps
    evo_inverse = [x.inverse() for x in evo]
    
    circuit = append_trotter_steps(num_trotter_steps,
            evo if not use_inverse_flag else evo_inverse,
            num_qubits,
            circuit)

    # Append K Trotter steps of inverse, if method 3
    inv = None
    bitstring = None

    # append simple inverse Trotter steps if method 3
    if method == 3:  
        inv = evo_inverse
        for x in inv: x.name = "e^iHt"
        circuit = append_trotter_steps(num_trotter_steps, inv, ham_op.num_qubits, circuit)
        if num_qubits <= 6:
            INV_ = inv
        
    # if requested, add measurement gates    
    if append_measurements:
        circuit.measure_all()

    return circuit, bitstring, evo if not use_inverse_flag else evo.inverse()


################ Create Trotterized Circuit with Initial State
#               (Mirrored using pyGSTi methodology)

def create_circuit_from_op_pygsti(
    num_qubits: int = 0,
    ham_op: Union[
        List[Tuple[str, complex]],
        List[Tuple[Dict[int, str], complex]],
        SparsePauliOp
    ] = None, 
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
    Create a quantum circuit from the given Hamiltonian data.
    This version creates a special mirrored circuit using functions from pyGSTi library.

    Steps:
        1. Extract Hamiltonian data from an HDF5 file.
        2. Process the data to obtain a SparsePauliOp and determine the number of qubits.
        3. Build a quantum circuit with an initial state and an evolution gate based on the Hamiltonian.
        4. Measure all qubits and print the circuit details.

    Returns:
        tuple: A tuple containing the constructed QuantumCircuit and the Hamiltonian as a SparsePauliOp.
    """
    # do this here, so import not required by default
    from pygsti_mirror import convert_to_mirror_circuit
       
    global QCI_, INV_
    
    # convert from any form to SparsePauliOp
    ham_op = ensure_sparse_pauli_op(ham_op, num_qubits)

    # Build the evolution gate
    # label = "e\u2071\u1D34\u1D57"    # superscripted, but doesn't look good
    evo_label = "e^-iHt"
    time_step = time/num_trotter_steps if num_trotter_steps > 0 else 0.0
    evo = PauliEvolutionGate(ham_op, time=time_step, label=evo_label)

    # Create a circuit, but with no initial state
    circuit_without_initial_state = QuantumCircuit(ham_op.num_qubits)
    
    # Append K trotter steps           
    circuit_without_initial_state = append_trotter_steps(num_trotter_steps,
            evo if not use_inverse_flag else evo.inverse(),
            ham_op,
            circuit_without_initial_state)

    # Append K Trotter steps of inverse, if method 3
    bitstring = None
    
    # if random init flag state is set, then discard the inital state input and use a completely (harr) random one
    if random_init_flag:
        circuit, bitstring = convert_to_mirror_circuit(
                circuit_without_initial_state,
                random_pauli = True,
                init_state = None)
    else: 
        i_state = initial_state(num_qubits, init_state)
        circuit, bitstring = convert_to_mirror_circuit(
                circuit_without_initial_state,
                random_pauli = True,
                init_state = i_state
                )
        if num_qubits <= 6:
            QCI_ = i_state
        
    return circuit, bitstring, evo if not use_inverse_flag else evo.inverse()


################ Append Trotter steps   
 
def append_trotter_steps(num_trotter_steps, evo, num_qubits, circuit):
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
        for evo_group in evo:
            circuit.append(evo_group, range(num_qubits))
    circuit.barrier()
    return circuit

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
            
    elif set(init_state).issubset({'0', '1'}):
        qc.name = "b" + init_state
        for k in range(0, n_spins):
            if init_state[k] == '1':
                qc.x([n_spins - k - 1])

    return qc
    
######################################################################
# EXTERNAL API FUNCTIONS
   
############### Hamiltonian Circuit Definition

def HamiltonianSimulation(
        num_qubits: int = 0,
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
        initial_circuit = None,
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
    if num_qubits <= 0:
        return None, None
        
    circuit_id = f"{K}-{t}"

    # Allocate qubits in a QuantumCircuit
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(qr, cr, name=f"hamsim-{num_qubits}-{circuit_id}")
    
    # if no Hamiltonian given, just return initial state is specified
    # DEVNOTE: this is not assigning the proper useful circuit name
    if ham_op is None:
    
        # if a string is passed in, create initialized state from it
        if init_state is not None and isinstance(init_state, str):
            qc = initial_state(num_qubits, init_state)
            
        # if initial circuit is passed in , then just use it
        elif initial_circuit is not None:
            qc = initial_circuit.copy()
            
        return qc, None
    
    # print("Number of qubits:", num_qubits)
    if verbose:
        print(f"... HamiltonianSimulation(), with evolution operator = {ham_op}")
        
    # create a Trotterized quantum circuit for this Hamiltonian, with various options
    if not random_pauli_flag:
        qc, bitstring, evo = create_circuit_from_op(
            num_qubits=num_qubits,
            ham_op=ham_op,
            method=method,
            init_state=init_state if initial_circuit is None else None,
            time=t,
            num_trotter_steps=K,
            append_measurements=append_measurements,
            use_inverse_flag=use_inverse_flag
            )
    # to generate circuits with random paulis, use the pygsti version
    else:
        qc, bitstring, evo = create_circuit_from_op_pygsti(
            num_qubits=num_qubits,
            ham_op=ham_op,
            method=method,
            init_state=init_state if initial_circuit is None else None,
            time=t,
            num_trotter_steps=K,
            append_measurements=append_measurements,
            use_inverse_flag=use_inverse_flag,
            random_pauli_flag=random_pauli_flag,
            random_init_flag=random_init_flag
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
    
    # an initial circuit passed in must be cloned before use, since it wasn't created here
    if initial_circuit:
        initial_circuit.compose(qc2, qubits=list(range(qc2.num_qubits)), inplace=True)
        qc2 = initial_circuit.copy()
        QCI_ = qc2
        
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
        #print(EVO_)
        for evo in EVO_:
            qctt.append(evo, range(QC_.num_qubits))
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

