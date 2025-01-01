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

import hamlib_utils

verbose = False

# Saved circuits and subcircuits for display
QC_ = None
QCI_ = None
HAM_ = None
EVO_ = None
INV_ = None


#####################################################################################
# GLOBAL SETTINGS FOR PARAMETERS

# These will be deprecated and removed soon.
# Need to first convert all the notebooks to pass in the parameter array.
# (and possibly figure out when to do the active_hamiltonian_datasets initialize to None.)

# global vars that specify the current hdf5 filename, dataset template, and parameters set by the user
dataset_name_template = ""
filename = ""

global_U = None
global_enc = None
global_ratio = None
global_rinst = None
global_h = None
global_pbc_val = None
        
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
        num_qubits=num_qubits,
        ham_op=ham_op,
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

