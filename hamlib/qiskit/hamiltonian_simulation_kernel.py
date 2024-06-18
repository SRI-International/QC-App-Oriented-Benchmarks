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

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
# from hamlib_test import create_circuit, HamiltonianSimulationExact
import h5py
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

# Saved circuits and subcircuits for display
QC_ = None
XX_ = None
YY_ = None
ZZ_ = None
XXYYZZ_ = None

# Mirror Gates of the previous four gates
XX_mirror_ = None
YY_mirror_ = None
ZZ_mirror_ = None
XXYYZZ_mirror_ = None

# For validating the implementation of XXYYZZ operation (saved for possible use in drawing)
_use_XX_YY_ZZ_gates = False

def extract_dataset_hdf5_2(filename, dataset_name):
    """
    Extract a dataset from an HDF5 file.
    
    Args:
        filename (str): The name of the HDF5 file.
        dataset_name (str): The name of the dataset to extract.

    Returns:
        data: The extracted data. Returns None if the dataset is not found.
    """
    # Open the HDF5 file
    with h5py.File(filename, 'r') as file:
        # Check if the dataset exists
        if dataset_name in file:
            dataset = file[dataset_name]
            # Check if the dataset is a scalar
            if dataset.shape == ():
                data = dataset[()]  # Correct way to read scalar data
            else:
                data = dataset[:]  # For non-scalar, use slicing
            # Convert bytes to string if necessary
            if isinstance(data, bytes):
                data = data.decode('utf-8')
        else:
            data = None
            print(f"Dataset {dataset_name} not found in the file.")
    return data

def parse_hamiltonian(hamiltonian):
    """
    Parse a Hamiltonian string into a list of Pauli terms with coefficients and involved qubits.
    
    Args:
        hamiltonian (str): The Hamiltonian string to be parsed.
    
    Returns:
        list: A list of tuples, each containing a Pauli string, a list of involved qubits, and a coefficient.
    """
    import re
    # Regular expression to capture numeric coefficients and the involved qubits
    term_pattern = re.compile(r'([+-]?\d*\.?\d*) \[([Z\d\s]*)\]')
    
    pauli_list = []
    
    # Iterate over all matches
    for match in term_pattern.finditer(hamiltonian):
        coefficient = float(match.group(1))  # Convert coefficient to float
        if not coefficient:  # Skip terms with a coefficient of 0
            continue
        
        # Extract qubits and construct the Pauli string
        qubits = []
        pauli_string = ''
        if match.group(2):  # Check if there are any qubits
            qubits_info = match.group(2).split()
            for qubit_info in qubits_info:
                qubit_index = int(qubit_info[1:])  # Extract the qubit index from the format Z0, Z1, etc.
                qubits.append(qubit_index)
                pauli_string += 'Z'
        
        pauli_list.append((pauli_string, qubits, coefficient))
    
    return pauli_list

def determine_qubit_count(terms):
    """
    Determine the number of qubits required for a given list of Pauli terms.
    
    Args:
        terms (list): A list of tuples, each containing a Pauli string, a list of involved qubits, and a coefficient.
    
    Returns:
        int: The total number of qubits required, determined by the highest qubit index in the terms.
    """
    max_qubit = 0
    for _, qubits, _ in terms:
        if qubits:  # Ensure the list is not empty
            max_in_term = max(qubits)
            if max_in_term > max_qubit:
                max_qubit = max_in_term
    return max_qubit + 1  # Since qubit indices start at 0

def sparse_pauliop(terms, num_qubits):
    """
    Create a SparsePauliOp representation from a list of Pauli terms and the number of qubits.
    
    Args:
        terms (list): A list of tuples, each containing a Pauli string, a list of involved qubits, and a coefficient.
        num_qubits (int): The total number of qubits in the system.
    
    Returns:
        SparsePauliOp: The Hamiltonian represented as a SparsePauliOp object.
    """
    edges = []
    pauli_list = []
    
    for pauli_string, qubits, coefficient in terms:
        # Generate PauliOp representation
        label = ''.join(['I']*num_qubits)  # Start with identity on all qubits
        if qubits:
            label_list = list(label)
            for qubit in qubits:
                label_list[qubit] = 'Z'
            label = ''.join(label_list)
        pauli_list.append((label, coefficient))
        
        # Generate edges for ZZ interactions (representing graph edges)
        if len(qubits) == 2 and pauli_string == 'ZZ':
            edges.append((qubits[0], qubits[1], coefficient))
    
    # Creating the SparsePauliOp from the list of Pauli terms
    hamiltonian = SparsePauliOp.from_list(pauli_list)
    return hamiltonian

def create_circuit():
    """
    Create and return a quantum circuit based on the Hamiltonian extracted from an HDF5 file.
    
    The function performs the following steps:
    1. Extracts the dataset from the HDF5 file.
    2. Parses the Hamiltonian string into Pauli terms.
    3. Determines the number of qubits required.
    4. Converts the Pauli terms into a SparsePauliOp object.
    5. Constructs a quantum circuit with an initial state and evolution gate.
    6. Measures all qubits and returns the circuit and Hamiltonian.

    Returns:
        tuple: A tuple containing the quantum circuit and the Hamiltonian as a SparsePauliOp object.
    """
    # Usage
    dataset_name = 'graph-1D-grid-nonpbc-qubitnodes_Lx-2_h-0.1'
    # filename = '/Users/avimitachatterjee/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/PSU/Internship/SRI/work/hamiltonnian/QC-App-Oriented-Benchmarks/hamiltonian-simulation/qiskit/tfim.hdf5'
    filename = '/Users/avimitachatterjee/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/PSU/Internship/SRI/work/hamiltonnian/hamlib_tfim/tfim.hdf5'
    data = extract_dataset_hdf5_2(filename, dataset_name)

    if data:
        parsed_pauli_list = parse_hamiltonian(data)
        print(parsed_pauli_list)
    else:
        print("No data extracted.")

    # Usage
    num_qubits = determine_qubit_count(parsed_pauli_list)
    hamiltonian = sparse_pauliop(parsed_pauli_list, num_qubits)

    print("Number of qubits",num_qubits)
    print("Hamiltonian:", hamiltonian)
    
    # Assuming 'hamiltonian' is already defined as a SparsePauliOp object from your previous code:
    operator = hamiltonian  # Use the SparsePauliOp object directly
    time = 0.2

    # Build the evolution gate
    evo = PauliEvolutionGate(operator, time=time)

    # Plug it into a circuit
    circuit = QuantumCircuit(operator.num_qubits)
    

    init_state = "checkerboard"

    # apply initial state
    circuit.append(initial_state(num_qubits, init_state), range(operator.num_qubits))
    circuit.barrier()
    circuit.append(evo, range(operator.num_qubits))
    circuit.barrier()
    circuit.measure_all() 
    print (circuit)
    circuit.draw(output="mpl") 
    global QC_
    QC_ = circuit
    return circuit, hamiltonian
    #        

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
        for k in range(0, n_spins, 2):
            qc.x([k])
    elif initial_state.strip().lower() == "ghz":
        # GHZ state: 1/sqrt(2) (|00...> + |11...>)
        qc.h(0)
        for k in range(1, n_spins):
            qc.cx(k-1, k)

    return qc


def HamiltonianSimulation(n_spins: int, K: int, t: float,
            hamiltonian: str, w: float, hx: list[float], hz: list[float],
            use_XX_YY_ZZ_gates: bool = False,
            method: int = 1) -> QuantumCircuit:
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
    global _use_XX_YY_ZZ_gates
    _use_XX_YY_ZZ_gates = use_XX_YY_ZZ_gates
    
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

    if hamiltonian == "heisenberg": 

        init_state = "checkerboard"

        # apply initial state
        qc.append(initial_state(n_spins, init_state), qr)
        qc.barrier()

        # Loop over each Trotter step, adding gates to the circuit defining the Hamiltonian
        for k in range(K):
            # Pauli spin vector product
            [qc.rx(2 * tau * w * h_x[i], qr[i]) for i in range(n_spins)]
            [qc.rz(2 * tau * w * h_z[i], qr[i]) for i in range(n_spins)]
            qc.barrier()
            
            # Basic implementation of exp(i * t * (XX + YY + ZZ))
            if use_XX_YY_ZZ_gates:
                # XX operator on each pair of qubits in linear chain
                for j in range(2):
                    for i in range(j%2, n_spins - 1, 2):
                        qc.append(xx_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

                # YY operator on each pair of qubits in linear chain
                for j in range(2):
                    for i in range(j%2, n_spins - 1, 2):
                        qc.append(yy_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

                # ZZ operation on each pair of qubits in linear chain
                for j in range(2):
                    for i in range(j%2, n_spins - 1, 2):
                        qc.append(zz_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
                        
            else:
                # Optimized XX + YY + ZZ operator on each pair of qubits in linear chain
                for j in reversed(range(2)):
                    for i in reversed(range(j % 2, n_spins - 1, 2)):
                        qc.append(xxyyzz_opt_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
            qc.barrier()

        if (method == 3):
            # Add mirror gates for negative time simulation
            for k in range(K): 
                # Basic implementation of exp(-i * t * (XX + YY + ZZ)):
                if use_XX_YY_ZZ_gates:
                    # regular inverse of XX + YY + ZZ operators on each pair of quibts in linear chain
                    # XX operator on each pair of qubits in linear chain
                    for j in range(2):
                        for i in range(j%2, n_spins - 1, 2):
                            qc.append(zz_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

                    # YY operator on each pair of qubits in linear chain
                    for j in range(2):
                        for i in range(j%2, n_spins - 1, 2):
                            qc.append(yy_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

                    # ZZ operation on each pair of qubits in linear chain
                    for j in range(2):
                        for i in range(j%2, n_spins - 1, 2):
                            qc.append(xx_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

                else:
                    # optimized Inverse of XX + YY + ZZ operator on each pair of qubits in linear chain
                    for j in range(2):
                        for i in range(j % 2, n_spins - 1, 2):
                            qc.append(xxyyzz_opt_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
                qc.barrier()

                # the Pauli spin vector product
                [qc.rz(-2 * tau * w * h_z[i], qr[i]) for i in range(n_spins)]
                [qc.rx(-2 * tau * w * h_x[i], qr[i]) for i in range(n_spins)]
                qc.barrier()
    
    elif hamiltonian == "tfim":
        h = 0.2  # Strength of transverse field
        init_state = "ghz"

        #apply initial state
        qc.append(initial_state(n_spins, init_state), qr)
        qc.barrier()

        # Calculate TFIM
        for k in range(K):
            for i in range(n_spins):
                qc.rx(2 * tau * h, qr[i])
            qc.barrier()

            for j in range(2):
                for i in range(j % 2, n_spins - 1, 2):
                    qc.append(zz_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
            qc.barrier()
        
        if (method == 3):
            for k in range(k):
                for j in range(2):
                    for i in range(j % 2, n_spins - 1, 2):
                        qc.append(zz_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
                qc.barrier()
                for i in range(n_spins):
                    qc.rx(-2 * tau * h, qr[i])
                qc.barrier()
    
    elif hamiltonian == "hamlib":
        qc, _ = create_circuit()

    else:
        raise ValueError("Invalid Hamiltonian specification.")

    # Measure all qubits
    # for i_qubit in range(n_spins):
    #     qc.measure(qr[i_qubit], cr[i_qubit])

    # Save smaller circuit example for display
    global QC_
    if QC_ is None or n_spins <= 6:
        if n_spins < 9:
            QC_ = qc

    # Collapse the sub-circuits used in this benchmark (for Qiskit)
    qc2 = qc.decompose()
            
    return qc2
    

############### XX, YY, ZZ Gate Implementations

def xx_gate(tau: float) -> QuantumCircuit:
    """
    Simple XX gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The XX gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="xx_gate")
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    
    global XX_
    XX_ = qc
    
    return qc

def yy_gate(tau: float) -> QuantumCircuit:
    """
    Simple YY gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The YY gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="yy_gate")
    qc.s(qr[0])
    qc.s(qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.sdg(qr[0])
    qc.sdg(qr[1])

    global YY_
    YY_ = qc

    return qc

def zz_gate(tau: float) -> QuantumCircuit:
    """
    Simple ZZ gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The ZZ gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="zz_gate")
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])

    global ZZ_
    ZZ_ = qc

    return qc

def xxyyzz_opt_gate(tau: float) -> QuantumCircuit:
    """
    Optimal combined XXYYZZ gate (with double coupling) on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The optimal combined XXYYZZ gate circuit.
    """
    alpha = tau
    beta = tau
    gamma = tau
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="xxyyzz_opt")
    qc.rz(3.1416 / 2, qr[1])
    qc.cx(qr[1], qr[0])
    qc.rz(3.1416 * gamma - 3.1416 / 2, qr[0])
    qc.ry(3.1416 / 2 - 3.1416 * alpha, qr[1])
    qc.cx(qr[0], qr[1])
    qc.ry(3.1416 * beta - 3.1416 / 2, qr[1])
    qc.cx(qr[1], qr[0])
    qc.rz(-3.1416 / 2, qr[0])

    global XXYYZZ_
    XXYYZZ_ = qc

    return qc

    


############### Mirrors of XX, YY, ZZ Gate Implementations   
def xx_gate_mirror(tau: float) -> QuantumCircuit:
    """
    Simple XX mirror gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The XX_mirror_ gate circuit.
    """
    qr = QuantumRegister(2, 'q')
    qc = QuantumCircuit(qr, name="xx_gate_mirror")
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(-3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])

    global XX_mirror_
    XX_mirror_ = qc

    return qc

def yy_gate_mirror(tau: float) -> QuantumCircuit:
    """
    Simple YY mirror gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The YY_mirror_ gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="yy_gate_mirror")
    qc.s(qr[0])
    qc.s(qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(-3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.sdg(qr[0])
    qc.sdg(qr[1])

    global YY_mirror_
    YY_mirror_ = qc

    return qc   

def zz_gate_mirror(tau: float) -> QuantumCircuit:
    """
    Simple ZZ mirror gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The ZZ_mirror_ gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="zz_gate_mirror")
    qc.cx(qr[0], qr[1])
    qc.rz(-3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])

    global ZZ_mirror_
    ZZ_mirror_ = qc

    return qc

def xxyyzz_opt_gate_mirror(tau: float) -> QuantumCircuit:
    """
    Optimal combined XXYYZZ mirror gate (with double coupling) on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The optimal combined XXYYZZ_mirror_ gate circuit.
    """
    alpha = tau
    beta = tau
    gamma = tau
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="xxyyzz_opt_mirror")
    qc.rz(3.1416 / 2, qr[0])
    qc.cx(qr[1], qr[0])
    qc.ry(-3.1416 * beta + 3.1416 / 2, qr[1])
    qc.cx(qr[0], qr[1])
    qc.ry(-3.1416 / 2 + 3.1416 * alpha, qr[1])
    qc.rz(-3.1416 * gamma + 3.1416 / 2, qr[0])
    qc.cx(qr[1], qr[0])
    qc.rz(-3.1416 / 2, qr[1])

    global XXYYZZ_mirror_
    XXYYZZ_mirror_ = qc

    return qc


############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw(hamiltonian: str = "heisenberg", use_XX_YY_ZZ_gates: bool = False, method: int = 1):
                          
    # Print a sample circuit
    print("Sample Circuit:")
    print(QC_ if QC_ is not None else "  ... too large!")

    if hamiltonian == "heisenberg": 
        if use_XX_YY_ZZ_gates:
            print("\nXX, YY, ZZ = ")
            print(XX_)
            print(YY_)
            print(ZZ_)
            if method == 3:
                print("\nXX, YY, ZZ mirror = ")
                print(XX_mirror_)
                print(YY_mirror_)
                print(ZZ_mirror_)
        else:
            print("\nXXYYZZ_opt = ")
            print(XXYYZZ_)  
            if method == 3:
                print("\nXXYYZZ_opt_mirror = ")
                print(XXYYZZ_mirror_)
    
    if hamiltonian == "tfim": 
        print("\nZZ = ")
        print(ZZ_)
        if method == 3:
            print("\nZZ mirror = ")
            print(ZZ_mirror_)

    if hamiltonian == "hamlib": 
        print("\n circuit = ")
        print(QC_)
        if method == 3:
            print("\nZZ mirror = ")
            print(ZZ_mirror_)
    