'''
Phase Estimation Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

from typing import Union, List, Tuple, Dict
import numpy as np

import cudaq
from cudaq import spin
    
# saved circuits for display
QC_ = None
Uf_ = None

############### State Initialization

# Prepare the initial quantum state of the system. 

@cudaq.kernel
def get_initial_state(n_spins: int) -> None:
    """Create initial state |1010...>"""
    qubits = cudaq.qvector(n_spins)
    for i in range(0, n_spins, 2):
        x(qubits[i])
        
 
############### Hamiltonian Trotterization

"""
This function performs a single Trotter step for simulating a spin chain.
If _use_XXYYZZ_gate is True, it constructs specific two-qubit gates using decomposition
for exponentiation.
It is a custom and efficient exponentiation combination of XX+YY+ZZ gates as a single operation.
Otherwise, it uses a standrd CUDA-Q exp_pauli operation for the exponentiation of
Pauli gates in Hamiltonian.

Input parameters:
- **state:** The current quantum state.  
- **dt:** Time step for Trotter evolution.  
- **Jx, Jy, Jz:** Coupling constants for the spin-spin interactions along the x, y, and z axes.  
- **h_x, h_y, h_z:** Local magnetic field strengths (unused in the code provided).  
- **_use_XXYYZZ_gate:** Flag to determine if custom XX+YY+ZZ gates should be used.  
- **coefficients:** Coefficients of Pauli terms for exponential evolution.  
- **words:** Corresponding Pauli operators (e.g., X, Y, Z) for evolution.  
"""

@cudaq.kernel
def trotter_step(state: cudaq.State, dt: float, Jx: float, Jy: float, Jz: float,
                 h_x: list[float], h_y: list[float], h_z: list[float], _use_XXYYZZ_gate: bool,
                 coefficients: List[complex], words: List[cudaq.pauli_word]) -> None:
    """Perform single Trotter step"""
    qubits = cudaq.qvector(state)
    n_spins = len(qubits)
   
    # Apply two-qubit interaction terms
    if _use_XXYYZZ_gate:
        for j in range(2):
            for i in range(j % 2, n_spins - 1, 2):
                rx(-np.pi/2,qubits[i])
                rx(np.pi/2,qubits[i+1])
                x.ctrl(qubits[i], qubits[i+1])
                h(qubits[i])
                s(qubits[i])
                rz(-2*Jy*dt,qubits[i+1])
                x.ctrl(qubits[i], qubits[i+1])
                h(qubits[i])
                rx(2*Jx*dt,qubits[i])
                rz(-2*Jz*dt,qubits[i+1])
                x.ctrl(qubits[i], qubits[i+1])
    else:
        for i in range(len(coefficients)):
            exp_pauli(coefficients[i].real * dt, qubits, words[i])
            

############### Construct Heisenberg Hamiltonian

"""
Construct the Heisenberg Hamiltonian as a cudaq.SpinOperator object,
considering nearest-neighbor interactions along X, Y, and Z directions.
It supports arbitrary coupling constants Jx, Jy, and Jz. See formula above. Input:

n_spins: Number of spins (qubits) in the system.
Jx, Jy, Jz: Coupling constants for interactions in the X, Y, and Z directions.
h_x, h_y, h_z: Magnetic field components (currently unused).
"""

def create_hamiltonian_heisenberg(n_spins: int, Jx: float, Jy: float, Jz: float, h_x: list[float], h_y: list[float], h_z: list[float]) -> cudaq.SpinOperator:
    """Create the Hamiltonian operator"""
    ham = cudaq.SpinOperator(num_qubits=n_spins)

    # Add two-qubit interaction terms for Heisenberg Hamiltonian
    for i in range(0, n_spins - 1):
        ham += Jx * spin.x(i) * spin.x(i + 1)
        ham += Jy * spin.y(i) * spin.y(i + 1)
        ham += Jz * spin.z(i) * spin.z(i + 1)
   
    return ham

############### Construct TFIM Hamiltonian

"""
Construct the TFIM Hamiltonian as a cudaq.SpinOperator object. See formula above.
"""

def create_hamiltonian_tfim(n_spins: int, h_field: float) -> cudaq.SpinOperator:
    """Create the Hamiltonian operator"""
    ham = cudaq.SpinOperator(num_qubits=n_spins)
   
    # Add single-qubit terms
    for i in range(0, n_spins):
        ham += -1 * h_field * spin.x(i)

    # Add two-qubit interaction terms for Ising Hamiltonian
    for i in range(0, n_spins-1):
        ham += -1 * spin.z(i) * spin.z(i + 1)
   
    return ham
    
 
############### Extract Coefficients and Pauli Words

"""
Extract the coefficients and Pauli words from the provided Hamiltonian for use in the Trotter step.
"""
def extractCoefficients(hamiltonian: cudaq.SpinOperator) -> List[complex]:
    result = []
    hamiltonian.for_each_term(lambda term: result.append(term.get_coefficient()))
    return result

def extractWords(hamiltonian: cudaq.SpinOperator) -> List[str]:
    result = []
    hamiltonian.for_each_term(lambda term: result.append(term.to_string(False)))
    return result
 
 
#####################################################           
 
############### Convert Hamiltonian to Spin Operator
           
def convert_to_spin_op (num_qubits: int,
        ham_op: Union[
            List[Tuple[str, complex]],
            List[Tuple[Dict[int, str], complex]]
        ] = None,
    ) -> cudaq.SpinOperator:

    # Parameters
    n_spins = num_qubits  # Number of spins in the chain
    
    spin_op = cudaq.SpinOperator(num_qubits=n_spins)
    
    for term in ham_op:
        #print(f"... term = {term}")
        qops = term[0]
        coeff = term[1]
        
        spins = -coeff
        for qidx, pauli in qops.items():
            #print(f"        {qidx} : {pauli}")
            if pauli == 'Z':
                spins *= spin.z(qidx)
            elif pauli == 'Y':
                spins *= spin.y(qidx)
            elif pauli == 'X':
                spins *= spin.x(qidx)
            elif pauli == '':
                spins *= spin.i(qidx)
                
        if len(qops) > 0:       
            spin_op += spins
    """   
    ham_type = "heisenberg"  # Choose between "heisenberg" and "tfim"
    ham_type = "tfim"  # Choose between "heisenberg" and "tfim"
    
    Jx, Jy, Jz = 1.0, 1.0, 1.0  # Coupling coefficients for Heisenberg Hamiltonian
    h_field = 2.0  # Transverse field strength for TFIM
    
    #K = 100  # Number of Trotter steps
    #t = np.pi  # Total evolution time
    dt = t / K  # Time step size

    # Optimized XXYYZZ exponentiation. Works only for Heisenberg Hamiltonian
    _use_XXYYZZ_gate = False
    if _use_XXYYZZ_gate == True and ham_type == "tfim":
        print ("XXYYZZ exponentiation works only for Heisenberg")
        sys.exit(0)
    """
    
    """
    # Create Hamiltonian
    if ham_type == "heisenberg":
        # Initialize field for Heisenberg Hamiltonian
        h_x = np.ones(n_spins)
        h_y = np.ones(n_spins)
        h_z = np.ones(n_spins)
        hamiltonian = create_hamiltonian_heisenberg(n_spins, Jx, Jy, Jz, h_x, h_y,h_z)
        
    elif ham_type == "tfim":
        hamiltonian = create_hamiltonian_tfim(n_spins, h_field)
        
    else:
        raise ValueError("Invalid Hamiltonian type. Choose 'heisenberg' or 'tfim'.")

    # Extract coefficients and words
    coefficients = extractCoefficients(hamiltonian)
    words = extractWords(hamiltonian)
    
    
    cudaq.set_target("nvidia")

    # Initialize and save the initial state
    print ("Initialize state")
    initial_state = cudaq.get_state(get_initial_state, n_spins)
    state = initial_state

    #result = cudaq.sample(kernel, parameters, shots_count=num_shots)

    # Apply single Trotter step
    state = cudaq.get_state(trotter_step, state, dt, Jx, Jy, Jz, h_x, h_y, h_z, _use_XXYYZZ_gate, coefficients, words)
    """
    
    # Extract coefficients and words
    """
    coefficients = extractCoefficients(spin_op)
    words = extractWords(spin_op)
    print(coefficients)
    print(words)
    """
   
    return spin_op
    #return hamiltonian

############### Hamiltonian Simulation Kernel Definition
 
@cudaq.kernel           
def trotter_step (

@cudaq.kernel
def trotter_step_2(state: cudaq.State,
                dt: float, 
                coefficients: List[complex],
                words: List[cudaq.pauli_word]
             ) -> None:
             
    """Perform single Trotter step"""
    qubits = cudaq.qvector(state)
    n_spins = len(qubits)
   
    for i in range(len(coefficients)):
        exp_pauli(coefficients[i].real * dt, qubits, words[i])


@cudaq.kernel           
def hamsim_kernel (num_qubits: int, K: int = 5, t: float = 1.0):

    # Parameters
    n_spins = num_qubits  # Number of spins in the chain
    
 
#####################################################           
 
############### Phase Esimation Circuit Definition

# Inverse Quantum Fourier Transform
@cudaq.kernel
def iqft(register: cudaq.qview):
    M_PI = 3.1415926536
    
    input_size = register.size()
     
    # use this as a barrier when drawing circuit; comment out otherwise
    for i in range(input_size / 2):
        swap(register[i], register[input_size - i - 1])
        swap(register[i], register[input_size - i - 1])
            
    # Generate multiple groups of diminishing angle CRZs and H gate
    for i_qubit in range(input_size):
        ri_qubit = input_size - i_qubit - 1             # map to cudaq qubits
        
        # precede with an H gate (applied to all qubits)
        h(register[ri_qubit])
        
        # number of controlled Z rotations to perform at this level
        num_crzs = input_size - i_qubit - 1
        
        # if not the highest order qubit, add multiple controlled RZs of decreasing angle
        if i_qubit < input_size - 1:   
            for j in range(0, num_crzs):
                #divisor = 2 ** (j + 1)     # DEVNOTE: #1663 This does not work on Quantum Cloud

                X = j + 1                   # WORKAROUND
                divisor = 1
                for i in range(X):
                    divisor *= 2
                
                r1.ctrl( -M_PI / divisor , register[ri_qubit], register[ri_qubit - j - 1])       
    
@cudaq.kernel           
#def pe_kernel (num_qubits: int, theta: float, do_uopt: bool):
def pe_kernel (num_qubits: int, theta: float):
    M_PI = 3.1415926536
    
    init_phase = 2 * M_PI * theta
    do_uopt = True
    
    # size of input is one less than available qubits
    input_size = num_qubits - 1
        
    # Allocate on less than the specified number of qubits for phase register
    counting_qubits = cudaq.qvector(input_size)
    
    # Allocate an extra qubit for the state register (can expand this later)
    state_register = cudaq.qubit()

    # Prepare the auxillary qubit.
    x(state_register)

    # Place the phase register in a superposition state.
    h(counting_qubits)

    # Perform `ctrl-U^j`        (DEVNOTE: need to pass in as lambda later)
    for i in range(input_size):
        ii_qubit = input_size - i - 1
        i_qubit = i
        
        # optimize by collapsing all phase gates at one level to single gate
        if do_uopt:
            exp_phase = init_phase * (2 ** i_qubit) 
            r1.ctrl(exp_phase, counting_qubits[i_qubit], state_register);
        else:
            for j in range(2 ** i_qubit):
                r1.ctrl(init_phase, counting_qubits[i_qubit], state_register);
    
    # Apply inverse quantum Fourier transform
    iqft(counting_qubits)

    # Apply measurement gates to just the `qubits`
    # (excludes the auxillary qubit).
    mz(counting_qubits)
 
 
######################################################################
# EXTERNAL API FUNCTIONS
   
############### Hamiltonian Circuit Definition

def HamiltonianSimulation(
        num_qubits: int = 0,
        ham_op: Union[
            List[Tuple[str, complex]],
            List[Tuple[Dict[int, str], complex]]
        ] = None,
        K: int = 5, t: float = 1.0,
        init_state = None,
        method: int = 1,
        use_inverse_flag: bool = False,
        random_pauli_flag = False,
        random_init_flag = False,
        append_measurements = True,
    ) -> Tuple:
    

    spin_op = convert_to_spin_op(num_qubits, ham_op)
    print(f"... spin_op = {spin_op}")
        
    theta = 1.0
    
    qc = [pe_kernel, [num_qubits, theta]]
    
    global QC_
    if num_qubits <= 6:
        QC_ = qc

    return qc, None
    

############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw(hamiltonian: str = "hamlib", method: int = 1):
    print("Sample Circuit:");
    if QC_ != None:
        print(cudaq.draw(QC_[0], *QC_[1]))
    else:
        print("  ... too large!")
    