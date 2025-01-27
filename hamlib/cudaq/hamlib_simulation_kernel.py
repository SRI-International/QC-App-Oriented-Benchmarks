'''
HamLib Simulation Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''

from typing import Union, List, Tuple, Dict
import numpy as np

import cudaq
from cudaq import spin
    
# saved circuits for display
QC_ = None
Uf_ = None


######################################################################
# CUDAQ KERNEL FUNCTIONS

############### State Initialization

# Prepare the initial quantum state of the system. 

@cudaq.kernel
def get_initial_state(n_spins: int) -> None:
    """Create initial state |1010...>"""
    qubits = cudaq.qvector(n_spins)
    for i in range(0, n_spins, 2):
        x(qubits[i])
        
@cudaq.kernel
def append_initial_state(qubits: cudaq.qview, n_spins: int, init_phases: List[float]) -> None:
    """Create initial state |1010...>"""
    #qubits = cudaq.qvector(n_spins)
    #for i in range(0, n_spins, 2):
        #x(qubits[i])
    
    num_qubits = n_spins
    
    # Rotate each qubit into its initial state, 0 or 1
    for index, phase in enumerate(init_phases):
        if phase > 0.0:
            x(qubits[num_qubits - index - 1])
                        

############### Hamiltonian Simulation Kernel Definition

@cudaq.kernel           
def hamsim_kernel(
        num_qubits: int,
        init_phases: List[float],
        K: int = 5,
        t: float = 1.0,
        coefficients: List[complex] = None,
        words: List[cudaq.pauli_word] = None,
        append_measurements: bool = True
    ):
    
    # create the qubit vector
    qubits = cudaq.qvector(num_qubits)
    
    # add on the initial state
    append_initial_state(qubits, num_qubits, init_phases)
    
    # determin delta for one trotter step
    dt = t / K
    
    # Apply K Trotter steps
    for _ in range(K): 
        append_trotter_step(qubits, dt, coefficients, words)

@cudaq.kernel           
def hamsim_kernel_measured(
        num_qubits: int,
        init_phases: List[float],
        K: int = 5,
        t: float = 1.0,
        coefficients: List[complex] = None,
        words: List[cudaq.pauli_word] = None,
        append_measurements: bool = True
    ):
    
    # create the qubit vector
    qubits = cudaq.qvector(num_qubits)
    
    # add on the initial state
    append_initial_state(qubits, num_qubits, init_phases)
    
    # determin delta for one trotter step
    dt = t / K
    
    # Apply K Trotter steps
    for _ in range(K): 
        append_trotter_step(qubits, dt, coefficients, words)
 
    # Apply measurement gates to the `qubits`
    if append_measurements == True:
        mz(qubits)
        
# Append a trotter step defined by the time step dt and hamiltonian terms
@cudaq.kernel
def append_trotter_step(
            qubits: cudaq.qview,
            dt: float, 
            coefficients: List[complex],
            words: List[cudaq.pauli_word]
        ) -> None:
   
    for i in range(len(coefficients)):
        exp_pauli(coefficients[i].real * dt, qubits, words[i])   # this crashes jupyter kernel
 
#DEVNOTE: use this as a barrier when drawing circuit; comment out otherwise
@cudaq.kernel
def barrier(qubits: cudaq.qview, num_qubits: int):
	for i in range(num_qubits / 2):
		swap(qubits[i*2], qubits[i*2 + 1])
		swap(qubits[i*2], qubits[i*2 + 1])
  
  
######################################################################
# CUDAQ HELPER FUNCTIONS

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
 
   
    return spin_op

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
            append_measurements: bool = False,
        ) -> Tuple:
    
    # convert the Hamiltonian to cudaq format
    spin_op = convert_to_spin_op(num_qubits, ham_op)
    #print(f"... spin_op = {spin_op}")
    
    # Extract coefficients and words from the spin operator (for kernel)
    coefficients = extractCoefficients(spin_op)
    words = extractWords(spin_op)   
    #print(coefficients)
    #print(words)
    
    # convert the initial state from string form to vector of floats
    bitset = init_state_to_ivec(num_qubits, init_state)
    bitsetf = [float(v) for v in bitset]
    #print(f"... init_state_to_ivec, bitsetf = {bitsetf}")
     
    # Return a kernel with our without measurement gates
    # CUDAQ ISSUE: the mz() operation cannot be controlled by a flag
    if append_measurements:
        qc = [hamsim_kernel_measured, [num_qubits, bitsetf, K, t,
                coefficients, words, append_measurements]]
    else:
        qc = [hamsim_kernel, [num_qubits, bitsetf, K, t,
                coefficients, words, append_measurements]]
            
    global QC_
    if num_qubits <= 6:
        QC_ = qc

    return qc, None
    

############## Convert Input string to an integer vector

def init_state_to_ivec(n_spins: int, init_state: str):
    """
    Initialize the quantum state.
    
    Args:
        n_spins (int): Number of spins (qubits).
        init_state (str): The chosen initial state. By default applies the checkerboard state, but can also be set to "ghz", the GHZ state.

    Returns:
        int []: The initialized integer array.
    """
    
    init_state = init_state.strip().lower()
    
    # create an array to hold one integer per bit
    bitset = [0] * n_spins
    
    if init_state == "checkerboard" or init_state == "neele":
        # Checkerboard state, or "Neele" state
        for k in range(0, n_spins, 2):
            #qc.x([k]) 
            bitset[k] = 1
            
    #print(f"... init_state_to_ivec, bitset = {bitset}")
    
    return bitset
  
# Routine to convert the secret integer into an array of integers, each representing one bit
# DEVNOTE: do we need to convert to string, or can we just keep shifting?
def str_to_ivec(input_size: int, s_int: int):

    # convert the secret integer into a string so we can scan the characters
    s = ('{0:0' + str(input_size) + 'b}').format(s_int)
    
    # create an array to hold one integer per bit
    bitset = []
    
    # assign bits in reverse order of characters in string
    for i in range(input_size):

        if s[input_size - 1 - i] == '1':
            bitset.append(1)
        else:
            bitset.append(0)
    
    return bitset
    

############### Hamiltonian Expectation Functions

def get_expectation(
        qc: List = None, 
        num_qubits: int = 0,
        ham_op: Union[
                List[Tuple[str, complex]],
                List[Tuple[Dict[int, str], complex]]
            ] = None
        ):

    #print(f"... cudaq_kernel.get_expectation()")

    spin_op = convert_to_spin_op(num_qubits, ham_op)
    #print(f"... spin_op = {spin_op}")
    
    result = cudaq.observe(qc[0], spin_op, *qc[1])       

    return result.expectation()
    
    
############### HamLib Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw(hamiltonian: str = "hamlib", method: int = 1):
    print("Sample Circuit:");
    if QC_ != None:
        try:
            #when using exp_pauli, this is crashing with seg violation
            #print(cudaq.draw(QC_[0], *QC_[1]))
            print(f"WARNING: cudaq cannot draw kernels with Trotter steps")
            pass
        except:
            print(f"ERROR attemtping to draw the kernel")
        
    else:
        print("  ... too large!")
    