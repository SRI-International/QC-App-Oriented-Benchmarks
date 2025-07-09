'''
Bernstein-Vazirani Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

# DEVNOTE: Method 2 of this benchmark does not work correctly due to limitations in the ability
# for the Python version of cudaq to collect and return an array of measured values (Issue #????).
# Only the final measurements are returned, meaning the fidelity is not determined correctly.
import cudaq

from typing import List

# saved circuits for display
QC_ = None
Uf_ = None

############### BV Circuit Definition

@cudaq.kernel
def oracle(register: cudaq.qview, auxillary_qubit: cudaq.qubit,
           hidden_bits: List[int]):
    input_size = len(hidden_bits)
    for index, bit in enumerate(hidden_bits):
        if bit == 1:
            # apply a `cx` gate with the current qubit as
            # the control and the auxillary qubit as the target.
            x.ctrl(register[input_size - index - 1], auxillary_qubit)

@cudaq.kernel           
def bv_kernel (num_qubits: int, secret_int: int, hidden_bits: List[int], method: int = 1):
    
    # size of input is one less than available qubits
    input_size = num_qubits - 1
    
    # for method 2, we only use a single qubit for primary register
    if method == 2:
        input_size = 1
        
    # Allocate the specified number of qubits - this
    # corresponds to the length of the hidden bitstring.
    qubits = cudaq.qvector(input_size)
    
    # Allocate an extra auxillary qubit.
    auxillary_qubit = cudaq.qubit()
    
    # method 1 is the traditional algorithm with oracle consuming all but one qubit
    if method == 1:

        # Prepare the auxillary qubit.
        h(auxillary_qubit)
        z(auxillary_qubit)

        # Place the rest of the register in a superposition state.
        h(qubits)

        # Query the oracle.
        oracle(qubits, auxillary_qubit, hidden_bits)

        # Apply another set of Hadamards to the register.
        h(qubits)
        
        # DEVNOTE: compare to Qiskit version - do we need to flip the aux bit back? and measure all?

        # Apply measurement gates to just the `qubits`
        # (excludes the auxillary qubit).
        mz(qubits)
        
    # method 2 uses mid-circuit measurement to create circuits with only 2 qubits
    elif method == 2:
        
        # put ancilla in |-> state
        x(auxillary_qubit)
        h(auxillary_qubit)

        # perform CX for each qubit that matches a bit in secret string
        # DEVNOTE: the commented code below is an attempt to find a way to capture the mid-crcuit measurements
        # but it does not work correctly.  CUDA Q does not seem to permit saving measures to an array
        #ba = [0] * 4
        ba = [0,1,1,0]
        for index, bit in enumerate(hidden_bits):
            if bit == 1:
                h(qubits)
                cx(qubits, auxillary_qubit)
                h(qubits)
            
            #b = mz(qubits)
            
            if index == 0:
                b0 = mz(qubits)
            elif index == 1:
                b1 = mz(qubits)
            elif index == 2:
                b2 = mz(qubits)
            elif index == 3:
                b3 = mz(qubits)
            else:
                b4 = mz(qubits)
                
            #b5 = 1
            #b5 = b0
            
            #ba[index] = mz(qubits)
            #hidden_bits[index] = b
 
def BersteinVazirani (num_qubits: int, secret_int: int, hidden_bits: List[int], method: int = 1):

    qc = [bv_kernel, [num_qubits, secret_int, hidden_bits, method]]
    
    global QC_
    if num_qubits <= 6:
        QC_ = qc

    return qc

############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
    print("Sample Circuit:");
    if QC_ != None:
        print(cudaq.draw(QC_[0], *QC_[1]))
    else:
        print("  ... too large!")
    
     