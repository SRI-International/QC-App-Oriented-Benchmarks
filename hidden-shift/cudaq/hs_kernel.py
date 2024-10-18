'''
Hidden Shift Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import cudaq

from typing import List

# saved circuits for display
QC_ = None
Uf_ = None

############### BV Circuit Definition
   
# Uf oracle where Uf|x> = f(x)|x>, f(x) = {-1,1}   
@cudaq.kernel
def Uf_oracle(qubits: cudaq.qview, num_qubits: int, hidden_bits: list[int]): 
            
    for i_qubit in range(num_qubits):
        if hidden_bits[num_qubits-1-i_qubit] == 1 :
            x(qubits[i_qubit])

    for i_qubit in range(0,num_qubits-1,2):
        cz(qubits[i_qubit], qubits[i_qubit+1])
        
    for i_qubit in range(num_qubits):
        if hidden_bits[num_qubits-1-i_qubit] == 1:
            x(qubits[i_qubit])

# Generate Ug oracle where Ug|x> = g(x)|x>, g(x) = f(x+s)            
@cudaq.kernel
def Ug_oracle(qubits: cudaq.qview, num_qubits: int): 
    
    for i_qubit in range(0,num_qubits-1,2):
        cz(qubits[i_qubit], qubits[i_qubit+1])
          

@cudaq.kernel           
def hs_kernel (num_qubits: int, secret_int: int, hidden_bits: List[int], method: int = 1):
        
    # Allocate the specified number of qubits - this
    # corresponds to the length of the hidden bitstring.
    qubits = cudaq.qvector(num_qubits)

    # Start with Hadamard on all input qubits
    h(qubits)

    # Query the oracle.
    #oracle(qubits, auxillary_qubit, hidden_bits)
    Uf_oracle(qubits, num_qubits, hidden_bits)

    # Apply another set of Hadamards to the qubits.
    h(qubits)

    Ug_oracle(qubits, num_qubits)
    
    # Apply another set of Hadamards to the qubits.
    h(qubits)
    
    # Measure all qubits
    mz(qubits)
        
 
def HiddenShift (num_qubits: int, secret_int: int, hidden_bits: List[int], method: int = 1):

    qc = [hs_kernel, [num_qubits, secret_int, hidden_bits, method]]
    
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
    
     