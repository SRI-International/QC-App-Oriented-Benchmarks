'''
Phase Estimation Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import cudaq

# saved circuits for display
QC_ = None
Uf_ = None

############### BV Circuit Definition
'''
@cudaq.kernel
def oracle(register: cudaq.qview, auxillary_qubit: cudaq.qubit,
           hidden_bits: List[int]):
    input_size = len(hidden_bits)
    for index, bit in enumerate(hidden_bits):
        if bit == 1:
            # apply a `cx` gate with the current qubit as
            # the control and the auxillary qubit as the target.
            x.ctrl(register[input_size - index - 1], auxillary_qubit)
'''
@cudaq.kernel           
def pe_kernel (num_qubits: int, theta: float):
    
    # size of input is one less than available qubits
    input_size = num_qubits - 1
        
    # Allocate the specified number of qubits - this
    # corresponds to the length of the hidden bitstring.
    qubits = cudaq.qvector(input_size)
    
    # Allocate an extra auxillary qubit.
    auxillary_qubit = cudaq.qubit()

    # Prepare the auxillary qubit.
    h(auxillary_qubit)
    z(auxillary_qubit)

    # Place the rest of the register in a superposition state.
    h(qubits)

    # Query the oracle.
    #oracle(qubits, auxillary_qubit, hidden_bits)

    # Apply another set of Hadamards to the register.
    h(qubits)
    
    # DEVNOTE: compare to Qiskit version - do we need to flip the aux bit back? and measure all?

    # Apply measurement gates to just the `qubits`
    # (excludes the auxillary qubit).
    mz(qubits)
        
 
def PhaseEstimation (num_qubits: int, theta: float):

    qc = [pe_kernel, [num_qubits, theta]]
    
    global QC_
    if num_qubits <= 9:
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
    