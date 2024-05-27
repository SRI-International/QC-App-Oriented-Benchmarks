
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
def bv_kernel (num_qubits: int, hidden_bits: List[int], method: int = 1):
    
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
        '''
        # allocate qubits
        qr = QuantumRegister(2); cr = ClassicalRegister(input_size); qc = QuantumCircuit(qr, cr, name="main")

        # put ancilla in |-> state
        qc.x(qr[1])
        qc.h(qr[1])

        qc.barrier()

        # perform CX for each qubit that matches a bit in secret string
        s = ('{0:0' + str(input_size) + 'b}').format(secret_int)
        for i in range(input_size):
            if s[input_size - 1 - i] == '1':
                qc.h(qr[0])
                qc.cx(qr[0], qr[1])
                qc.h(qr[0])
            qc.measure(qr[0], cr[i])

            # Perform num_resets reset operations
            qc.reset([0]*num_resets)
        '''
        
        # put ancilla in |-> state
        x(auxillary_qubit)
        h(auxillary_qubit)
        '''
        # perform CX for each qubit that matches a bit in secret integer's bits
        for i_qubit in range(input_size):
            #if hidden_bits[input_size - 1 - i_qubit] == 1:             # DEVNOTE:
            if hidden_bits[i_qubit] == 1:
                #qc.cx(qr[i_qubit], qr[input_size])
                qc.h(qr[0])
                qc.cx(qr[0], qr[1])
                qc.h(qr[0])
            qc.measure(qr[0], cr[i_qubit])
            
            # Perform num_resets reset operations
            qc.reset([0]*num_resets)
        ''' 
        #ba = [0] * 4
        ba = [0,1,1,0]
        for index, bit in enumerate(hidden_bits):
            if bit == 1:
                h(qubits)
                cx(qubits, auxillary_qubit)
                h(qubits)
                
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
                
            #ba[index] = b
            #hidden_bits[index] = b
 
def BersteinVazirani (num_qubits: int, hidden_bits: List[int], method: int = 1):

    qc = [bv_kernel, [num_qubits, hidden_bits, method]]
    
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
    
     