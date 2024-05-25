
import cudaq

from typing import List

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

        # Apply measurement gates to just the `qubits`
        # (excludes the auxillary qubit).
        mz(qubits)
    
        '''
        # allocate qubits
        qr = QuantumRegister(num_qubits); cr = ClassicalRegister(input_size)
        qc = QuantumCircuit(qr, cr, name=f"bv({method})-{num_qubits}-{secret_int}")

        # put ancilla in |1> state
        qc.x(qr[input_size])

        # start with Hadamard on all qubits, including ancilla
        for i_qubit in range(num_qubits):
             qc.h(qr[i_qubit])

        qc.barrier()

        #generate Uf oracle
        Uf = create_oracle(num_qubits, input_size, secret_int)
        qc.append(Uf,qr)

        qc.barrier()

        # start with Hadamard on all qubits, including ancilla
        for i_qubit in range(num_qubits):
             qc.h(qr[i_qubit])

        # uncompute ancilla qubit, not necessary for algorithm
        qc.x(qr[input_size])

        qc.barrier()

        # measure all data qubits
        for i in range(input_size):
            qc.measure(i, i)
        '''
        
        '''
        global Uf_
        if Uf_ == None or num_qubits <= 6:
            if num_qubits < 9: Uf_ = Uf
        '''
        
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
        pass
        
    '''
    # save smaller circuit example for display
    global QC_
    if QC_ == None or num_qubits <= 6:
        if num_qubits < 9: QC_ = qc
    '''
 
def BersteinVazirani (num_qubits: int, hidden_bits: List[int], method: int = 1):

    return [bv_kernel, [num_qubits, hidden_bits, method]]
  