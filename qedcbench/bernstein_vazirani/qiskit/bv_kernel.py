'''
Bernstein-Vazirani Benchmark Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from typing import List

# saved circuits for display
QC_ = None
Uf_ = None

# Variable for number of resets to perform after mid circuit measurements
num_resets = 1

############### BV Circuit Definition
  
def create_oracle(num_qubits: int, input_size: int, hidden_bits: List[int]):

    # Initialize first n qubits and single ancilla qubit
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr, name="Uf")

    # perform CX for each qubit that matches a bit in secret integer's bits
    for i_qubit in range(input_size):
        if hidden_bits[i_qubit] == 1:
            qc.cx(qr[i_qubit], qr[input_size])
    return qc

def BersteinVazirani (num_qubits: int, secret_int: int, hidden_bits: List[int], method: int = 1):
    
    # size of input is one less than available qubits
    input_size = num_qubits - 1

    # method 1 is the traditional algorithm with oracle consuming all but one qubit
    if method == 1:
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
        Uf = create_oracle(num_qubits, input_size, hidden_bits)
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
    
    # method 2 uses mid-circuit measurement to create circuits with only 2 qubits    
    elif method == 2:
        # allocate qubits
        qr = QuantumRegister(2); cr = ClassicalRegister(input_size);
        qc = QuantumCircuit(qr, cr, name=f"bv({method})-{num_qubits}-{secret_int}")

        # put ancilla in |-> state
        qc.x(qr[1])
        qc.h(qr[1])

        qc.barrier()

        # there is no oracle to save in this method
        Uf = None
  
        # perform CX for each qubit that matches a bit in secret integer's bits
        for i_qubit in range(input_size):
            if hidden_bits[i_qubit] == 1:
                qc.h(qr[0])
                qc.cx(qr[0], qr[1])
                qc.h(qr[0])
            qc.measure(qr[0], cr[i_qubit])
            
            # Perform num_resets reset operations
            qc.reset([0]*num_resets)           
            
    # save circuit examples for display
    global QC_, Uf_
    if QC_ == None or num_qubits <= 6:
        if num_qubits < 9: QC_ = qc
    if Uf_ == None or num_qubits <= 6:
        if num_qubits < 9: Uf_ = Uf
            
    # collapse the sub-circuit levels used in this benchmark (for qiskit)
    qc2 = qc.decompose()
            
    # return a handle on the circuit
    return qc2

############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
    print("Sample Circuit:");
    print(QC_ if QC_ != None else "  ... too large!")
    #if method == 1:
    if Uf_ != None:
        print("\nQuantum Oracle 'Uf' =");
        print(Uf_ if Uf_ != None else " ... too large!")
    