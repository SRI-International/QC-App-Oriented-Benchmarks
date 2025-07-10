'''
Hidden Shift Benchmark Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from typing import List

# saved circuits for display
QC_ = None
Uf_ = None
Ug_ = None

############### Circuit Definition

# Uf oracle where Uf|x> = f(x)|x>, f(x) = {-1,1}
def Uf_oracle(num_qubits, hidden_bits):
    # Initialize qubits qubits
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr, name="Uf")

    # Perform X on each qubit that matches a bit in secret string
    #s = ('{0:0'+str(num_qubits)+'b}').format(secret_int)
    for i_qubit in range(num_qubits):
        if hidden_bits[i_qubit]==1:
            qc.x(qr[i_qubit])

    for i_qubit in range(0,num_qubits-1,2):
        qc.cz(qr[i_qubit], qr[i_qubit+1])

    # Perform X on each qubit that matches a bit in secret string
    #s = ('{0:0'+str(num_qubits)+'b}').format(secret_int)
    for i_qubit in range(num_qubits):
        if hidden_bits[i_qubit]==1:
            qc.x(qr[i_qubit])

    return qc

# Generate Ug oracle where Ug|x> = g(x)|x>, g(x) = f(x+s)
def Ug_oracle(num_qubits):
    # Initialize first n qubits
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr, name="Ug")

    for i_qubit in range(0,num_qubits-1,2):
        qc.cz(qr[i_qubit], qr[i_qubit+1])

    return qc

def HiddenShift (num_qubits, secret_int, hidden_bits: List[int], method: int = 1):
    
    # allocate qubits
    qr = QuantumRegister(num_qubits); cr = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(qr, cr, name=f"hs-{num_qubits}-{secret_int}")
    
    # Start with Hadamard on all input qubits
    for i_qubit in range(num_qubits):
         qc.h(qr[i_qubit])

    qc.barrier()

    # Generate Uf oracle where Uf|x> = f(x)|x>, f(x) = {-1,1}
    Uf = Uf_oracle(num_qubits, hidden_bits)
    qc.append(Uf,qr)

    qc.barrier()
    
    # Again do Hadamard on all qubits
    for i_qubit in range(num_qubits):
         qc.h(qr[i_qubit])

    qc.barrier()

    # Generate Ug oracle where Ug|x> = g(x)|x>, g(x) = f(x+s)
    Ug = Ug_oracle(num_qubits)
    qc.append(Ug,qr)

    qc.barrier()

    # End with Hadamard on all qubits
    for i_qubit in range(num_qubits):
         qc.h(qr[i_qubit])
        
    qc.barrier()
    
    # measure all qubits
    qc.measure(qr, cr)

    # save smaller circuit example for display
    global QC_, Uf_, Ug_
    if QC_ == None or num_qubits <= 6:
        if num_qubits < 9: QC_ = qc
    if Uf_ == None or num_qubits <= 6:
        if num_qubits < 9: Uf_ = Uf
    if Ug_ == None or num_qubits <= 6:
        if num_qubits < 9: Ug_ = Ug
    
    # collapse the sub-circuit levels used in this benchmark (for qiskit)
    qc2 = qc.decompose()
            
    # return a handle on the circuit
    return qc2


############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
  
    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    print("\nQuantum Oracle 'Uf' ="); print(Uf_ if Uf_ != None else " ... too large!")
    print("\nQuantum Oracle 'Ug' ="); print(Ug_ if Ug_ != None else " ... too large!")
 
