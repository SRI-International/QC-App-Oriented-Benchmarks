'''
Phase Estimation Benchmark Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# saved subcircuits circuits for printing
QC_ = None
QFTI_ = None
U_ = None

############### Circuit Definition

def PhaseEstimation(num_qubits, theta):
    
    qr = QuantumRegister(num_qubits)
    
    num_counting_qubits = num_qubits - 1 # only 1 state qubit
    
    cr = ClassicalRegister(num_counting_qubits)
    qc = QuantumCircuit(qr, cr, name=f"qpe-{num_qubits}-{theta}")

    # initialize counting qubits in superposition
    for i in range(num_counting_qubits):
        qc.h(qr[i])

    # change to |1> in state qubit, so phase will be applied by cphase gate
    qc.x(num_counting_qubits)

    qc.barrier()

    repeat = 1
    for j in reversed(range(num_counting_qubits)):
        # controlled operation: adds phase exp(i*2*pi*theta*repeat) to the state |1>
        #                       does nothing to state |0>
        cp, _ = CPhase(2*math.pi*theta, repeat)
        qc.append(cp, [j, num_counting_qubits])
        repeat *= 2

    #Define global U operator as the phase operator
    _, U = CPhase(2*math.pi*theta, 1)

    qc.barrier()
    
    # inverse quantum Fourier transform only on counting qubits
    qc.append(inv_qft_gate(num_counting_qubits), qr[:num_counting_qubits])
    
    qc.barrier()
    
    # measure counting qubits
    qc.measure([qr[m] for m in range(num_counting_qubits)], list(range(num_counting_qubits)))

    # save smaller circuit example for display
    global QC_, U_, QFTI_
    if QC_ == None or num_qubits <= 5:
        if num_qubits < 9: QC_ = qc
    if U_ == None or num_qubits <= 5:
        if num_qubits < 9: U_ = U
    if QFTI_ == None or num_qubits <= 5:
        if num_qubits < 9: QFTI_ = inv_qft_gate(num_counting_qubits)
        
    # collapse the 3 sub-circuit levels used in this benchmark (for qiskit)
    qc2 = qc.decompose().decompose().decompose()
            
    # return a handle on the circuit
    return qc2

#Construct the phase gates and include matching gate representation as readme circuit
def CPhase(angle, exponent):

    qc = QuantumCircuit(1, name=f"U^{exponent}")
    qc.p(angle*exponent, 0)
    phase_gate = qc.to_gate().control(1)

    return phase_gate, qc

############### Inverse QFT Circuit

def inv_qft_gate(input_size):
    #global QFTI_, num_gates, depth
    global QFTI_
    qr = QuantumRegister(input_size); qc = QuantumCircuit(qr, name="inv_qft")
    
    # Generate multiple groups of diminishing angle CRZs and H gate
    for i_qubit in reversed(range(0, input_size)):
    
        # start laying out gates from highest order qubit (the hidx)
        hidx = input_size - i_qubit - 1
        
        # precede with an H gate (applied to all qubits)
        qc.h(qr[hidx])
        
        # if not the highest order qubit, add multiple controlled RZs of decreasing angle
        if hidx < input_size - 1:   
            num_crzs = i_qubit
            for j in reversed(range(0, num_crzs)):
                divisor = 2 ** (num_crzs - j)
                qc.crz( -math.pi / divisor , qr[hidx], qr[input_size - j - 1])
            
        qc.barrier()  
    
    if QFTI_ == None or input_size <= 5:
        if input_size < 9: QFTI_= qc
            
    # return a handle on the circuit
    return qc

############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
  
    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    print("\nPhase Operator 'U' = "); print(U_ if U_ != None else "  ... too large!")
    print("\nInverse QFT Circuit ="); print(QFTI_ if QFTI_ != None else "  ... too large!")
    
    