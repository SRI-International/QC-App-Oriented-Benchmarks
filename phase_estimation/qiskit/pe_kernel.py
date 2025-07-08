'''
Phase Estimation Benchmark Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# saved subcircuits circuits for printing
QC_ = None
QFTDI_ = None
QFTI_ = None
U_ = None

############### Circuit Definition

def PhaseEstimation(num_qubits, theta, use_midcircuit_measurement):
    
    qr = QuantumRegister(num_qubits)
    
    num_counting_qubits = num_qubits - 1 # only 1 state qubit
    
    cr = ClassicalRegister(num_counting_qubits, name = "c0")
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
    
    # Dynamic circuits can only be added using "compose" because they are not unitary.
    # The "append" method requires the added circuit to be unitary, which dynamic circuits are not.

    if use_midcircuit_measurement:
        dynamic_inv_qft = dyn_inv_qft_gate(num_counting_qubits)
        qc.compose(dynamic_inv_qft, qubits=qr[:num_counting_qubits], clbits = cr[:num_counting_qubits], inplace=True)
    else:
        static_inv_qft = inv_qft_gate(num_counting_qubits)
        qc.append(static_inv_qft, qr[:num_counting_qubits])

    qc.barrier()
    
    # measure counting qubits
    qc.measure([qr[m] for m in range(num_counting_qubits)], list(range(num_counting_qubits)))

    # save smaller circuit example for display
    global QC_, U_, QFTI_, QFTDI_
    if QC_ == None or num_qubits <= 5:
        if num_qubits < 9: QC_ = qc
    if U_ == None or num_qubits <= 5:
        if num_qubits < 9: U_ = U
    
    if use_midcircuit_measurement:
        if QFTDI_ == None or num_qubits <= 5:
            if num_qubits < 9: QFTDI_ = dynamic_inv_qft
    else:
        if QFTI_ == None or num_qubits <= 5:
            if num_qubits < 9: QFTI_ = static_inv_qft
    
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
    for i_qubit in range(input_size):
        
        # precede with an H gate (applied to all qubits)
        qc.h(qr[i_qubit])
        
        # number of controlled Z rotations to perform at this level
        num_crzs = input_size - i_qubit - 1
        
        # if not the highest order qubit, add multiple controlled RZs of decreasing angle
        if i_qubit < input_size - 1:   
            for j in range(0, num_crzs):
                divisor = 2 ** (j + 1)
                qc.crz( -math.pi / divisor , qr[i_qubit], qr[i_qubit + j + 1])
            
        qc.barrier()
    
    if QFTI_ == None or input_size <= 5:
        if input_size < 9: QFTI_= qc
            
    # return a handle on the circuit
    return qc

############### Dynamic Inverse QFT Circuit

def dyn_inv_qft_gate(input_size):
   
    global QFTDI_, num_gates, depth
    qr = QuantumRegister(input_size, name="q_dyn_inv")
    cr = ClassicalRegister(input_size, name = "c0")
    qc = QuantumCircuit(qr, cr, name="dyn_inv_qft")

    # mirror the static inv-QFT loop order, but with mid-circuit feed-forward
    for i_qubit in reversed(range(input_size)):
        hidx = input_size - 1 - i_qubit

        # H on the “hidx” wire
        qc.h(qr[hidx])
        qc.barrier()

        # measure for feed-forward
        qc.measure(qr[hidx], cr[hidx])
        qc.barrier()

        # if measured == 1, apply RZ(-θ) on each target
        if hidx < input_size - 1:
            for j in reversed(range(i_qubit)):
                θ = math.pi / (2 ** (i_qubit - j))
                with qc.if_test((cr[hidx], 1)):
                    qc.rz(-θ, qr[input_size - 1 - j])
        qc.barrier()

    # cache a small-size example for printing
    if QFTDI_ is None and input_size <= 5:
        QFTDI_ = qc

    return qc

############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
  
    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    print("\nPhase Operator 'U' = "); print(U_ if U_ != None else "  ... too large!")
    if QFTDI_ != None:
        print("\nDynamic Inverse QFT Circuit ="); print(QFTDI_ if QFTDI_ != None else "  ... too large!")
    else:
        print("\nInverse QFT Circuit ="); print(QFTI_ if QFTI_ != None else "  ... too large!")

    
    
    