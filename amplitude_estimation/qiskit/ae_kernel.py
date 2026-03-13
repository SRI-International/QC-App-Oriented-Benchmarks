'''
Amplitude Estimation Benchmark Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import copy
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


############### Inverse QFT Gate (inlined from QFT kernel)

def inv_qft_gate(input_size):
    """Build inverse QFT circuit."""
    qr = QuantumRegister(input_size)
    qc = QuantumCircuit(qr, name="inv_qft")

    # Generate multiple groups of diminishing angle CRZs and H gate
    for i_qubit in reversed(range(0, input_size)):
        # start laying out gates from highest order qubit
        hidx = input_size - i_qubit - 1

        # precede with an H gate (applied to all qubits)
        qc.h(qr[hidx])

        # if not the highest order qubit, add multiple controlled RZs of decreasing angle
        if hidx < input_size - 1:
            num_crzs = i_qubit
            for j in reversed(range(0, num_crzs)):
                divisor = 2 ** (num_crzs - j)
                qc.crz(-math.pi / divisor, qr[hidx], qr[input_size - j - 1])

        qc.barrier()

    return qc


# saved subcircuits for printing
A_ = None
Q_ = None
cQ_ = None
QC_ = None
QFTI_ = None


############### Circuit Definition

def AmplitudeEstimation(num_state_qubits, num_counting_qubits, a, psi_zero=None, psi_one=None):

    num_qubits = num_state_qubits + 1 + num_counting_qubits

    qr_state = QuantumRegister(num_state_qubits+1)
    qr_counting = QuantumRegister(num_counting_qubits)
    cr = ClassicalRegister(num_counting_qubits)
    qc = QuantumCircuit(qr_counting, qr_state, cr, name=f"qae-{num_qubits}-{a}")

    # create the Amplitude Generator circuit
    A = A_gen(num_state_qubits, a, psi_zero, psi_one)

    # create the Quantum Operator circuit and a controlled version of it
    cQ, Q = Ctrl_Q(num_state_qubits, A)

    # save small example subcircuits for visualization
    global A_, Q_, cQ_, QFTI_
    if (cQ_ and Q_) == None or num_state_qubits <= 6:
        if num_state_qubits < 9: cQ_ = cQ; Q_ = Q; A_ = A
    if QFTI_ == None or num_qubits <= 5:
        if num_qubits < 9: QFTI_ = inv_qft_gate(num_counting_qubits)

    # Prepare state from A, and counting qubits with H transform
    qc.append(A, [qr_state[i] for i in range(num_state_qubits+1)])
    for i in range(num_counting_qubits):
        qc.h(qr_counting[i])

    repeat = 1
    for j in reversed(range(num_counting_qubits)):
        for _ in range(repeat):
            qc.append(cQ, [qr_counting[j]] + [qr_state[l] for l in range(num_state_qubits+1)])
        repeat *= 2

    qc.barrier()

    # inverse quantum Fourier transform only on counting qubits
    qc.append(inv_qft_gate(num_counting_qubits), qr_counting)

    qc.barrier()

    # measure counting qubits
    qc.measure([qr_counting[m] for m in range(num_counting_qubits)], list(range(num_counting_qubits)))

    # save smaller circuit example for display
    global QC_
    if QC_ == None or num_qubits <= 5:
        if num_qubits < 9: QC_ = qc
    return qc


# Construct A operator that takes |0>_{n+1} to sqrt(1-a) |psi_0>|0> + sqrt(a) |psi_1>|1>
def A_gen(num_state_qubits, a, psi_zero=None, psi_one=None):

    if psi_zero==None:
        psi_zero = '0'*num_state_qubits
    if psi_one==None:
        psi_one = '1'*num_state_qubits

    theta = 2 * np.arcsin(np.sqrt(a))
    # Let the objective be qubit index n; state is on qubits 0 through n-1
    qc_A = QuantumCircuit(num_state_qubits+1, name="A")

    # takes state to |0>_{n} (sqrt(1-a) |0> + sqrt(a) |1>)
    qc_A.ry(theta, num_state_qubits)

    # takes state to sqrt(1-a) |psi_0>|0> + sqrt(a) |0>_{n}|1>
    qc_A.x(num_state_qubits)
    for i in range(num_state_qubits):
        if psi_zero[i]=='1':
            qc_A.cx(num_state_qubits,i)
    qc_A.x(num_state_qubits)

    # takes state to sqrt(1-a) |psi_0>|0> + sqrt(a) |psi_1>|1>
    for i in range(num_state_qubits):
        if psi_one[i]=='1':
            qc_A.cx(num_state_qubits,i)

    return qc_A


# Construct the grover-like operator and a controlled version of it
def Ctrl_Q(num_state_qubits, A_circ):

    # index n is the objective qubit, and indexes 0 through n-1 are state qubits
    qc = QuantumCircuit(num_state_qubits+1, name="Q")

    temp_A = copy.copy(A_circ)
    A_gate = temp_A.to_gate()
    A_gate_inv = temp_A.inverse().to_gate()

    ### Each cycle in Q applies in order: -S_chi, A_circ_inverse, S_0, A_circ
    # -S_chi
    qc.x(num_state_qubits)
    qc.z(num_state_qubits)
    qc.x(num_state_qubits)

    # A_circ_inverse
    qc.append(A_gate_inv, [i for i in range(num_state_qubits+1)])

    # S_0
    for i in range(num_state_qubits+1):
        qc.x(i)
    qc.h(num_state_qubits)

    qc.mcx([x for x in range(num_state_qubits)], num_state_qubits)

    qc.h(num_state_qubits)
    for i in range(num_state_qubits+1):
        qc.x(i)

    # A_circ
    qc.append(A_gate, [i for i in range(num_state_qubits+1)])

    # Create a gate out of the Q operator
    qc.to_gate(label='Q')

    # and also a controlled version of it
    Ctrl_Q_ = qc.control(1)

    # and return both
    return Ctrl_Q_, qc


############### Kernel Draw

def kernel_draw():
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    print("\nControlled Quantum Operator 'cQ' ="); print(cQ_ if cQ_ != None else " ... too large!")
    print("\nQuantum Operator 'Q' ="); print(Q_ if Q_ != None else " ... too large!")
    print("\nAmplitude Generator 'A' ="); print(A_ if A_ != None else " ... too large!")
    print("\nInverse QFT Circuit ="); print(QFTI_ if QFTI_ != None else "  ... too large!")
