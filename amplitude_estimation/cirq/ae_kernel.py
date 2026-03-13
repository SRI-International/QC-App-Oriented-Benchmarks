'''
Amplitude Estimation Benchmark Program - Cirq Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import copy
import math
import numpy as np
import cirq

import cirq_utils


############### Inverse QFT Gate (inlined from QFT benchmark)

def inv_qft_gate(input_size):
    """Build inverse QFT gate for cirq."""
    qr = [cirq.GridQubit(i, 0) for i in range(input_size)]
    qc = cirq.Circuit()

    # Generate multiple groups of diminishing angle CRZs and H gate
    for i_qubit in reversed(range(0, input_size)):
        # start laying out gates from highest order qubit
        hidx = input_size - i_qubit - 1

        # precede with an H gate
        qc.append(cirq.H(qr[hidx]))

        # if not the highest order qubit, add multiple controlled RZs of decreasing angle
        if hidx < input_size - 1:
            num_crzs = i_qubit
            for j in reversed(range(0, num_crzs)):
                divisor = 2 ** (num_crzs - j)
                qc.append(cirq.CZ(qr[hidx], qr[input_size - j - 1])**(-1.0/divisor))

    # Convert to gate
    return cirq_utils.to_gate(num_qubits=input_size, circ=qc, name="inv_qft")


# saved subcircuits for printing
A_ = None
Q_ = None
cQ_ = None
QC_ = None
QFTI_ = None


############### Circuit Definition

def AmplitudeEstimation(num_state_qubits, num_counting_qubits, a, psi_zero=None, psi_one=None):

    qr_state = cirq.GridQubit.rect(1, num_state_qubits+1, 0)
    qr_counting = cirq.GridQubit.rect(1, num_counting_qubits, 1)
    qc = cirq.Circuit()

    num_qubits = num_state_qubits + 1 + num_counting_qubits

    # create the Amplitude Generator circuit
    A_circuit = A_gen(num_state_qubits, a, psi_zero, psi_one)
    A = cirq_utils.to_gate(num_state_qubits+1, A_circuit, name="A")

    # create the Quantum Operator circuit and a controlled version of it
    cQ, Q = Ctrl_Q(num_state_qubits, A_circuit)

    # save small example subcircuits for visualization
    global A_, Q_, cQ_
    if (cQ_ and Q_) == None or num_state_qubits <= 6:
        if num_state_qubits < 9: cQ_ = cQ; Q_ = Q; A_ = A

    # Prepare state from A, and counting qubits with H transform
    qc.append(A.on(*qr_state))
    for i in range(num_counting_qubits):
        qc.append(cirq.H.on(qr_counting[i]))

    repeat = 1
    for j in reversed(range(num_counting_qubits)):
        for _ in range(repeat):
            qc.append(cQ.on(qr_counting[j], *qr_state))
        repeat *= 2

    # inverse quantum Fourier transform only on counting qubits
    QFT_inv_gate = inv_qft_gate(num_counting_qubits)
    qc.append(QFT_inv_gate.on(*qr_counting))

    # measure counting qubits
    qc.append(cirq.measure(*[qr_counting[i_qubit] for i_qubit in range(num_counting_qubits)], key='result'))

    # save smaller circuit example for display
    global QC_, QFTI_
    if QC_ == None or num_qubits <= 5:
        if num_qubits < 9: QC_ = qc
    if QFTI_ == None or num_qubits <= 5:
        if num_qubits < 9: QFTI_ = QFT_inv_gate

    return qc


def A_gen(num_state_qubits, a, psi_zero=None, psi_one=None):
    """Construct A operator."""
    if psi_zero == None:
        psi_zero = '0' * num_state_qubits
    if psi_one == None:
        psi_one = '1' * num_state_qubits

    theta = 2 * np.arcsin(np.sqrt(a))
    qr_A = cirq.GridQubit.rect(1, num_state_qubits+1, 0)
    qc_A = cirq.Circuit()

    # takes state to |0>_{n} (sqrt(1-a) |0> + sqrt(a) |1>)
    qc_A.append(cirq.ry(theta).on(qr_A[num_state_qubits]))

    # takes state to sqrt(1-a) |psi_0>|0> + sqrt(a) |0>_{n}|1>
    qc_A.append(cirq.X(qr_A[num_state_qubits]))
    for i in range(num_state_qubits):
        if psi_zero[i] == '1':
            qc_A.append(cirq.CNOT(qr_A[num_state_qubits], qr_A[i]))
    qc_A.append(cirq.X(qr_A[num_state_qubits]))

    # takes state to sqrt(1-a) |psi_0>|0> + sqrt(a) |psi_1>|1>
    for i in range(num_state_qubits):
        if psi_one[i] == '1':
            qc_A.append(cirq.CNOT(qr_A[num_state_qubits], qr_A[i]))

    return qc_A


def Ctrl_Q(num_state_qubits, A_circ):
    """Construct the grover-like operator and a controlled version of it."""
    qr_Q = cirq.GridQubit.rect(1, num_state_qubits+1, 0)
    qc_Q = cirq.Circuit()

    A_gate = cirq_utils.to_gate(num_state_qubits+1, A_circ, name="A")
    A_gate_inv = cirq.inverse(copy.copy(A_gate))

    # -S_chi
    qc_Q.append(cirq.X(qr_Q[num_state_qubits]))
    qc_Q.append(cirq.Z(qr_Q[num_state_qubits]))
    qc_Q.append(cirq.X(qr_Q[num_state_qubits]))

    # A_circ_inverse
    qc_Q.append(A_gate_inv.on(*qr_Q))

    # S_0
    for i in range(num_state_qubits+1):
        qc_Q.append(cirq.X.on(qr_Q[i]))
    qc_Q.append(cirq.H(qr_Q[num_state_qubits]))

    qc_Q.append(cirq.X.controlled(num_controls=num_state_qubits).on(*qr_Q))

    qc_Q.append(cirq.H(qr_Q[num_state_qubits]))
    for i in range(num_state_qubits+1):
        qc_Q.append(cirq.X.on(qr_Q[i]))

    # A_circ
    qc_Q.append(A_gate.on(*qr_Q))

    # Create a gate out of the Q operator
    Q_ = cirq_utils.to_gate(num_qubits=num_state_qubits+1, circ=qc_Q, name="Q")

    # and also a controlled version of it
    Ctrl_Q_ = cirq.ops.ControlledGate(Q_, num_controls=1)

    return Ctrl_Q_, Q_


############### Kernel Draw

def kernel_draw():
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")

    if cQ_ != None and Q_ != None and A_ != None:
        # Create registers for printing
        num_state_qubits = Q_.num_qubits - 1 if hasattr(Q_, 'num_qubits') else 2
        qr_state = cirq.GridQubit.rect(1, num_state_qubits+1, 0)
        qr_control = cirq.GridQubit.rect(1, 1, 1)

        print("\nControlled Quantum Operator 'cQ' =")
        print(cirq.Circuit(cQ_.on(qr_control[0], *qr_state)) if cQ_ != None else "  ... too large!")
        print("\nQuantum Operator 'Q' =")
        print(cirq.Circuit(cirq.decompose(Q_.on(*qr_state))) if Q_ != None else "  ... too large!")
        print("\nAmplitude Generator 'A' =")
        print(cirq.Circuit(cirq.decompose(A_.on(*qr_state))) if A_ != None else "  ... too large!")

    if QFTI_ != None:
        qr_qft = cirq.GridQubit.rect(1, QFTI_.num_qubits, 0)
        print("\nInverse QFT Circuit =")
        print(cirq.Circuit(cirq.decompose(QFTI_.on(*qr_qft))))
    else:
        print("\nInverse QFT Circuit =  ... too large!")
