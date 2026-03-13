'''
Monte Carlo Sampling Benchmark Program - Cirq Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import copy
import functools
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.polynomial import polyfit
import cirq

import cirq_utils

import mc_utils


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

def MonteCarloSampling(target_dist, f, num_state_qubits, num_counting_qubits, epsilon=0.05, degree=2, method=2):

    A_qr = [cirq.GridQubit(i, 0) for i in range(num_state_qubits+1)]
    A = cirq.Circuit()

    num_qubits = num_state_qubits + 1 + num_counting_qubits

    # method 1 takes in the abitrary function f and arbitrary dist
    if method == 1:
        state_prep(A, A_qr, target_dist, num_state_qubits)
        f_on_objective(A, A_qr, f, epsilon=epsilon, degree=degree)
    # method 2 chooses to have lower circuit depth by choosing specific f and dist
    elif method == 2:
        uniform_prep(A, A_qr, num_state_qubits)
        square_on_objective(A, A_qr)

    qc = AE_Subroutine(num_state_qubits, num_counting_qubits, A)

    # save smaller circuit example for display
    global QC_
    if QC_ == None or num_qubits <= 5:
        if num_qubits < 9: QC_ = qc

    return qc

###############

def f_on_objective(qc, qr, f, epsilon=0.05, degree=3):
    """
    Assume last qubit is the objective. Function f is evaluated on first n-1 qubits
    """
    num_state_qubits = len(qr) - 1
    c_star = (2*epsilon)**(1/(degree+1))

    f_ = functools.partial(f, num_state_qubits=num_state_qubits)
    zeta_ = functools.partial(mc_utils.zeta_from_f, func=f_, epsilon=epsilon, degree=degree, c=c_star)

    x_eval = np.linspace(0.0, 2**(num_state_qubits) - 1, num= degree+1)
    poly = Polynomial(polyfit(x_eval, zeta_(x_eval), degree))

    b_exp = mc_utils.binary_expansion(num_state_qubits, poly)

    for controls in b_exp.keys():
        theta = 2*b_exp[controls]
        controls = list(controls)
        if len(controls)==0:
            qc.append(cirq.ry(-theta).on(qr[num_state_qubits]))
        else:
            qc.append(cirq.ry(-theta).controlled(num_controls=len(controls)).on(*[qr[i] for i in controls]+[qr[num_state_qubits]]))

def square_on_objective(qc, qr):
    """
    Assume last qubit is the objective.
    Shifted square wave function: if x is even, f(x) = 0; if x i s odd, f(x) = 1
    """
    num_state_qubits = len(qr) - 1
    for control in range(num_state_qubits):
        qc.append(cirq.CX.on(qr[control], qr[num_state_qubits]))

def state_prep(qc, qr, target_dist, num_state_qubits):
    """
    Use controlled Ry gates to construct the superposition Sum \sqrt{p_i} |i>
    """
    r_probs = mc_utils.region_probs(target_dist, num_state_qubits)
    regions = r_probs.keys()
    r_norm = {}

    for r in regions:
        num_controls = len(r) - 1
        super_key = r[:num_controls]

        if super_key=='':
            r_norm[super_key] = 1
        elif super_key == '1':
            r_norm[super_key] = r_probs[super_key]
            r_norm['0'] = 1-r_probs[super_key]
        else:
            try:
                r_norm[super_key] = r_probs[super_key]

            except KeyError:
                r_norm[super_key] = r_norm[super_key[:num_controls-1]] - r_probs[super_key[:num_controls-1] + '1']


        norm = r_norm[super_key]
        p = 0
        if norm != 0:
            p = r_probs[r] / norm
        theta = -2*np.arcsin(np.sqrt(p))

        if r == '1':
            qc.append(cirq.ry(theta).on(qr[num_state_qubits-1]))
        else:
            for k in range(num_controls):
                if r[k] == '0':
                    qc.append(cirq.X.on(qr[num_state_qubits-1 - k]))

            controls = [qr[num_state_qubits-1 - i] for i in range(num_controls)]
            qc.append(cirq.ry(theta).controlled(num_controls=num_controls).on(*controls+[qr[num_state_qubits-1-num_controls]]))

            for k in range(num_controls):
                if r[k] == '0':
                    qc.append(cirq.X.on(qr[num_state_qubits-1 - k]))

def uniform_prep(qc, qr, num_state_qubits):
    """
    Generates a uniform distribution over all states
    """
    for i in range(num_state_qubits):
        qc.append(cirq.H.on(qr[i]))

def AE_Subroutine(num_state_qubits, num_counting_qubits, A_circuit):
    qr_state = cirq.GridQubit.rect(1, num_state_qubits+1, 0)
    qr_counting = cirq.GridQubit.rect(1, num_counting_qubits, 1)
    qc_full = cirq.Circuit()

    A = cirq_utils.to_gate(num_state_qubits+1, A_circuit, name="A")
    cQ, Q = Ctrl_Q(num_state_qubits, A_circuit)

    # save small example subcircuits for visualization
    global A_, Q_, cQ_, QFTI_
    if cQ_ == None or num_state_qubits <= 6:
        if num_state_qubits < 9: cQ_ = cQ
    if (Q_ or A_) == None or num_state_qubits <= 3:
        if num_state_qubits < 5: A_ = A; Q_ = Q
    if QFTI_ == None or num_counting_qubits <= 3:
        if num_counting_qubits < 4: QFTI_ = inv_qft_gate(num_counting_qubits)

    # Prepare state from A, and counting qubits with H transform
    qc_full.append(A.on(*qr_state))
    for i in range(num_counting_qubits):
        qc_full.append(cirq.H.on(qr_counting[i]))

    repeat = 1
    for j in reversed(range(num_counting_qubits)):
        for _ in range(repeat):
            qc_full.append(cQ.on(*[qr_counting[j]]+qr_state))
        repeat *= 2

    # inverse quantum Fourier transform only on counting qubits
    QFT_inv_gate = inv_qft_gate(num_counting_qubits)
    qc_full.append(QFT_inv_gate.on(*qr_counting))

    qc_full.append(cirq.measure(*qr_counting, key='result'))

    return qc_full


###############################

# Construct the grover-like operator and a controlled version of it
def Ctrl_Q(num_state_qubits, A_circ):

    # index n is the objective qubit, and indexes 0 through n-1 are state qubits
    qr_Q = cirq.GridQubit.rect(1, num_state_qubits+1, 0)
    qc_Q = cirq.Circuit()

    A_gate = cirq_utils.to_gate(num_state_qubits+1, A_circ, name="A")
    A_gate_inv = cirq.inverse(copy.copy(A_gate))

    ### Each cycle in Q applies in order: -S_chi, A_circ_inverse, S_0, A_circ
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

    # and return both
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
