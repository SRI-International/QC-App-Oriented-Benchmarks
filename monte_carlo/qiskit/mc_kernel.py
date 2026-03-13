'''
Monte Carlo Sampling Benchmark Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import copy
import functools
import math
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.polynomial import polyfit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.standard_gates.ry import RYGate

import mc_utils


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
R_ = None
F_ = None
QFTI_ = None


############### Circuit Definition

def MonteCarloSampling(target_dist, f, num_state_qubits, num_counting_qubits, epsilon=0.05, degree=2, method=2):

    A_qr = QuantumRegister(num_state_qubits+1)
    A = QuantumCircuit(A_qr, name="A")

    num_qubits = num_state_qubits + 1 + num_counting_qubits

    # initialize R and F circuits
    R_qr = QuantumRegister(num_state_qubits+1)
    F_qr = QuantumRegister(num_state_qubits+1)
    R = QuantumCircuit(R_qr, name="R")
    F = QuantumCircuit(F_qr, name="F")

    # method 1 takes in the abitrary function f and arbitrary dist
    if method == 1:
        state_prep(R, R_qr, target_dist, num_state_qubits)
        f_on_objective(F, F_qr, f, epsilon=epsilon, degree=degree)
    # method 2 chooses to have lower circuit depth by choosing specific f and dist
    elif method == 2:
        uniform_prep(R, R_qr, num_state_qubits)
        square_on_objective(F, F_qr)

    # append R and F circuits to A
    A.append(R.to_gate(), A_qr)
    A.append(F.to_gate(), A_qr)

    # run AE subroutine given our A composed of R and F
    qc = AE_Subroutine(num_state_qubits, num_counting_qubits, A, method)

    # save smaller circuit example for display
    global QC_, R_, F_
    if QC_ == None or num_qubits <= 5:
        if num_qubits < 9: QC_ = qc
    if (R_ and F_) == None or num_state_qubits <= 3:
        if num_state_qubits < 5: R_ = R; F_ = F

    return qc

###############

def f_on_objective(qc, qr, f, epsilon=0.05, degree=2):
    """
    Assume last qubit is the objective. Function f is evaluated on first n-1 qubits
    """
    num_state_qubits = qc.num_qubits - 1
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
            qc.ry(-theta, qr[num_state_qubits])
        else:
            # define a MCRY gate:
            # this does the same thing as qc.mcry, but is clearer in the circuit printing
            MCRY = RYGate(-theta).control(len(controls))
            qc.append(MCRY, [*(qr[i] for i in controls), qr[num_state_qubits]])

def square_on_objective(qc, qr):
    """
    Assume last qubit is the objective.
    Shifted square wave function: if x is even, f(x) = 0; if x i s odd, f(x) = 1
    """
    num_state_qubits = qc.num_qubits - 1
    for control in range(num_state_qubits):
        qc.cx(control, num_state_qubits)

def state_prep(qc, qr, target_dist, num_state_qubits):
    r"""
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
        theta = 2*np.arcsin(np.sqrt(p))

        if r == '1':
            qc.ry(-theta, num_state_qubits-1)
        else:
            controls = [qr[num_state_qubits-1 - i] for i in range(num_controls)]

            # define a MCRY gate:
            # this does the same thing as qc.mcry, but is clearer in the circuit printing
            MCRY = RYGate(-theta).control(num_controls, ctrl_state=r[:-1])
            qc.append(MCRY, [*controls, qr[num_state_qubits-1 - num_controls]])

def uniform_prep(qc, qr, num_state_qubits):
    """
    Generates a uniform distribution over all states
    """
    for i in range(num_state_qubits):
        qc.h(i)

def AE_Subroutine(num_state_qubits, num_counting_qubits, A_circuit, method):

    num_qubits = num_state_qubits + num_counting_qubits

    qr_state = QuantumRegister(num_state_qubits+1)
    qr_counting = QuantumRegister(num_counting_qubits)
    cr = ClassicalRegister(num_counting_qubits)
    qc = QuantumCircuit(qr_state, qr_counting, cr, name=f"qmc({method})-{num_qubits}-{0}")

    A = A_circuit
    cQ, Q = Ctrl_Q(num_state_qubits, A)

    # save small example subcircuits for visualization
    global A_, Q_, cQ_, QFTI_
    if (cQ_ and Q_) == None or num_state_qubits <= 6:
        if num_state_qubits < 9: cQ_ = cQ; Q_ = Q
    if A_ == None or num_state_qubits <= 3:
        if num_state_qubits < 5: A_ = A
    if QFTI_ == None or num_counting_qubits <= 3:
        if num_counting_qubits < 4: QFTI_ = inv_qft_gate(num_counting_qubits)

    # Prepare state from A, and counting qubits with H transform
    qc.append(A, qr_state)
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

    qc.measure([qr_counting[m] for m in range(num_counting_qubits)], list(range(num_counting_qubits)))

    return qc


###############################

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
    print("\nDistribution Generator 'R' ="); print(R_ if R_ != None else " ... too large!")
    print("\nFunction Generator 'F' ="); print(F_ if F_ != None else " ... too large!")
    print("\nInverse QFT Circuit ="); print(QFTI_ if QFTI_ != None else "  ... too large!")
