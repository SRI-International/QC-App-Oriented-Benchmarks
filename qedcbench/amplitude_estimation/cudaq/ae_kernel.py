'''
Amplitude Estimation Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import math
from typing import List

import cudaq

# saved circuit for display
QC_ = None

############### Circuit Definition

# Construct A operator that takes |0>_{n+1} to sqrt(1-a) |psi_0>|0> + sqrt(a) |psi_1>|1>
@cudaq.kernel
def A_op(state_obj: cudaq.qview, num_state_qubits: int, theta: float,
         psi_zero: List[int], psi_one: List[int]):
    objective = state_obj[0]

    ry(theta, objective)

    x(objective)
    for i in range(num_state_qubits):
        if psi_zero[i] == 1:
            x.ctrl(objective, state_obj[i + 1])
    x(objective)

    for i in range(num_state_qubits):
        if psi_one[i] == 1:
            x.ctrl(objective, state_obj[i + 1])


@cudaq.kernel
def A_op_inv(state_obj: cudaq.qview, num_state_qubits: int, theta: float,
             psi_zero: List[int], psi_one: List[int]):
    objective = state_obj[0]

    for i_rev in range(num_state_qubits):
        i = num_state_qubits - i_rev - 1
        if psi_one[i] == 1:
            x.ctrl(objective, state_obj[i + 1])

    x(objective)
    for i_rev in range(num_state_qubits):
        i = num_state_qubits - i_rev - 1
        if psi_zero[i] == 1:
            x.ctrl(objective, state_obj[i + 1])
    x(objective)

    ry(-theta, objective)


# Construct the grover-like operator (-S_chi, A_inv, S_0, A)
@cudaq.kernel
def Q_op(state_obj: cudaq.qview, num_state_qubits: int, theta: float,
         psi_zero: List[int], psi_one: List[int]):
    objective = state_obj[0]

    # -S_chi
    x(objective)
    z(objective)
    x(objective)

    # A_circ_inverse
    A_op_inv(state_obj, num_state_qubits, theta, psi_zero, psi_one)

    # S_0
    for i in range(num_state_qubits + 1):
        x(state_obj[i])
    h(objective)
    if num_state_qubits == 1:
        x.ctrl(state_obj[1], objective)
    else:
        x.ctrl(state_obj[1:num_state_qubits + 1], objective)
    h(objective)
    for i in range(num_state_qubits + 1):
        x(state_obj[i])

    # A_circ
    A_op(state_obj, num_state_qubits, theta, psi_zero, psi_one)


@cudaq.kernel
def ae_iqft(register: cudaq.qview):
    M_PI = 3.1415926536
    input_size = register.size()
    for i_qubit in range(input_size):
        ri_qubit = input_size - i_qubit - 1
        h(register[ri_qubit])
        num_crzs = input_size - i_qubit - 1
        if i_qubit < input_size - 1:
            for j in range(num_crzs):
                divisor = 2 ** (j + 1)
                r1.ctrl(-M_PI / divisor,
                        register[ri_qubit], register[ri_qubit - j - 1])


@cudaq.kernel
def ae_kernel(num_state_qubits: int, num_counting_qubits: int, theta: float,
              psi_zero: List[int], psi_one: List[int]):
    num_qubits = num_state_qubits + 1 + num_counting_qubits
    qubits = cudaq.qvector(num_qubits)

    # Prepare state from A, and counting qubits with H transform
    A_op(qubits[0:num_state_qubits + 1], num_state_qubits, theta,
         psi_zero, psi_one)
    for i in range(num_state_qubits + 1, num_qubits):
        h(qubits[i])

    repeat = 1
    for j in range(num_counting_qubits):
        for _ in range(repeat):
            cudaq.control(Q_op,
                          qubits[num_state_qubits + 1 + j],
                          qubits[0:num_state_qubits + 1],
                          num_state_qubits, theta, psi_zero, psi_one)
        repeat = repeat * 2

    # inverse quantum Fourier transform only on counting qubits
    ae_iqft(qubits[num_state_qubits + 1:num_qubits])

    # measure counting qubits
    mz(qubits[num_state_qubits + 1:num_qubits])


def _psi_bits(psi, default_bit: int, num_state_qubits: int, name: str):
    if psi is None:
        return [default_bit] * num_state_qubits

    try:
        bits = [int(bit) for bit in psi]
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{name} must be a bit string or sequence of 0/1 values") from ex

    if len(bits) != num_state_qubits:
        raise ValueError(f"{name} length must match num_state_qubits")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError(f"{name} must contain only 0/1 values")
    return bits


def AmplitudeEstimation(num_state_qubits: int, num_counting_qubits: int, a,
                        psi_zero=None, psi_one=None):
    theta = 2.0 * math.asin(math.sqrt(a))
    psi_zero_bits = _psi_bits(psi_zero, 0, num_state_qubits, "psi_zero")
    psi_one_bits = _psi_bits(psi_one, 1, num_state_qubits, "psi_one")
    qc = [ae_kernel,
          [num_state_qubits, num_counting_qubits, theta,
           psi_zero_bits, psi_one_bits],
          {"counts_dict": True}]

    global QC_
    if num_counting_qubits + num_state_qubits + 1 <= 6:
        QC_ = qc

    return qc

############### Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
    print("Sample Circuit:")
    if QC_ is not None:
        try:
            print(cudaq.draw(QC_[0], *QC_[1]))
        except Exception as ex:
            print(f"ERROR attempting to draw the kernel")
            print(ex)
    else:
        print("  ... too large!")
