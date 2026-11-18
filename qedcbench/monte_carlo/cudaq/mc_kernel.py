'''
Monte Carlo Sampling Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import functools
import math

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.polynomial import polyfit

import cudaq

import mc_utils

from typing import List

QC_ = None


def _empty_A_arrays():
    return {
        "kinds": [], "targets": [], "thetas": [],
        "ctrl_indices": [], "ctrl_states": [], "ctrl_offsets": [0],
    }


def _append_gate(A, kind, target, theta=0.0, controls=None, ctrl_state=None):
    controls = list(controls or [])
    if ctrl_state is None:
        ctrl_state = '1' * len(controls)
    A["kinds"].append(int(kind))
    A["targets"].append(int(target))
    A["thetas"].append(float(theta))
    for c, s in zip(controls, ctrl_state):
        A["ctrl_indices"].append(int(c))
        A["ctrl_states"].append(1 if s == '1' else 0)
    A["ctrl_offsets"].append(len(A["ctrl_indices"]))


# R = state_prep (mc_utils.region_probs bisection tree) +
# F = f_on_objective (polynomial fit via mc_utils.binary_expansion)
def _build_A_arrays_method1(target_dist, f, num_state_qubits, epsilon, degree):
    A = _empty_A_arrays()
    S = num_state_qubits

    # ----- R (state_prep) -----
    r_probs = mc_utils.region_probs(target_dist, S)
    regions = list(r_probs.keys())
    r_norm = {}
    for r in regions:
        num_controls = len(r) - 1
        super_key = r[:num_controls]
        if super_key == '':
            r_norm[super_key] = 1
        elif super_key == '1':
            r_norm[super_key] = r_probs[super_key]
            r_norm['0'] = 1 - r_probs[super_key]
        else:
            try:
                r_norm[super_key] = r_probs[super_key]
            except KeyError:
                r_norm[super_key] = (
                    r_norm[super_key[:num_controls - 1]]
                    - r_probs[super_key[:num_controls - 1] + '1'])
        norm = r_norm[super_key]
        p = 0.0 if norm == 0 else r_probs[r] / norm
        theta = 2 * np.arcsin(np.sqrt(p))

        if r == '1':
            _append_gate(A, kind=2, target=S - 1, theta=-theta)
        else:
            controls = [S - 1 - i for i in range(num_controls)]
            target = S - 1 - num_controls
            ctrl_state = r[:-1] if num_controls > 0 else ''
            _append_gate(A, kind=2, target=target, theta=-theta,
                         controls=controls, ctrl_state=ctrl_state)

    # ----- F (f_on_objective) -----
    c_star = (2 * epsilon) ** (1 / (degree + 1))
    f_ = functools.partial(f, num_state_qubits=S)
    zeta_ = functools.partial(mc_utils.zeta_from_f, func=f_,
                              epsilon=epsilon, degree=degree, c=c_star)
    x_eval = np.linspace(0.0, 2 ** S - 1, num=degree + 1)
    poly = Polynomial(polyfit(x_eval, zeta_(x_eval), degree))
    b_exp = mc_utils.binary_expansion(S, poly)
    objective = S
    for ctrl_tuple, coeff in b_exp.items():
        theta = 2.0 * coeff
        ctrls = list(ctrl_tuple)
        if not ctrls:
            _append_gate(A, kind=2, target=objective, theta=-theta)
        else:
            _append_gate(A, kind=2, target=objective, theta=-theta,
                         controls=ctrls, ctrl_state='1' * len(ctrls))
    return A


# R = uniform_prep (H on each state qubit) +
# F = square_on_objective (CX from each state qubit to objective)
def _build_A_arrays_method2(num_state_qubits):
    A = _empty_A_arrays()
    S = num_state_qubits
    for i in range(S):
        _append_gate(A, kind=0, target=i)
    for i in range(S):
        _append_gate(A, kind=1, target=S, controls=[i])
    return A


@cudaq.kernel
def _apply_A_gates(state_obj: cudaq.qview,
                   kinds: List[int], targets: List[int], thetas: List[float],
                   ctrl_indices: List[int], ctrl_states: List[int],
                   ctrl_offsets: List[int]):
    L = len(kinds)
    for i in range(L):
        kind = kinds[i]
        target = state_obj[targets[i]]
        theta = thetas[i]
        c_start = ctrl_offsets[i]
        c_end = ctrl_offsets[i + 1]
        nc = c_end - c_start

        for k in range(c_start, c_end):
            if ctrl_states[k] == 0:
                x(state_obj[ctrl_indices[k]])

        if nc == 0:
            if kind == 0:
                h(target)
            elif kind == 1:
                x(target)
            elif kind == 2:
                ry(theta, target)
        else:
            ctrls = [state_obj[ctrl_indices[c_start + j]] for j in range(nc)]
            if kind == 1:
                x.ctrl(ctrls, target)
            elif kind == 2:
                ry.ctrl(theta, ctrls, target)

        for k in range(c_start, c_end):
            if ctrl_states[k] == 0:
                x(state_obj[ctrl_indices[k]])


@cudaq.kernel
def _apply_A_gates_inv(state_obj: cudaq.qview,
                       kinds: List[int], targets: List[int], thetas: List[float],
                       ctrl_indices: List[int], ctrl_states: List[int],
                       ctrl_offsets: List[int]):
    L = len(kinds)
    for i_iter in range(L):
        i = L - 1 - i_iter
        kind = kinds[i]
        target = state_obj[targets[i]]
        theta_inv = -thetas[i]
        c_start = ctrl_offsets[i]
        c_end = ctrl_offsets[i + 1]
        nc = c_end - c_start

        for k in range(c_start, c_end):
            if ctrl_states[k] == 0:
                x(state_obj[ctrl_indices[k]])

        if nc == 0:
            if kind == 0:
                h(target)
            elif kind == 1:
                x(target)
            elif kind == 2:
                ry(theta_inv, target)
        else:
            ctrls = [state_obj[ctrl_indices[c_start + j]] for j in range(nc)]
            if kind == 1:
                x.ctrl(ctrls, target)
            elif kind == 2:
                ry.ctrl(theta_inv, ctrls, target)

        for k in range(c_start, c_end):
            if ctrl_states[k] == 0:
                x(state_obj[ctrl_indices[k]])


# Each cycle in Q applies in order: -S_chi, A_circ_inverse, S_0, A_circ
@cudaq.kernel
def _apply_Q(state_obj: cudaq.qview, num_state_qubits: int,
             kinds: List[int], targets: List[int], thetas: List[float],
             ctrl_indices: List[int], ctrl_states: List[int],
             ctrl_offsets: List[int]):
    obj = num_state_qubits
    # -S_chi
    x(state_obj[obj])
    z(state_obj[obj])
    x(state_obj[obj])

    # A_circ_inverse
    _apply_A_gates_inv(state_obj, kinds, targets, thetas,
                       ctrl_indices, ctrl_states, ctrl_offsets)

    # S_0
    for i in range(num_state_qubits + 1):
        x(state_obj[i])
    h(state_obj[obj])
    x.ctrl(state_obj[0:num_state_qubits], state_obj[obj])
    h(state_obj[obj])
    for i in range(num_state_qubits + 1):
        x(state_obj[i])

    # A_circ
    _apply_A_gates(state_obj, kinds, targets, thetas,
                   ctrl_indices, ctrl_states, ctrl_offsets)


@cudaq.kernel
def _mc_iqft(register: cudaq.qview):
    M_PI = 3.141592653589793
    input_size = register.size()
    for i_qubit in range(input_size):
        ri_qubit = input_size - i_qubit - 1
        h(register[ri_qubit])
        num_crzs = input_size - i_qubit - 1
        if i_qubit < input_size - 1:
            for j in range(num_crzs):
                exp = j + 1
                divisor = 1
                for _ in range(exp):
                    divisor = divisor * 2
                r1.ctrl(-M_PI / divisor, register[ri_qubit], register[ri_qubit - j - 1])


@cudaq.kernel
def mc_kernel(num_state_qubits: int, num_counting_qubits: int,
              kinds: List[int], targets: List[int], thetas: List[float],
              ctrl_indices: List[int], ctrl_states: List[int],
              ctrl_offsets: List[int]):
    total = num_counting_qubits + num_state_qubits + 1
    qubits = cudaq.qvector(total)
    counting = qubits[0:num_counting_qubits]
    state_obj = qubits[num_counting_qubits:total]

    # Prepare state from A, and counting qubits with H transform
    _apply_A_gates(state_obj, kinds, targets, thetas,
                   ctrl_indices, ctrl_states, ctrl_offsets)

    for i in range(num_counting_qubits):
        h(counting[i])

    repeat = 1
    for j in range(num_counting_qubits):
        ctrl_q = counting[j]
        for _ in range(repeat):
            cudaq.control(_apply_Q, ctrl_q,
                          state_obj, num_state_qubits,
                          kinds, targets, thetas,
                          ctrl_indices, ctrl_states, ctrl_offsets)
        repeat = repeat * 2

    # inverse quantum Fourier transform only on counting qubits
    _mc_iqft(counting)
    mz(counting)


def MonteCarloSampling(target_dist, f, num_state_qubits, num_counting_qubits,
                       epsilon=0.05, degree=2, method=2):
    if method == 1:
        A = _build_A_arrays_method1(target_dist, f, num_state_qubits, epsilon, degree)
    else:
        A = _build_A_arrays_method2(num_state_qubits)

    params = [num_state_qubits, num_counting_qubits,
              A["kinds"], A["targets"], A["thetas"],
              A["ctrl_indices"], A["ctrl_states"], A["ctrl_offsets"]]
    qc = [mc_kernel, params, {"counts_dict": True}]

    global QC_
    if num_state_qubits + 1 + num_counting_qubits <= 6:
        QC_ = qc
    return qc


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
