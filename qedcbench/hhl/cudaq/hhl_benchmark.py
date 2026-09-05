'''
HHL Linear Solver Benchmark - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import math
import time

import numpy as np

import cudaq

from qedclib import metrics
from qedclib import qcb_mpi as mpi
from qedclib.cudaq import execute as ex
from qedclib.backend_utils import api_display_name
from hhl._common.hhl_helpers import (
    generate_sparse_H,
    analyze_and_print_result as _hhl_analyze_and_print_result,
    ucr_sequence_pure,
    alpha2theta_pure,
)

from typing import List

benchmark_name = "HHL"
verbose = False
QC_ = None


############### Building blocks: state prep, QFT, IQFT

@cudaq.kernel
def hhl_initialize_state(qreg: cudaq.qview, b_bits: List[int]):
    for q in range(len(b_bits)):
        if b_bits[q] == 1:
            x(qreg[q])


@cudaq.kernel
def hhl_qft(qreg: cudaq.qview):
    M_PI = 3.141592653589793
    n = qreg.size()
    for i in range(n):
        h(qreg[i])
        for j in range(n - 1, i, -1):
            divisor = 2 ** (j - i)
            r1.ctrl(M_PI / divisor, qreg[i], qreg[j])


@cudaq.kernel
def hhl_iqft(qreg: cudaq.qview):
    M_PI = 3.141592653589793
    n = qreg.size()
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            divisor = 2 ** (j - i)
            r1.ctrl(-M_PI / divisor, qreg[i], qreg[j])
        h(qreg[i])


############### Sparse-Hamiltonian simulation building blocks

@cudaq.kernel
def hhl_W_gate(q0: cudaq.qubit, q1: cudaq.qubit):
    M_PI = 3.141592653589793
    rz(-M_PI, q1)
    rz(-3.0 * M_PI / 2.0, q0)
    ry(-M_PI / 2.0, q1)
    ry(-M_PI / 2.0, q0)
    rz(-M_PI, q1)
    rz(-3.0 * M_PI / 2.0, q0)

    cx(q0, q1)

    rz(-7.0 * M_PI / 4.0, q1)
    rx(-M_PI / 2.0, q0)
    rx(-M_PI, q1)
    rz(-3.0 * M_PI / 2.0, q0)

    cx(q0, q1)

    ry(-M_PI / 2.0, q1)
    rx(-M_PI / 4.0, q1)

    cx(q1, q0)

    rz(-3.0 * M_PI / 2.0, q0)


@cudaq.kernel
def hhl_V_gate(qreg_a: cudaq.qview, qreg_b: cudaq.qview, k_bits: List[int]):
    n = len(k_bits)
    for q in range(n):
        if k_bits[q] == 1:
            x(qreg_a[q])
            cx(qreg_a[q], qreg_b[q])
            x(qreg_a[q])
        else:
            cx(qreg_a[q], qreg_b[q])


@cudaq.kernel
def hhl_controlled_ham_sim(qreg_a: cudaq.qview, qreg_b: cudaq.qview,
                           anc: cudaq.qubit, control: cudaq.qubit,
                           k_bits: List[int],
                           t: float, diag_el: float, off_diag_el: float,
                           sign_: float):
    n = len(k_bits)

    # Diagonal phase via ancilla phase-kickback
    x(anc)
    r1.ctrl(-t * (diag_el + sign_ * off_diag_el), control, anc)
    x(anc)

    # V (Hamiltonian oracle)
    hhl_V_gate(qreg_a, qreg_b, k_bits)

    # W + X + CCX on each pair
    for q in range(n):
        hhl_W_gate(qreg_a[q], qreg_b[q])
        x(qreg_b[q])
        x.ctrl([qreg_a[q], qreg_b[q]], anc)

    # Off-diagonal phase
    r1.ctrl(sign_ * 2.0 * t * off_diag_el, control, anc)

    # Uncompute (reverse order)
    for q in range(n - 1, -1, -1):
        x.ctrl([qreg_a[q], qreg_b[q]], anc)
        x(qreg_b[q])
        hhl_W_gate(qreg_a[q], qreg_b[q])

    # V is its own inverse
    hhl_V_gate(qreg_a, qreg_b, k_bits)


############### Uniformly-controlled RY (Walsh–Hadamard expansion)

@cudaq.kernel
def hhl_ucr_ry(qubits: cudaq.qview, anc: cudaq.qubit,
               ry_angles: List[float], ctrl_indices: List[int]):
    L = len(ry_angles)
    for i in range(L):
        ry(ry_angles[i], anc)
        cx(qubits[ctrl_indices[i]], anc)


############### Main HHL kernel
# Single static kernel for any n_input. Encodes the ancilla bit + input
# register into an int return value; the framework formats it with
# result_width = n_input + 1, producing a bitstring with ancilla leftmost
# (matching the qiskit `{cr_aux} {cr}` format the analyzer's postselect()
# expects).

@cudaq.kernel
def hhl_kernel(n_input: int, n_t: int,
               b_bits: List[int], k_bits: List[int],
               diag_el: float, off_diag_el: float, sign_: float,
               ry_angles_fwd: List[float],
               ctrl_idx_fwd: List[int]) -> int:
    M_PI = 3.141592653589793
    # Allocate every register at once and slice it. Allocating them one at a
    # time grows the simulator state repeatedly, and the last of those
    # reallocations is rejected at the maximum single-GPU width.
    qb_start = n_input
    qt_start = 2 * n_input
    anc_index = 2 * n_input + n_t
    all_qubits = cudaq.qvector(anc_index + 1)
    qa = all_qubits[0:qb_start]
    qb = all_qubits[qb_start:qt_start]
    qt = all_qubits[qt_start:anc_index]
    anc = all_qubits[anc_index]

    # 1) State prep |b>
    hhl_initialize_state(qa, b_bits)

    # 2) Hadamard clock
    for q in range(n_t):
        h(qt[q])

    # 3) QPE: controlled e^(-iAt) with t_q = -2π·2^q.
    for q in range(n_t):
        pow_q = 2 ** q
        t_q = -(2.0 * M_PI) * pow_q
        hhl_controlled_ham_sim(qa, qb, anc, qt[q], k_bits,
                               t_q, diag_el, off_diag_el, sign_)

    # 4) Inverse QFT on clock
    hhl_iqft(qt)

    # 5) Reset ancilla, apply uniformly-controlled RY rotation
    reset(anc)
    hhl_ucr_ry(qt, anc, ry_angles_fwd, ctrl_idx_fwd)

    # 6) Mid-circuit ancilla measurement (post-selection signal)
    a_bit = mz(anc)

    # 7) Reset ancilla, apply inverse QPE
    reset(anc)
    hhl_qft(qt)
    for q_iter in range(n_t):
        q = n_t - 1 - q_iter
        pow_q = 2 ** q
        t_q = (2.0 * M_PI) * pow_q
        hhl_controlled_ham_sim(qa, qb, anc, qt[q], k_bits,
                               t_q, diag_el, off_diag_el, sign_)

    # Final Hadamard on clock
    for q in range(n_t):
        h(qt[q])

    # Encode: ancilla at bit 2^n_input (becomes leftmost char in result_width
    # format), input register at bits 2^0 .. 2^(n_input-1).
    encoded = 0
    if a_bit:
        encoded = encoded + (1 << n_input)
    for q in range(n_input):
        if mz(qa[q]):
            encoded = encoded + (1 << q)
    return encoded


############### Build kernel inputs from a problem instance

def _problem_to_kernel_args(A, b, num_clock_qubits):
    N = len(A)
    n = int(np.log2(N))

    diag_el = float(A[0, 0])
    # Find the off-diagonal index k in row 0 of A
    k = None
    for j in range(1, N):
        if A[0, j] != 0:
            k = j
            off_diag_el = float(A[0, j])
            break
    if k is None:
        raise ValueError("A has no non-zero off-diagonal in row 0")

    k_bits = [(k >> q) & 1 for q in range(n)]
    parity = bin(k).count('1') % 2
    sign_ = -1.0 if parity == 1 else 1.0

    b_bits = [(b >> q) & 1 for q in range(n)]

    # Inversion-rotation thetas
    C = 1.0 / 4.0
    n_t = num_clock_qubits
    alpha = [2.0 * np.arcsin(C)]
    for x in range(1, 2 ** n_t):
        x_bin_rev = np.binary_repr(x, width=n_t)[::-1]
        lam = int(x_bin_rev, 2) / (2 ** n_t)
        if lam < C:
            alpha.append(0.0)
        else:
            alpha.append(2.0 * np.arcsin(C / lam))

    theta = alpha2theta_pure(alpha)
    ry_angles, ctrl_indices = ucr_sequence_pure(n_t, list(theta))

    return {
        "b_bits": b_bits,
        "k_bits": k_bits,
        "diag_el": diag_el,
        "off_diag_el": off_diag_el,
        "sign_": sign_,
        "ry_angles": ry_angles,
        "ctrl_indices": ctrl_indices,
    }


############### Run loop

def run(min_qubits=3, max_qubits=6, skip_qubits=1, max_circuits=3, num_shots=100,
        method=1, use_best_widths=True, min_register_qubits=1,
        backend_id=None, provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None, api=None, warmup=False, get_circuits=False,
        draw_circuits=True, plot_results=True):

    mpi.init()

    max_qubits = max(4, max_qubits)
    min_qubits = min(max(4, min_qubits), max_qubits)
    skip_qubits = max(1, skip_qubits)
    if context is None:
        context = f"{benchmark_name} Benchmark"

    min_input_qubits = int((min_qubits - 1) / 3)
    max_input_qubits = int((max_qubits - 1) / 3)
    min_clock_qubits = min_qubits - 1 - 2 * min_input_qubits
    max_clock_qubits = max_qubits - 1 - 2 * max_input_qubits

    print(f"{benchmark_name} Benchmark Program - {api_display_name(api)}")

    metrics.init_metrics(warmup)

    def execution_handler(qc, result, num_qubits_arg, circuit_id, num_shots_arg):
        num_qubits_int = int(num_qubits_arg)
        counts, fidelity = _hhl_analyze_and_print_result(
            qc, result, num_qubits_int, num_shots_arg, s_int=int(circuit_id),
            verbose=verbose)
        metrics.store_metric(num_qubits_int, circuit_id, 'fidelity', fidelity)

    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
                            hub=hub, group=group, project=project,
                            exec_options=exec_options)

    diag_el = 0.5
    off_diag_el = -0.25

    # match qiskit's seeded RNG so problem instances align
    np.random.seed(0)

    for num_input_qubits in range(min_input_qubits, max_input_qubits + 1, skip_qubits):
        N = 2 ** num_input_qubits

        for num_clock_qubits in range(min_clock_qubits, max_clock_qubits + 1, skip_qubits):
            num_qubits = 2 * num_input_qubits + num_clock_qubits + 1

            if use_best_widths:
                if num_input_qubits != int((num_qubits - 1) / 3) \
                   or num_clock_qubits != (num_qubits - 1 - 2 * num_input_qubits):
                    continue

            if min_register_qubits > 1 \
               and (num_input_qubits < min_register_qubits or num_clock_qubits < min_register_qubits):
                continue

            print(f"************\nExecuting {max_circuits} circuits with {num_qubits} qubits, "
                  f"using {num_input_qubits} input qubits and {num_clock_qubits} clock qubits")

            for i in range(max_circuits):
                b = int(np.random.choice(range(1, N)))
                off_diag_index = int(np.random.choice(range(1, N)))
                # Pack (i+1, odi, b) into 24-bit fields so the analyzer can
                # recover them losslessly for any n_input up to 24.
                s_int = ((i + 1) << 48) | (off_diag_index << 24) | b
                circuit_id = s_int

                A = generate_sparse_H(num_input_qubits, off_diag_index,
                                      diag_el=diag_el, off_diag_el=off_diag_el)

                ts = time.time()
                args = _problem_to_kernel_args(A, b, num_clock_qubits)
                metrics.store_metric(num_qubits, circuit_id, 'create_time',
                                     time.time() - ts)

                kernel_args = [
                    num_input_qubits, num_clock_qubits,
                    args["b_bits"], args["k_bits"],
                    args["diag_el"], args["off_diag_el"], args["sign_"],
                    args["ry_angles"], args["ctrl_indices"],
                ]
                qc = [hhl_kernel, kernel_args, {"result_width": num_input_qubits + 1}]

                global QC_
                if QC_ is None or num_qubits <= 6:
                    QC_ = qc

                ex.submit_circuit(qc, num_qubits, circuit_id, shots=num_shots)

            ex.throttle_execution(metrics.finalize_group)

    ex.finalize_execution(metrics.finalize_group)


def kernel_draw():
    pass  # cudaq.draw not implemented for this port
