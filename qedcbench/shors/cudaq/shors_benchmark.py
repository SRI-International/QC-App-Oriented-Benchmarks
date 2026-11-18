'''
Shor's Order Finding Benchmark - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import math
import time

import numpy as np

import cudaq
from typing import List

from qedclib import metrics
from qedclib import qcb_mpi as mpi
from qedclib.cudaq import execute as ex
from qedclib.backend_utils import api_display_name

from shors._common.shors_utils import generate_base, multiplicative_order
from shors._common.shors_helpers import (
    build_shors_m1_arrays,
    build_shors_m2_iteration_arrays,
    analyze_and_print_result as _shors_analyze_and_print_result,
    expected_shor_dist,
)


benchmark_name = "Shor's Order Finding"
verbose = False
QC_ = None


############### Op-kind dispatch (must match shors._common.shors_helpers)
# 1 = h(q1)
# 2 = x(q1)
# 3 = cx(q1, q2)
# 4 = swap.ctrl(q1, q2, q3)        cswap
# 5 = r1(angle, q1)                phase
# 6 = r1.ctrl(angle, q1, q2)       cp (single-controlled phase)
# 7 = r1.ctrl(angle, q1, q2, q3)   ccp (doubly-controlled phase)
# 8 = rz.ctrl(angle, q1, q2)       crz


############### Method 1 kernel (static, sample path via slice mz)
# Single kernel for any n: num_qubits and num_counting are runtime int
# params (used as qvector size and slice bound — not JIT-trap data ints).
# `mz(qubits[0:num_counting])` produces an LSB-leftmost count key; the
# {"reverse_bit_order": True} option on the qc list flips it to MSB-leftmost
# to match qiskit's `bin(s)[2:].zfill(2n)` analyzer format.

@cudaq.kernel
def shors_kernel_m1(num_qubits: int, num_counting: int,
                    op_kind: List[int],
                    q1: List[int], q2: List[int], q3: List[int],
                    angle: List[float]):
    qubits = cudaq.qvector(num_qubits)

    L = len(op_kind)
    for i in range(L):
        kind = op_kind[i]
        if kind == 1:
            h(qubits[q1[i]])
        elif kind == 2:
            x(qubits[q1[i]])
        elif kind == 3:
            cx(qubits[q1[i]], qubits[q2[i]])
        elif kind == 4:
            swap.ctrl(qubits[q1[i]], qubits[q2[i]], qubits[q3[i]])
        elif kind == 5:
            r1(angle[i], qubits[q1[i]])
        elif kind == 6:
            r1.ctrl(angle[i], qubits[q1[i]], qubits[q2[i]])
        elif kind == 7:
            r1.ctrl(angle[i], qubits[q1[i]], qubits[q2[i]], qubits[q3[i]])
        elif kind == 8:
            rz.ctrl(angle[i], qubits[q1[i]], qubits[q2[i]])

    mz(qubits[0:num_counting])


############### Method 2 kernel factory (iterative QPE with classical feedforward)
# Loop body per iteration k in 0..2n-1:
#   1. If k > 0 and previous bit was 1: x(counting) — reset.
#   2. h(counting).
#   3. Apply controlled_Ua(2^(2n-1-k)) gates (this iteration's slice).
#   4. Conditional phase rotations from past measurements.
#   5. h(counting); past_bits[k] = mz(counting).

@cudaq.kernel
def shors_kernel_m2(num_qubits: int, num_counting: int,
                    op_kind: List[int],
                    q1: List[int], q2: List[int], q3: List[int],
                    angle: List[float],
                    offsets: List[int]) -> int:
    M_PI = 3.141592653589793
    qubits = cudaq.qvector(num_qubits)

    # Initialize mult register to |1>
    x(qubits[1])

    # Accumulated past-measurement bits as an int (avoids exec/linecache for
    # a literal-length List[bool]). `data` carries the running integer; the
    # per-bit feedback decomposition (claude's pattern) inspects bit j of
    # `data` to apply r1(pi/2^(k-j)) — O(n^2) total feedback, vs codex's
    # integer-equality fan-out which is O(2^n).
    data = 0
    previous_bit = 0

    for k in range(num_counting):
        if previous_bit == 1:
            x(qubits[0])

        h(qubits[0])

        for i in range(offsets[k], offsets[k + 1]):
            kind = op_kind[i]
            if kind == 1:
                h(qubits[q1[i]])
            elif kind == 2:
                x(qubits[q1[i]])
            elif kind == 3:
                cx(qubits[q1[i]], qubits[q2[i]])
            elif kind == 4:
                swap.ctrl(qubits[q1[i]], qubits[q2[i]], qubits[q3[i]])
            elif kind == 5:
                r1(angle[i], qubits[q1[i]])
            elif kind == 6:
                r1.ctrl(angle[i], qubits[q1[i]], qubits[q2[i]])
            elif kind == 7:
                r1.ctrl(angle[i], qubits[q1[i]], qubits[q2[i]], qubits[q3[i]])
            elif kind == 8:
                rz.ctrl(angle[i], qubits[q1[i]], qubits[q2[i]])

        for j in range(k):
            if ((data >> j) & 1) == 1:
                divisor = 2 ** (k - j)
                r1(M_PI / divisor, qubits[0])

        h(qubits[0])

        bit = mz(qubits[0])
        previous_bit = 0
        if bit:
            previous_bit = 1
            data = data + (1 << k)

    return data


def _shors_m2_qc(number, base):
    n = int(math.ceil(math.log(number, 2)))
    arrays = build_shors_m2_iteration_arrays(n, base, number)
    params = [arrays["num_qubits"], 2 * n,
              arrays["op_kind"],
              arrays["q1"], arrays["q2"], arrays["q3"],
              arrays["angle"],
              arrays["offsets"]]
    qc = [shors_kernel_m2, params, {"result_width": 2 * n}]

    global QC_
    QC_ = qc
    return qc


def ShorsAlgorithm(number, base, method, verbose=False):
    if method != 1:
        raise NotImplementedError(
            f"ShorsAlgorithm wrapper supports method=1; for method=2 use "
            f"_shors_m2_qc directly.")

    arrays = build_shors_m1_arrays(number, base)
    n = arrays["n"]

    params = [arrays["num_qubits"], 2 * n,
              arrays["op_kind"], arrays["q1"], arrays["q2"], arrays["q3"],
              arrays["angle"]]
    qc = [shors_kernel_m1, params, {"reverse_bit_order": True}]

    global QC_
    QC_ = qc
    return qc


############### Run loop (mirrors qiskit run for methods 1 and 2)

def run(min_qubits=3, max_circuits=1, max_qubits=18, num_shots=100, method=1,
        verbose=False, backend_id=None, provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None, api=None, warmup=False, get_circuits=False,
        draw_circuits=True, plot_results=True):

    mpi.init()

    if method not in (1, 2):
        raise NotImplementedError(
            f"Shors cudaq port supports methods 1 and 2 only "
            f"(got method={method}).")

    print(f"{benchmark_name} ({method}) Benchmark - {api_display_name(api)}")

    if method == 1:
        qubit_multiple = 4
        min_qubits = max(min_qubits, 10)
    else:
        qubit_multiple = 2
        min_qubits = max(min_qubits, 7)
    max_qubits = max(max_qubits, min_qubits)
    if max_qubits < min_qubits:
        print(f"Max number of qubits {max_qubits} too low for method {method}")
        return

    metrics.init_metrics(warmup)

    def execution_handler(qc, result, num_qubits_arg, circuit_id, num_shots_arg):
        num_qubits_int = int(num_qubits_arg)
        order = eval(circuit_id)[1]
        if method == 2:
            num_bits = int((num_qubits_int - 3) / 2)
            counts_dict = result.get_counts()
            correct_dist = expected_shor_dist(num_bits, order, num_shots_arg)
            fidelity = metrics.polarization_fidelity(counts_dict, correct_dist)
        else:
            _, fidelity = _shors_analyze_and_print_result(
                qc, result, num_qubits_int, num_shots_arg,
                order=order, method=method)
        metrics.store_metric(num_qubits_int, circuit_id, 'fidelity', fidelity)

    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
                            hub=hub, group=group, project=project,
                            exec_options=exec_options)

    # match qiskit's seeded RNG
    np.random.seed(0)

    for num_qubits in range(min_qubits, max_qubits + 1, qubit_multiple):
        if method == 1:
            num_bits = int((num_qubits - 2) / 4)
        else:
            num_bits = int((num_qubits - 3) / 2)
        num_circuits = min(2 ** (num_qubits - 1), max_circuits)
        print(f"************\nExecuting [{num_circuits}] circuits with "
              f"num_qubits = {num_qubits}")

        for _ in range(num_circuits):
            base = 1
            while base == 1:
                number = np.random.randint(2 ** (num_bits - 1) + 1, 2 ** num_bits)
                order = np.random.randint(2, number)
                base = generate_base(number, order)
            order = multiplicative_order(base, number)
            number_order = (number, order)
            circuit_id = number_order

            ts = time.time()
            if method == 1:
                qc = ShorsAlgorithm(number, base, method=method, verbose=verbose)
            else:
                qc = _shors_m2_qc(number, base)
            metrics.store_metric(num_qubits, circuit_id, 'create_time',
                                 time.time() - ts)

            ex.submit_circuit(qc, num_qubits, circuit_id, num_shots)

        ex.throttle_execution(metrics.finalize_group)

    ex.finalize_execution(metrics.finalize_group)


def kernel_draw():
    pass


def load_data_and_plot(folder=None, backend_id=None, **kwargs):
    print("load_data_and_plot is not implemented for the cudaq shors port.")
