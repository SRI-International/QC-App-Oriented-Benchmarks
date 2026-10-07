'''
Shor's Order Finding Algorithm — qiskit-free helper module.
(C) Quantum Economic Development Consortium (QED-C) 2024.

Pure-Python gate-record emitter that mirrors the qiskit ShorsAlgorithm
construction (phiADD → ccphiADDmodN → cMULTamodN → controlled_Ua) without
importing qiskit. The emitter produces parallel arrays consumed by the
cudaq @cudaq.kernel walker so that the cudaq port has no qiskit / qiskit_aer
runtime dependency.

Method 1 only (the gate sequence with 4n+2 qubits and 2n counting qubits).
Method 2's dynamic-circuit feedback path lives elsewhere.

Op-kind dispatch table (must match the cudaq kernel walker):
    1 = h(target)                                   uses q1
    2 = x(target)                                   uses q1
    3 = cx(ctrl, target)                            uses q1=ctrl, q2=target
    4 = cswap(ctrl, a, b)                           uses q1=ctrl, q2=a, q3=b
    5 = r1(angle, target)                           uses q1=target, angle
    6 = r1.ctrl(angle, ctrl, target)                uses q1=ctrl, q2=target, angle
    7 = r1.ctrl(angle, c1, c2, target)  (2-control) uses q1=c1, q2=c2, q3=target, angle
    8 = rz.ctrl(angle, ctrl, target)                uses q1=ctrl, q2=target, angle
'''

import math

import numpy as np

from qedclib import metrics
from shors._common.shors_utils import getAngles, modinv


############### Gate-record emitter

class _GateRecorder:
    """Append gate records into parallel List[int]/List[float] arrays.

    Each record is one gate. Self-inverse gates (h, x, cx, cswap) keep
    their angle = 0.0; rotation gates (r1, cr1, ccr1, crz) carry the
    angle. Inverse-of-circuit is implemented by walking records in
    reverse and negating angles.
    """

    H, X, CX, CSWAP, R1, CR1, CCR1, CRZ = range(1, 9)

    def __init__(self):
        self.op_kind = []
        self.q1 = []
        self.q2 = []
        self.q3 = []
        self.angle = []

    def _emit(self, kind, q1, q2, q3, angle):
        self.op_kind.append(int(kind))
        self.q1.append(int(q1))
        self.q2.append(int(q2))
        self.q3.append(int(q3))
        self.angle.append(float(angle))

    # --- primitive gates ---
    def h(self, q):
        self._emit(self.H, q, 0, 0, 0.0)

    def x(self, q):
        self._emit(self.X, q, 0, 0, 0.0)

    def cx(self, c, t):
        self._emit(self.CX, c, t, 0, 0.0)

    def cswap(self, c, a, b):
        self._emit(self.CSWAP, c, a, b, 0.0)

    def r1(self, angle, t):
        self._emit(self.R1, t, 0, 0, angle)

    def cr1(self, angle, c, t):
        self._emit(self.CR1, c, t, 0, angle)

    def ccr1(self, angle, c1, c2, t):
        self._emit(self.CCR1, c1, c2, t, angle)

    def crz(self, angle, c, t):
        self._emit(self.CRZ, c, t, 0, angle)

    # --- structural ---
    def extend_inverse(self, sub):
        """Append the inverse of `sub` (another _GateRecorder) by walking
        its records in reverse with negated rotation angles. h/x/cx/cswap
        are self-inverse (angle stays 0.0); r1/cr1/ccr1/crz negate."""
        for i in range(len(sub.op_kind) - 1, -1, -1):
            kind = sub.op_kind[i]
            if kind in (self.H, self.X, self.CX, self.CSWAP):
                self._emit(kind, sub.q1[i], sub.q2[i], sub.q3[i], 0.0)
            else:
                self._emit(kind, sub.q1[i], sub.q2[i], sub.q3[i],
                           -sub.angle[i])

    # --- composite gates from qiskit's shors_benchmark.py ---

    def phi_add(self, num_qubits, a, qubit_offset, inverse=False):
        """phiADD(num_qubits, a) = qc.p(angle[i], i) for i in range(num_qubits).
        Inverse = walk in reverse with negated angles."""
        angles = getAngles(a, num_qubits)
        rng = (range(num_qubits - 1, -1, -1) if inverse
               else range(num_qubits))
        sign = -1.0 if inverse else 1.0
        for i in rng:
            self.r1(sign * float(angles[i]), qubit_offset + i)

    def cphi_add(self, num_qubits, a, ctrl, qubit_offset, inverse=False):
        """phiADD(num_qubits, a).to_gate().control(1).
        Single-controlled phase add."""
        angles = getAngles(a, num_qubits)
        rng = (range(num_qubits - 1, -1, -1) if inverse
               else range(num_qubits))
        sign = -1.0 if inverse else 1.0
        for i in rng:
            self.cr1(sign * float(angles[i]), ctrl, qubit_offset + i)

    def ccphi_add(self, num_qubits, a, c1, c2, qubit_offset, inverse=False):
        """phiADD(num_qubits, a).to_gate().control(2).
        Doubly-controlled phase add."""
        angles = getAngles(a, num_qubits)
        rng = (range(num_qubits - 1, -1, -1) if inverse
               else range(num_qubits))
        sign = -1.0 if inverse else 1.0
        for i in rng:
            self.ccr1(sign * float(angles[i]), c1, c2, qubit_offset + i)

    def qft(self, qubit_indices, inverse=False):
        """qft_gate(input_size) / inv_qft_gate(input_size) from
        shors/qiskit/shors_benchmark.py — Shors's local QFT convention.
        Operates on the given qubit indices in qiskit-register order
        (qubit_indices[i] ↔ qiskit's qr[i])."""
        size = len(qubit_indices)
        if not inverse:
            for i_qubit in range(size):
                hidx = size - 1 - i_qubit
                if hidx < size - 1:
                    num_crzs = i_qubit
                    for j in range(num_crzs):
                        divisor = 2 ** (num_crzs - j)
                        self.crz(math.pi / divisor,
                                 qubit_indices[hidx],
                                 qubit_indices[size - 1 - j])
                self.h(qubit_indices[hidx])
        else:
            for i_qubit in range(size - 1, -1, -1):
                hidx = size - 1 - i_qubit
                self.h(qubit_indices[hidx])
                if hidx < size - 1:
                    num_crzs = i_qubit
                    for j in range(num_crzs - 1, -1, -1):
                        divisor = 2 ** (num_crzs - j)
                        self.crz(-math.pi / divisor,
                                 qubit_indices[hidx],
                                 qubit_indices[size - 1 - j])

    def ccphi_add_modN(self, n, a, N, c1, c2, main_offset, anc):
        """ccphiADDmodN(num_qubits=n, a, N).
        Operates on qubits [c1, c2] + qr_main(n+1) + [anc] where
        qr_main = main_offset .. main_offset+n.
        """
        main_qubits = [main_offset + i for i in range(n + 1)]

        # qc.append(ccphiadda_gate, ctl_main_qubits)  — ccphiADD(n+1, a)
        self.ccphi_add(n + 1, a, c1, c2, main_offset, inverse=False)

        # qc.append(phiaddN_inv_gate, qr_main)        — phiADD(n+1, N).inverse()
        self.phi_add(n + 1, N, main_offset, inverse=True)

        # qc.append(inv_qft_gate(n+1), qr_main)
        self.qft(main_qubits, inverse=True)

        # qc.cx(qr_main[-1], qr_ancilla[0])
        self.cx(main_qubits[-1], anc)

        # qc.append(qft_gate(n+1), qr_main)
        self.qft(main_qubits, inverse=False)

        # qc.append(cphiaddN_gate, anc_main_qubits)   — cphiADD(n+1, N)
        self.cphi_add(n + 1, N, anc, main_offset, inverse=False)

        # qc.append(ccphiadda_inv_gate, ctl_main_qubits) — ccphiADD(n+1, a).inverse()
        self.ccphi_add(n + 1, a, c1, c2, main_offset, inverse=True)

        # qc.append(inv_qft_gate(n+1), qr_main)
        self.qft(main_qubits, inverse=True)

        # qc.x(qr_main[-1])
        self.x(main_qubits[-1])
        # qc.cx(qr_main[-1], qr_ancilla[0])
        self.cx(main_qubits[-1], anc)
        # qc.x(qr_main[-1])
        self.x(main_qubits[-1])

        # qc.append(qft_gate(n+1), qr_main)
        self.qft(main_qubits, inverse=False)

        # qc.append(ccphiadda_gate, ctl_main_qubits)
        self.ccphi_add(n + 1, a, c1, c2, main_offset, inverse=False)

    def cMULTamodN(self, n, a, N, ctrl, x_offset, main_offset, anc):
        """cMULTamodN(n, a, N).
        Operates on qubits [ctrl] + qr_x(n) + qr_main(n+1) + [anc].
        """
        main_qubits = [main_offset + i for i in range(n + 1)]

        # qc.append(qft_gate(n+1), qr_main)
        self.qft(main_qubits, inverse=False)

        # for i in range(n):
        #     ccphiADDmodN(n, (2**i)*a % N, N) on [ctrl, qr_x[i]] + qr_main + [anc]
        for i in range(n):
            a_eff = (2 ** i) * a % N
            self.ccphi_add_modN(n, a_eff, N,
                                ctrl, x_offset + i, main_offset, anc)

        # qc.append(inv_qft_gate(n+1), qr_main)
        self.qft(main_qubits, inverse=True)

    def controlled_Ua(self, n, a, exponent, N,
                      ctrl, x_offset, main_offset, anc_offset):
        """controlled_Ua(n, a, exponent, N).
        Operates on qubits [ctrl] + qr_x(n) + qr_main(n) + qr_ancilla(2),
        where the inner cMULTamodN sees its qr_main as qr_main(n) +
        qr_ancilla[0] = main_offset .. main_offset+n, and its qr_ancilla
        as qr_ancilla[1] = anc_offset + 1.
        """
        a_pow = a ** exponent
        a_inv = modinv(a_pow, N)

        # qc.append(cMULTamodN_gate, qubits)  — cMULTamodN(n, a^exponent, N)
        self.cMULTamodN(n, a_pow, N,
                        ctrl, x_offset, main_offset, anc_offset + 1)

        # for i in range(n): qc.cswap(qr_ctl, qr_x[i], qr_main[i])
        for i in range(n):
            self.cswap(ctrl, x_offset + i, main_offset + i)

        # qc.append(cMULTamodN_inv_gate, qubits)  — cMULTamodN(n, a_inv, N).inverse()
        sub = _GateRecorder()
        sub.cMULTamodN(n, a_inv, N,
                       ctrl, x_offset, main_offset, anc_offset + 1)
        self.extend_inverse(sub)


############### Top-level builders

def _shors_m2_qubit_layout(n):
    """m=2 layout: qr_counting(1) | qr_mult(n) | qr_aux(n+2). 2n+3 qubits.
    The single counting qubit is qubit 0 and is reused 2n times."""
    return {
        "ctrl": 0,
        "x_offset": 1,
        "main_offset": 1 + n,
        "anc_offset": 1 + n + n,
        "num_qubits": 2 * n + 3,
    }


def build_shors_m2_iteration_arrays(n, base, number):
    """Per-iteration controlled_Ua gate sequences for Shor's m=2.

    Mirrors the qiskit reference's loop body:

        for k in range(2*n):
            cUa_gate = controlled_Ua(n, base, 2**(2*n-1-k), number)
            qubits = [qr_counting[0]] + qr_mult + qr_aux
            qc.append(cUa_gate, qubits)

    Each iteration's controlled_Ua records are concatenated into one flat
    gate sequence; `offsets[k]..offsets[k+1]` gives iteration k's slice.
    The kernel walks 2*n iterations, applying each iteration's slice plus
    the dynamic-circuit feedback (h, conditional reset, conditional phase
    rotations from past measurements, h, mz) in between.
    """
    layout = _shors_m2_qubit_layout(n)
    rec = _GateRecorder()
    offsets = [0]

    for k in range(2 * n):
        rec.controlled_Ua(n, int(base), 2 ** (2 * n - 1 - k), int(number),
                          layout["ctrl"], layout["x_offset"],
                          layout["main_offset"], layout["anc_offset"])
        offsets.append(len(rec.op_kind))

    return {
        "op_kind": rec.op_kind,
        "q1": rec.q1,
        "q2": rec.q2,
        "q3": rec.q3,
        "angle": rec.angle,
        "offsets": offsets,
        "n": n,
        "num_qubits": layout["num_qubits"],
    }


def build_shors_m1_arrays(number, base):
    """Build the gate-record arrays for the full Shor's m=1 circuit.

    Mirrors qiskit's `ShorsAlgorithm(number, base, method=1)`: 4n+2 qubits
    laid out as qr_counting(2n) | qr_mult(n) | qr_aux(n+2). Returns the
    parallel arrays expected by the cudaq kernel plus the list of
    measured qubit indices in qiskit order (cr_data[0]..cr_data[2n-1]).
    """
    n = int(math.ceil(math.log(number, 2)))

    counting_offset = 0
    mult_offset = 2 * n
    aux_offset = 3 * n

    rec = _GateRecorder()

    # qc.h(qr_counting)
    for i in range(2 * n):
        rec.h(counting_offset + i)
    # qc.x(qr_mult[0])
    rec.x(mult_offset)

    # for i in reversed(range(2*n)):
    #     cUa = controlled_Ua(n, base, 2^(2n-1-i), number)
    #     qubits = [qr_counting[i]] + qr_mult + qr_aux
    for i in range(2 * n - 1, -1, -1):
        ctrl = counting_offset + i
        x_off = mult_offset
        main_off = aux_offset
        anc_off = aux_offset + n
        rec.controlled_Ua(n, int(base), 2 ** (2 * n - 1 - i),
                          int(number),
                          ctrl, x_off, main_off, anc_off)

    # qc.append(inv_qft_gate(2*n), qr_counting)
    rec.qft([counting_offset + i for i in range(2 * n)], inverse=True)

    # qc.measure(qr_counting, cr_data) — measured[k] = qr_counting[k] = counting_offset + k
    measured = [counting_offset + i for i in range(2 * n)]

    return {
        "op_kind": rec.op_kind,
        "q1": rec.q1,
        "q2": rec.q2,
        "q3": rec.q3,
        "angle": rec.angle,
        "measured": measured,
        "num_qubits": 4 * n + 2,
        "n": n,
    }


############### Analyzer (qiskit-free)

def expected_shor_dist(num_bits, order, num_shots):
    """Return the exact finite-register order-finding distribution.

    The counting register contains ``2 * num_bits`` qubits.  When its
    dimension is not divisible by the order, probability is spread around
    each ideal phase rather than concentrated at ``floor(Q * s / r)``.
    """
    qubits_measured = 2 * num_bits
    r = int(order)
    if r < 1:
        raise ValueError(f"order must be positive, got {order}")

    q = 1 << qubits_measured
    short_length, long_count = divmod(q, r)
    outcomes = np.arange(q, dtype=np.int64)
    residues = (r * outcomes) % q
    half_angles = np.pi * residues / q
    denominators = np.sin(half_angles)

    def geometric_magnitude(length):
        """Squared magnitude of sum(exp(2πi*j*r*y/Q), j=0..length-1)."""
        magnitudes = np.empty(q, dtype=np.float64)
        exact_peak = residues == 0
        exact_zero = (~exact_peak) & (((length * residues) % q) == 0)
        regular = ~(exact_peak | exact_zero)

        magnitudes[exact_peak] = float(length * length)
        magnitudes[exact_zero] = 0.0
        magnitudes[regular] = (
            np.sin(length * half_angles[regular]) / denominators[regular]
        ) ** 2
        return magnitudes

    probabilities = (
        long_count * geometric_magnitude(short_length + 1)
        + (r - long_count) * geometric_magnitude(short_length)
    ) / float(q * q)

    # Remove accumulated floating-point normalization error before metrics
    # compares this distribution with sampled counts.
    probabilities /= probabilities.sum()
    scale = float(num_shots)
    return {
        format(outcome, f"0{qubits_measured}b"): float(probability * scale)
        for outcome, probability in enumerate(probabilities)
        if probability > 0.0
    }


def analyze_and_print_result(qc, result, num_qubits, num_shots,
                             order=None, method=None, verbose=False):
    """Compute fidelity vs the analytical distribution.

    Mirrors shors/qiskit/shors_benchmark.analyze_and_print_result, but the
    cudaq path's count keys never carry the qiskit multi-classical-register
    `"{cr_aux} {cr_data}"` prefix — so the `key[2:]` strip in qiskit's
    method-2 branch is omitted here. Method-2 callers in the cudaq path
    can instead call `expected_shor_dist + metrics.polarization_fidelity`
    directly on the count dict.
    """
    if method == 1:
        num_bits = int((num_qubits - 2) / 4)
    elif method == 2:
        num_bits = int((num_qubits - 3) / 2)
    elif method == 3:
        num_bits = int((num_qubits - 2) / 2)
    else:
        raise ValueError(f"unknown method {method}")

    counts = result.get_counts(qc)
    correct_dist = expected_shor_dist(num_bits, order, num_shots)

    if verbose:
        print(f"For order value {order}, measured: {counts}")
        print(f"For order value {order}, correct_dist: {correct_dist}")

    fidelity = metrics.polarization_fidelity(counts, correct_dist)
    return counts, fidelity
