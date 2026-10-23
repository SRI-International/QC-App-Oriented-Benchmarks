"""Framework-agnostic QUOPS benchmark functions.

Extracted from the CUDA-Q ``src.py`` so that Qiskit and CUDA-Q kernels share
identical math, statistics, plotting, and circuit-array logic.  Nothing in this
file imports any quantum framework.

Equation and section numbers refer to the QUOPS paper and Supplementary Material.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd


# ===========================================================================
# Constants and gate conventions
# ===========================================================================

RX, RY, RZ, X, Y, Z, CX, U3 = range(8)

ROTATION_GATES = (RX, RY, RZ)
PAULI_GATES = (X, Y, Z)
TWO_QUBIT_GATES = (CX,)

GATE_NAMES = {
    RX: "rx", RY: "ry", RZ: "rz",
    X: "x", Y: "y", Z: "z",
    CX: "cx", U3: "u3",
}

# Pauli indices: 0=I, 1=Z, 2=X, 3=Y (low bit = Z, high bit = X).
PAULI_I, PAULI_Z, PAULI_X, PAULI_Y = range(4)
PAULI_GATE_ID = (None, Z, X, Y)

_PAULI_MATRICES = (
    np.eye(2, dtype=complex),
    np.diag([1.0, -1.0]).astype(complex),
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
)


# ===========================================================================
# Matrix helpers
# ===========================================================================


def pauli_matrix(index: int) -> np.ndarray:
    """Return the 2x2 matrix for a Pauli index (0=I, 1=Z, 2=X, 3=Y)."""
    return _PAULI_MATRICES[int(index)]


def compose_paulis(left: int, right: int) -> int:
    """Return the Pauli index of ``left * right``, ignoring the global phase."""
    left, right = int(left), int(right)
    z = (left & 1) ^ (right & 1)
    x = (left >> 1) ^ (right >> 1)
    return x << 1 | z


def conjugate_pauli_through_cnot(control: int, target: int) -> tuple[int, int]:
    """Return ``CX (P_c x P_t) CX`` as a pair of Pauli indices.

    Under conjugation by CNOT: ``X_c -> X_c X_t``, ``Z_t -> Z_c Z_t``, while
    ``Z_c`` and ``X_t`` are unchanged.
    """
    z_c, x_c = control & 1, control >> 1
    z_t, x_t = target & 1, target >> 1
    out_control = x_c << 1 | (z_c ^ z_t)
    out_target = (x_c ^ x_t) << 1 | z_t
    return out_control, out_target


def rz_matrix(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def ry_matrix(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle / 2.0), math.sin(angle / 2.0)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=complex)


def rx_matrix(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle / 2.0), math.sin(angle / 2.0)
    return np.array([[cosine, -1j * sine], [-1j * sine, cosine]], dtype=complex)


def u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """Return ``Rz(phi) Ry(theta) Rz(lam)``, CUDA-Q's ``u3`` up to global phase."""
    return rz_matrix(phi) @ ry_matrix(theta) @ rz_matrix(lam)


def single_qubit_matrix(gate: int, angle: float, phi: float = 0.0, lam: float = 0.0):
    """Return the 2x2 matrix for any single-qubit gate id in this module."""
    if gate == RX:
        return rx_matrix(angle)
    if gate == RY:
        return ry_matrix(angle)
    if gate == RZ:
        return rz_matrix(angle)
    if gate == X:
        return pauli_matrix(PAULI_X)
    if gate == Y:
        return pauli_matrix(PAULI_Y)
    if gate == Z:
        return pauli_matrix(PAULI_Z)
    if gate == U3:
        return u3_matrix(angle, phi, lam)
    raise ValueError(f"not a single-qubit gate id: {gate}")


def u3_angles_from_matrix(unitary: np.ndarray) -> tuple[float, float, float]:
    """Decompose a 2x2 unitary into ``(theta, phi, lam)`` up to global phase.

    Inverse of :func:`u3_matrix`, so ``u3_matrix(*u3_angles_from_matrix(U))``
    equals ``U`` up to a global phase.
    """
    unitary = np.asarray(unitary, dtype=complex)
    a, b = unitary[0, 0], unitary[0, 1]
    c, d = unitary[1, 0], unitary[1, 1]

    theta = 2.0 * math.atan2(abs(c), abs(a))

    if abs(a) < 1e-12:
        return theta, float(np.angle(c) - np.angle(-b)), 0.0
    if abs(c) < 1e-12:
        return theta, float(np.angle(d) - np.angle(a)), 0.0

    phi = float(np.angle(c) - np.angle(a))
    lam = float(np.angle(-b) - np.angle(a))
    return theta, phi, lam


# ===========================================================================
# Array-level circuit operations
# ===========================================================================


def circuit_size(gates: Iterable[int]) -> int:
    """Return the QUOPS size: one per single-qubit gate, two per CNOT."""
    return sum(2 if gate in TWO_QUBIT_GATES else 1 for gate in gates)


def adjoint_arrays(gates, q0s, q1s, angles, phis=None, lams=None):
    """Return adjoint gate arrays: reversed order with inverted rotations.

    Returns ``(gates, q0s, q1s, angles, phis, lams)``.
    """
    n = len(gates)
    if phis is None:
        phis = [0.0] * n
    if lams is None:
        lams = [0.0] * n
    out_gates, out_q0, out_q1, out_angles, out_phis, out_lams = [], [], [], [], [], []
    for i in reversed(range(n)):
        gate = gates[i]
        out_gates.append(gate)
        out_q0.append(q0s[i])
        out_q1.append(q1s[i])
        if gate in ROTATION_GATES:
            out_angles.append(-angles[i])
            out_phis.append(phis[i])
            out_lams.append(lams[i])
        elif gate == U3:
            out_angles.append(-angles[i])
            out_phis.append(-lams[i])
            out_lams.append(-phis[i])
        else:
            out_angles.append(angles[i])
            out_phis.append(phis[i])
            out_lams.append(lams[i])
    return out_gates, out_q0, out_q1, out_angles, out_phis, out_lams


def concatenate_arrays(*array_tuples):
    """Concatenate multiple ``(gates, q0s, q1s, angles, phis, lams, width)`` tuples.

    All must have the same width.  Returns ``(gates, q0s, q1s, angles, phis, lams, width)``.
    """
    if not array_tuples:
        raise ValueError("at least one array tuple is required")
    width = array_tuples[0][6]
    if any(entry[6] != width for entry in array_tuples[1:]):
        raise ValueError("all array tuples must have the same width")
    gates, q0s, q1s, angles, phis, lams = [], [], [], [], [], []
    for entry in array_tuples:
        gates.extend(entry[0])
        q0s.extend(entry[1])
        q1s.extend(entry[2])
        angles.extend(entry[3])
        phis.extend(entry[4])
        lams.extend(entry[5])
    return gates, q0s, q1s, angles, phis, lams, width


# ===========================================================================
# Randomized compilation (array-level)
# ===========================================================================

RC_STYLES = ("pauli_frame", "explicit_paulis")

_module_rng = np.random.default_rng()


def _resolve_rng(seed):
    return _module_rng if seed is None else np.random.default_rng(seed)


def _emit_pauli_arrays(index, qubit, gates, q0s, q1s, thetas, phis, lams):
    gate = PAULI_GATE_ID[int(index)]
    if gate is not None:
        gates.append(gate)
        q0s.append(int(qubit))
        q1s.append(0)
        thetas.append(0.0)
        phis.append(0.0)
        lams.append(0.0)


def pauli_frame_rewrite_arrays(gates, q0s, q1s, angles, phis, lams, width, rng):
    """Apply pauli-frame randomized compilation at the array level.

    Returns ``(out_gates, out_q0s, out_q1s, out_thetas, out_phis, out_lams)``.
    """
    n = len(gates)

    first_gate_is_cnot = [False] * width
    seen = [False] * width
    cnot_follows = [False] * n
    for index, gate in enumerate(gates):
        touched = (q0s[index], q1s[index]) if gate == CX else (q0s[index],)
        for qubit in touched:
            if not seen[qubit]:
                seen[qubit] = True
                first_gate_is_cnot[qubit] = gate == CX
    pending = [False] * width
    for index in reversed(range(n)):
        gate = gates[index]
        if gate == CX:
            cnot_follows[index] = True
            pending[q0s[index]] = True
            pending[q1s[index]] = True
        else:
            cnot_follows[index] = pending[q0s[index]]

    out_gates, out_q0, out_q1, out_theta, out_phi, out_lam = [], [], [], [], [], []
    frame = [PAULI_I] * width

    for qubit in range(width):
        if first_gate_is_cnot[qubit]:
            frame[qubit] = int(rng.integers(0, 4))
            _emit_pauli_arrays(
                frame[qubit], qubit, out_gates, out_q0, out_q1, out_theta, out_phi, out_lam
            )

    for index, gate in enumerate(gates):
        if gate == CX:
            control, target = q0s[index], q1s[index]
            out_gates.append(CX)
            out_q0.append(control)
            out_q1.append(target)
            out_theta.append(0.0)
            out_phi.append(0.0)
            out_lam.append(0.0)
            frame[control], frame[target] = conjugate_pauli_through_cnot(
                frame[control], frame[target]
            )
            continue

        qubit = q0s[index]
        new_frame = int(rng.integers(0, 4)) if cnot_follows[index] else PAULI_I
        matrix = (
            pauli_matrix(new_frame)
            @ single_qubit_matrix(gate, angles[index], phis[index], lams[index])
            @ pauli_matrix(frame[qubit])
        )
        theta, phi, lam = u3_angles_from_matrix(matrix)
        out_gates.append(U3)
        out_q0.append(qubit)
        out_q1.append(0)
        out_theta.append(theta)
        out_phi.append(phi)
        out_lam.append(lam)
        frame[qubit] = new_frame

    for qubit in range(width):
        _emit_pauli_arrays(
            frame[qubit], qubit, out_gates, out_q0, out_q1, out_theta, out_phi, out_lam
        )

    return out_gates, out_q0, out_q1, out_theta, out_phi, out_lam


def explicit_pauli_rewrite_arrays(gates, q0s, q1s, angles, phis, lams, width, rng):
    """Apply explicit-Pauli randomized compilation at the array level.

    Returns ``(out_gates, out_q0s, out_q1s, out_thetas, out_phis, out_lams)``.
    """
    out_gates, out_q0, out_q1, out_theta, out_phi, out_lam = [], [], [], [], [], []

    for index, gate in enumerate(gates):
        if gate != CX:
            out_gates.append(gate)
            out_q0.append(q0s[index])
            out_q1.append(q1s[index])
            out_theta.append(angles[index])
            out_phi.append(phis[index])
            out_lam.append(lams[index])
            continue

        control, target = q0s[index], q1s[index]
        pre_control = int(rng.integers(0, 4))
        pre_target = int(rng.integers(0, 4))
        post_control, post_target = conjugate_pauli_through_cnot(pre_control, pre_target)

        _emit_pauli_arrays(pre_control, control, out_gates, out_q0, out_q1, out_theta, out_phi, out_lam)
        _emit_pauli_arrays(pre_target, target, out_gates, out_q0, out_q1, out_theta, out_phi, out_lam)
        out_gates.append(CX)
        out_q0.append(control)
        out_q1.append(target)
        out_theta.append(0.0)
        out_phi.append(0.0)
        out_lam.append(0.0)
        _emit_pauli_arrays(post_control, control, out_gates, out_q0, out_q1, out_theta, out_phi, out_lam)
        _emit_pauli_arrays(post_target, target, out_gates, out_q0, out_q1, out_theta, out_phi, out_lam)

    return out_gates, out_q0, out_q1, out_theta, out_phi, out_lam


def rc_rewrite_arrays(gates, q0s, q1s, angles, width, phis=None, lams=None,
                      seed=None, style="pauli_frame"):
    """Randomized-compilation rewrite at the array level.

    Returns ``(out_gates, out_q0s, out_q1s, out_thetas, out_phis, out_lams)``.
    """
    if style not in RC_STYLES:
        raise ValueError(f"style must be one of {RC_STYLES}, got {style!r}")
    n = len(gates)
    if phis is None:
        phis = [0.0] * n
    if lams is None:
        lams = [0.0] * n
    rng = _resolve_rng(seed)
    if style == "pauli_frame":
        return pauli_frame_rewrite_arrays(gates, q0s, q1s, angles, phis, lams, width, rng)
    return explicit_pauli_rewrite_arrays(gates, q0s, q1s, angles, phis, lams, width, rng)


def compiled_gate_counts(gates) -> dict[str, int]:
    """Return one- and two-qubit gate counts, useful for reporting overhead."""
    two_qubit = sum(1 for gate in gates if gate == CX)
    return {
        "one_qubit": len(gates) - two_qubit,
        "two_qubit": two_qubit,
        "total": len(gates),
    }


# ===========================================================================
# QUOPS circuit ensemble
# ===========================================================================


def final_layer_qubits(width: int, zeta: float) -> int:
    """Return ``round(zeta * w)``, the number of qubits in the final layer."""
    return int(round(float(zeta) * int(width)))


def shape_size(width: int, depth: int, zeta: float) -> int:
    """Return the QUOPS size ``s`` of a ``(w, d, zeta)`` circuit.

    Closed form of Supplementary Section II B::

        s_bulk  = (2 * floor(w / 2) + w) * (d - 1)
        s_final = 2 * floor(zeta * w / 2) + zeta * w
    """
    width = int(width)
    depth = int(depth)
    final = final_layer_qubits(width, zeta)
    bulk = (2 * (width // 2) + width) * (depth - 1)
    return bulk + 2 * (final // 2) + final


def sample_quops_arrays(width: int, depth: int, zeta: float, seed=None):
    """Sample a QUOPS circuit as flat gate arrays.

    Returns ``(gates, q0, q1, angles, size)``.
    """
    width = int(width)
    depth = int(depth)
    if depth < 1:
        raise ValueError("benchmark depth d must be >= 1")
    final_active = final_layer_qubits(width, zeta)
    if not 1 <= final_active <= width:
        raise ValueError(
            f"zeta={zeta} gives {final_active} final-layer qubits; must be in 1..{width}"
        )

    rng = np.random.default_rng(seed)
    gates, q0, q1, angles = [], [], [], []
    for layer in range(depth):
        if layer < depth - 1:
            active = list(range(width))
        else:
            active = sorted(
                int(qubit) for qubit in rng.choice(width, size=final_active, replace=False)
            )
        perm = list(rng.permutation(active))
        for i in range(0, len(perm) - 1, 2):
            gates.append(CX)
            q0.append(int(perm[i]))
            q1.append(int(perm[i + 1]))
            angles.append(0.0)
        for qubit in active:
            gates.append(ROTATION_GATES[int(rng.integers(0, 3))])
            q0.append(int(qubit))
            q1.append(0)
            angles.append(float(rng.uniform(0.0, 2.0 * math.pi)))

    return gates, q0, q1, angles, circuit_size(gates)


def legal_zetas(width: int) -> tuple[float, ...]:
    """Return the admissible filling fractions ``{1/w, 2/w, ..., 1}``."""
    width = int(width)
    return tuple(k / width for k in range(1, width + 1))


def enumerate_shapes(
    width_range: Sequence[int],
    zeta: float | None = None,
    *,
    max_depth: int = 10_000,
) -> list[dict]:
    """List ``(width, depth, zeta, size)`` shapes inside the utility cone.

    ``width_range`` is ``(min_width, max_width)``, inclusive at both ends.
    The cone is ``w**2 <= s <= w**3`` (Supplementary Eq. 13).
    """
    low, high = int(width_range[0]), int(width_range[1])
    if low < 1:
        raise ValueError("widths must be at least one")
    shapes = []
    for width in range(low, high + 1):
        zetas = legal_zetas(width) if zeta is None else (float(zeta),)
        seen: dict[int, dict] = {}
        for depth in range(1, max_depth + 1):
            sizes_this_depth = []
            for value in zetas:
                size = shape_size(width, depth, value)
                sizes_this_depth.append(size)
                if width**2 <= size <= width**3 and size not in seen:
                    seen[size] = {
                        "width": width,
                        "depth": depth,
                        "zeta": value,
                        "size": size,
                    }
            if sizes_this_depth and min(sizes_this_depth) > width**3:
                break
        shapes.extend(seen[size] for size in sorted(seen))
    return shapes


# ===========================================================================
# MCFE caps: Clifford constants and array-level cap logic
# ===========================================================================

_HPI = math.pi / 2.0
SINGLE_QUBIT_CLIFFORD_EULER = (
    (0.0, 0.0, 0.0),
    (_HPI, 0.0, _HPI),
    (_HPI, _HPI, -math.pi),
    (math.pi, -math.pi, 0.0),
    (_HPI, -math.pi, -_HPI),
    (_HPI, -_HPI, math.pi),
    (math.pi, -_HPI, -_HPI),
    (_HPI, -math.pi, _HPI),
    (_HPI, -_HPI, 0.0),
    (0.0, 0.0, math.pi),
    (_HPI, 0.0, -_HPI),
    (_HPI, _HPI, 0.0),
    (_HPI, 0.0, math.pi),
    (_HPI, _HPI, -_HPI),
    (0.0, 0.0, _HPI),
    (_HPI, math.pi, -math.pi),
    (_HPI, -_HPI, _HPI),
    (math.pi, math.pi, -_HPI),
    (_HPI, -math.pi, 0.0),
    (_HPI, -_HPI, -_HPI),
    (math.pi, 3.0 * _HPI, -math.pi),
    (_HPI, 0.0, 0.0),
    (_HPI, _HPI, -3.0 * _HPI),
    (0.0, 0.0, -_HPI),
)

NUM_ONE_QUBIT_CLIFFORDS = len(SINGLE_QUBIT_CLIFFORD_EULER)

_CLIFFORD_MATRICES = tuple(
    u3_matrix(*euler) for euler in SINGLE_QUBIT_CLIFFORD_EULER
)


def clifford_matrix(index: int) -> np.ndarray:
    """Return the 2x2 matrix of the indexed one-qubit Clifford."""
    return _CLIFFORD_MATRICES[int(index)]


def _matching_clifford_index(unitary: np.ndarray) -> int:
    overlaps = [
        abs(np.trace(candidate.conj().T @ unitary)) / 2.0
        for candidate in _CLIFFORD_MATRICES
    ]
    index = int(np.argmax(overlaps))
    if overlaps[index] < 1.0 - 1e-10:
        raise RuntimeError("unitary does not match a one-qubit Clifford")
    return index


_FINAL_CAP_CLIFFORD_INDEX = tuple(
    tuple(
        _matching_clifford_index(pauli_matrix(pauli) @ clifford.conj().T)
        for pauli in range(4)
    )
    for clifford in _CLIFFORD_MATRICES
)


def target_bitstring_from_paulis(pauli_indices: Iterable[int]) -> str:
    """Return the q0-first bit string produced when ``P`` acts on ``|0...0>``."""
    indices = tuple(int(index) for index in pauli_indices)
    if not indices:
        raise ValueError("a Pauli layer must contain at least one qubit")
    if any(not 0 <= index < 4 for index in indices):
        raise ValueError("Pauli indices must use 0=I, 1=Z, 2=X, 3=Y")
    return "".join("1" if index in (2, 3) else "0" for index in indices)


def clifford_layer_arrays(clifford_indices):
    """Build gate arrays for a tensor product of indexed one-qubit Cliffords.

    Returns ``(gates, q0s, q1s, thetas, phis, lams, width)``.
    """
    indices = tuple(int(index) for index in clifford_indices)
    if not indices:
        raise ValueError("a Clifford layer must contain at least one qubit")
    if any(not 0 <= index < NUM_ONE_QUBIT_CLIFFORDS for index in indices):
        raise ValueError("one-qubit Clifford indices must be in 0..23")

    gates, q0s, q1s, thetas, phis, lams = [], [], [], [], [], []
    for qubit, index in enumerate(indices):
        theta, phi, lam = SINGLE_QUBIT_CLIFFORD_EULER[index]
        gates.append(U3)
        q0s.append(qubit)
        q1s.append(0)
        thetas.append(theta)
        phis.append(phi)
        lams.append(lam)

    return gates, q0s, q1s, thetas, phis, lams, len(indices)


def final_cap_clifford_indices(clifford_indices, pauli_indices):
    """Compute the Clifford indices for L' = P L^dag.

    Returns a tuple of Clifford indices, one per qubit.
    """
    clifford_indices = tuple(int(i) for i in clifford_indices)
    pauli_indices = tuple(int(i) for i in pauli_indices)
    if len(clifford_indices) != len(pauli_indices):
        raise ValueError("Clifford and Pauli layers must have the same width")
    return tuple(
        _FINAL_CAP_CLIFFORD_INDEX[clifford][pauli]
        for clifford, pauli in zip(clifford_indices, pauli_indices)
    )


def sample_mcfe_cap_indices(width: int, seed=None):
    """Sample random Clifford and Pauli indices for one MCFE cap set.

    Returns ``(clifford_indices, pauli_indices, final_clifford_indices, target)``.
    """
    width = int(width)
    if width < 1:
        raise ValueError("MCFE cap width must be at least one")
    rng = np.random.default_rng(seed)
    cliffords = tuple(int(v) for v in rng.integers(0, NUM_ONE_QUBIT_CLIFFORDS, size=width))
    paulis = tuple(int(v) for v in rng.integers(0, 4, size=width))
    final = final_cap_clifford_indices(cliffords, paulis)
    target = target_bitstring_from_paulis(paulis)
    return cliffords, paulis, final, target


# ===========================================================================
# Statistics
# ===========================================================================

QUOPS_THRESHOLD = 1.0 / math.sqrt(math.e)
DEFAULT_FAMILYWISE_ALPHA = 0.05
MCFE_ESTIMATORS = ("eq45", "eq42")


def hamming_distance(a: str, b: str) -> int:
    """Compute the Hamming distance between two bit strings."""
    a = str(a).replace(" ", "")
    b = str(b).replace(" ", "")
    if len(a) != len(b):
        raise ValueError(f"bit strings must have the same length: {a!r}, {b!r}")
    return sum(x != y for x, y in zip(a, b))


def effective_polarization_from_counts(
    counts: dict[str, int], width: int, target: str | None = None
) -> float:
    """Return the effective polarization of Supplementary Eqs. (33) and (35).

    For target bit string ``t`` and counts ``N_b``::

        m     = (1 / N) * sum_b N_b * (-1/2) ** HammingDistance(b, t)
        gamma = (4**n * m - 1) / (4**n - 1)
    """
    if target is None:
        target = "0" * width
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise ValueError("counts must contain at least one shot")

    moment = 0.0
    for bitstring, count in counts.items():
        moment += int(count) * (-0.5) ** hamming_distance(bitstring, target)
    moment /= total

    dimension_sq = 4**width
    return dimension_sq / (dimension_sq - 1.0) * moment - 1.0 / (dimension_sq - 1.0)


def normalized_mcfe_polarization(
    gamma_br: float,
    gamma_rr: float,
    gamma_ref: float,
    *,
    min_denominator: float = 1.0e-12,
) -> float:
    """Combine the three mirror ensembles as in Supplementary Eq. (36).

        lambda = gamma(M1) / (sqrt(gamma(M2)) * sqrt(gamma(M3)))

    Returns ``nan`` when either denominator factor is non-positive or too small.
    """
    gamma_br = float(gamma_br)
    gamma_rr = float(gamma_rr)
    gamma_ref = float(gamma_ref)
    min_denominator = float(min_denominator)
    if min_denominator < 0.0:
        raise ValueError("min_denominator must be non-negative")
    if (
        not all(np.isfinite(value) for value in (gamma_br, gamma_rr, gamma_ref))
        or gamma_rr <= min_denominator
        or gamma_ref <= min_denominator
    ):
        return float("nan")
    return float(gamma_br / math.sqrt(gamma_rr * gamma_ref))


def _finite_mean(values) -> float:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(array.mean()) if array.size else float("nan")


def mcfe_polarization(br_by_circuit, rr_by_circuit, ref_samples, estimator: str = "eq45") -> float:
    """Estimate ``lambda_bar(w, s)`` from per-circuit mirror polarizations.

    ``"eq45"`` averages each ensemble before taking the ratio (Supplementary
    Eq. 45).  ``"eq42"`` takes the ratio per circuit and then averages
    (Supplementary Eq. 42).
    """
    if estimator not in MCFE_ESTIMATORS:
        raise ValueError(f"estimator must be one of {MCFE_ESTIMATORS}, got {estimator!r}")

    gamma_ref = _finite_mean(ref_samples)
    if not np.isfinite(gamma_ref) or gamma_ref <= 0.0:
        return float("nan")

    if estimator == "eq45":
        gamma_br = _finite_mean([value for circuit in br_by_circuit for value in circuit])
        gamma_rr = _finite_mean([value for circuit in rr_by_circuit for value in circuit])
        return normalized_mcfe_polarization(gamma_br, gamma_rr, gamma_ref)

    per_circuit = [
        normalized_mcfe_polarization(_finite_mean(br), _finite_mean(rr), gamma_ref)
        for br, rr in zip(br_by_circuit, rr_by_circuit)
    ]
    return _finite_mean(per_circuit)


def mean_polarization(polarizations: Iterable[float]) -> tuple[float, float]:
    """Return the sample mean and standard error of a set of polarizations."""
    values = np.asarray([p for p in polarizations if np.isfinite(p)], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    stderr = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
    return float(values.mean()), stderr


# ---------------------------------------------------------------------------
# Shape-level data and confidence bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapePolarizationData:
    """Raw mirror polarizations collected at one circuit shape.

    ``br`` and ``rr`` hold one tuple of per-mirror polarizations per sampled
    QUOPS circuit; ``ref`` is the width-``w`` M3 ensemble.
    """

    width: int
    size: int
    depth: int
    zeta: float
    br: tuple[tuple[float, ...], ...]
    rr: tuple[tuple[float, ...], ...]
    ref: tuple[float, ...]
    metadata: dict = field(default_factory=dict)

    @property
    def num_circuits(self) -> int:
        return len(self.br)

    def estimate(self, estimator: str = "eq45") -> float:
        return mcfe_polarization(self.br, self.rr, self.ref, estimator=estimator)


def bootstrap_polarization(
    data: ShapePolarizationData,
    *,
    estimator: str = "eq45",
    resamples: int = 2000,
    seed=None,
) -> np.ndarray:
    """Resample circuits, mirrors, and the M3 ensemble to get a bootstrap sample."""
    rng = np.random.default_rng(seed)
    num_circuits = data.num_circuits
    if num_circuits == 0 or not data.ref:
        return np.array([], dtype=float)

    ref = np.asarray(data.ref, dtype=float)
    draws = np.empty(int(resamples), dtype=float)
    for index in range(int(resamples)):
        circuit_choice = rng.integers(0, num_circuits, size=num_circuits)
        br_resampled, rr_resampled = [], []
        for circuit in circuit_choice:
            for source, sink in ((data.br[circuit], br_resampled), (data.rr[circuit], rr_resampled)):
                values = np.asarray(source, dtype=float)
                if values.size == 0:
                    sink.append(())
                else:
                    sink.append(tuple(values[rng.integers(0, values.size, size=values.size)]))
        ref_resampled = ref[rng.integers(0, ref.size, size=ref.size)]
        draws[index] = mcfe_polarization(
            br_resampled, rr_resampled, ref_resampled, estimator=estimator
        )
    return draws


def one_sided_z_value(alpha: float) -> float:
    """Return the normal z value for a one-sided test at significance ``alpha``."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return float(NormalDist().inv_cdf(1.0 - alpha))


def one_sided_lower_bound(
    mean: float, stderr: float, *, alpha: float = DEFAULT_FAMILYWISE_ALPHA
) -> float:
    """Return a normal-approximation one-sided lower confidence bound."""
    if not (np.isfinite(mean) and np.isfinite(stderr)):
        return float("nan")
    return float(mean - one_sided_z_value(alpha) * stderr)


def bootstrap_lower_bound(draws, *, alpha: float = DEFAULT_FAMILYWISE_ALPHA) -> float:
    """Return the one-sided lower bound as the ``alpha`` quantile of the draws."""
    values = np.asarray([value for value in draws if np.isfinite(value)], dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, float(alpha)))


def pass_statistics(
    data: ShapePolarizationData,
    *,
    estimator: str = "eq45",
    threshold: float = QUOPS_THRESHOLD,
    alpha: float = DEFAULT_FAMILYWISE_ALPHA,
    resamples: int = 2000,
    seed=None,
) -> dict:
    """Test a shape against the QUOPS threshold with a bootstrap lower bound."""
    point = data.estimate(estimator)
    other = "eq42" if estimator == "eq45" else "eq45"
    draws = bootstrap_polarization(
        data, estimator=estimator, resamples=resamples, seed=seed
    )
    lower = bootstrap_lower_bound(draws, alpha=alpha)

    per_circuit = [
        normalized_mcfe_polarization(_finite_mean(br), _finite_mean(rr), _finite_mean(data.ref))
        for br, rr in zip(data.br, data.rr)
    ]
    naive_mean, naive_stderr = mean_polarization(per_circuit)

    return {
        "width": data.width,
        "size": data.size,
        "depth": data.depth,
        "zeta": data.zeta,
        "num_circuits": data.num_circuits,
        "estimator": estimator,
        "mean_polarization": point,
        "mean_polarization_alt": data.estimate(other),
        "threshold": threshold,
        "alpha": alpha,
        "bootstrap_lower": lower,
        "bootstrap_stderr": float(np.std(draws)) if draws.size else float("nan"),
        "normal_lower": one_sided_lower_bound(naive_mean, naive_stderr, alpha=alpha),
        "stderr": naive_stderr,
        "passes_threshold": bool(np.isfinite(point) and point >= threshold),
        "passes": bool(np.isfinite(lower) and lower >= threshold),
    }


# ===========================================================================
# Utility cone, scores, and the rate metric
# ===========================================================================


def utility_cone_bounds(width: int) -> tuple[int, int]:
    """Return ``(w**2, w**3)``, the size bounds of Supplementary Eq. (13)."""
    width = int(width)
    return width**2, width**3


def in_utility_cone(width: int, size: int) -> bool:
    """Return whether ``(w, s)`` lies in the utility cone ``w**2 <= s <= w**3``."""
    low, high = utility_cone_bounds(width)
    return low <= int(size) <= high


def quops_score_from_summary(
    summary_df: pd.DataFrame, *, pass_column: str = "passes"
) -> tuple[int, tuple[int, int] | None]:
    """Return the QUOPS score and the shape at which it is achieved.

    ``Q = max{s : (w, s) in capability region and in utility cone}``
    (Supplementary Eq. 14).
    """
    required = {"width", "size", pass_column}
    if summary_df.empty or not required.issubset(summary_df.columns):
        return 0, None

    passed = summary_df[summary_df[pass_column].astype(bool)]
    if passed.empty:
        return 0, None
    passed = passed[
        [in_utility_cone(int(row.width), int(row.size))
         for row in passed[["width", "size"]].itertuples(index=False)]
    ]
    if passed.empty:
        return 0, None

    best = passed.sort_values(["size", "width"], ascending=[False, True]).iloc[0]
    return int(best["size"]), (int(best["width"]), int(best["size"]))


def error_mitigation_overhead(threshold: float) -> float:
    """Return the sample overhead ``O* = 1 / gamma**2`` of Supplementary Eq. (18)."""
    threshold = float(threshold)
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    return 1.0 / threshold**2


def quops_rate(
    size: int,
    gamma: float,
    tau_wall: float,
    n_kept: int,
    n_reject: int = 0,
) -> float:
    """Return the QUOPS rate of Supplementary Eq. (46).

        Omega(w, s) = 2 s gamma**2 (N_kept + N_reject / 2) / tau_wall
    """
    tau_wall = float(tau_wall)
    if tau_wall <= 0.0:
        raise ValueError("tau_wall must be positive")
    if not np.isfinite(gamma):
        return float("nan")
    shots = float(n_kept) + float(n_reject) / 2.0
    return 2.0 * float(size) * float(gamma) ** 2 * shots / tau_wall


@dataclass(frozen=True)
class GateDurations:
    """Per-operation durations for a projected QUOPS rate, in seconds."""

    one_qubit: float
    two_qubit: float
    readout: float = 0.0
    reset: float = 0.0
    parallel_two_qubit: int = 1

    def circuit_time(self, one_qubit_count: int, two_qubit_count: int, width: int) -> float:
        parallel = max(1, int(self.parallel_two_qubit))
        return (
            one_qubit_count * self.one_qubit
            + math.ceil(two_qubit_count / parallel) * self.two_qubit
            + self.readout
            + self.reset
        )


def omega_projected(
    size: int,
    gamma: float,
    one_qubit_count: int,
    two_qubit_count: int,
    width: int,
    durations: GateDurations,
    n_kept: int,
    n_reject: int = 0,
) -> float:
    """Return a QUOPS rate from an explicit gate-duration model, Supp. Eq. (74)."""
    per_shot = durations.circuit_time(one_qubit_count, two_qubit_count, width)
    tau_wall = per_shot * (float(n_kept) + float(n_reject))
    if tau_wall <= 0.0:
        raise ValueError("gate durations must give a positive circuit time")
    if not np.isfinite(gamma):
        return float("nan")
    shots = float(n_kept) + float(n_reject) / 2.0
    return float(size) * float(gamma) ** 2 * shots / tau_wall


def omega_simulated(
    size: int, gamma: float, tau_wall: float, n_kept: int, n_reject: int = 0
) -> float:
    """Return the QUOPS rate measured against simulator wall-clock time."""
    return quops_rate(size, gamma, tau_wall, n_kept, n_reject)


# ===========================================================================
# Benchmark driver support
# ===========================================================================


def default_shape_schedule(shapes: Iterable[dict] | Sequence[int]) -> list[dict]:
    """Order shapes by increasing size, which is the partial order QUOPS needs."""
    shapes = list(shapes)
    if shapes and not isinstance(shapes[0], dict):
        shapes = enumerate_shapes(shapes)
    return sorted(shapes, key=lambda shape: (shape["size"], shape["width"]))


def _is_at_least(shape: dict, failed: tuple[int, int]) -> bool:
    """Return whether ``shape`` is equal to or larger than a failed shape."""
    failed_width, failed_size = failed
    return shape["width"] >= failed_width and shape["size"] >= failed_size


@dataclass
class BenchmarkResult:
    """Everything the QUOPS benchmark reports."""

    summary: pd.DataFrame
    quops: int
    shape: tuple[int, int] | None
    omega: float
    omega_note: str
    conditional_quops: int = 0
    conditional_shape: tuple[int, int] | None = None
    conditional_overhead: float = float("nan")
    consumed_significance: float = 0.0
    shape_data: list[ShapePolarizationData] = field(default_factory=list)

    def capability_region(self) -> list[tuple[int, int]]:
        """Return the measured capability region as ``(width, size)`` pairs."""
        if self.summary.empty or "passes" not in self.summary:
            return []
        passed = self.summary[self.summary["passes"].astype(bool)]
        return [(int(row.width), int(row.size)) for row in passed.itertuples(index=False)]

    def __str__(self) -> str:
        shape = f" at (w={self.shape[0]}, s={self.shape[1]})" if self.shape else ""
        lines = [f"QUOPS Q = {self.quops}{shape}", f"Omega = {self.omega:.3g} ({self.omega_note})"]
        if self.conditional_quops:
            lines.append(
                f"Conditional Q* = {self.conditional_quops} "
                f"at sample overhead {self.conditional_overhead:.3g}"
            )
        lines.append(f"significance consumed: {self.consumed_significance:.3f} of {DEFAULT_FAMILYWISE_ALPHA}")
        return "\n".join(lines)


# ===========================================================================
# Plots
# ===========================================================================


def _power_of_two_ticks(min_value: float, max_value: float) -> list[float]:
    lower = int(math.floor(math.log2(max(float(min_value), 1.0))))
    upper = int(math.ceil(math.log2(max(float(max_value), 1.0))))
    return [float(2**exponent) for exponent in range(lower, upper + 1)]


def _power_of_two_label(value: float, _position=None) -> str:
    if value <= 0 or not np.isfinite(value):
        return ""
    exponent = round(math.log2(value))
    if abs(value - 2**exponent) <= max(1e-09, 1e-09 * value):
        return f"$2^{{{exponent}}}$"
    return ""


def plot_quops_scan(
    summary_df: pd.DataFrame,
    *,
    title: str = "QUOPS capability region",
    label: str | None = None,
    pass_column: str = "passes",
    threshold: float = QUOPS_THRESHOLD,
    jitter_seed: int = 123,
    shape_figsize: tuple[float, float] = (9.0, 4.8),
    polarization_figsize: tuple[float, float] = (9.0, 4.2),
    dpi: int = 140,
):
    """Plot the capability region and the polarization-versus-size diagnostic."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    required = {"width", "size", pass_column, "mean_polarization"}
    if not required.issubset(summary_df.columns):
        missing = sorted(required - set(summary_df.columns))
        raise ValueError(f"summary_df is missing required columns: {missing}")

    plot_df = summary_df.copy().sort_values(["width", "size"])
    for column in ("width", "size", "mean_polarization"):
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    error_column = "bootstrap_stderr" if "bootstrap_stderr" in plot_df.columns else "stderr"
    if error_column in plot_df.columns:
        plot_df["yerr"] = pd.to_numeric(plot_df[error_column], errors="coerce").fillna(0.0)
    else:
        plot_df["yerr"] = 0.0
    plot_df = plot_df.dropna(subset=["width", "size", "mean_polarization"])
    if plot_df.empty:
        raise ValueError("summary_df does not contain plottable rows")
    plot_df[pass_column] = plot_df[pass_column].astype(bool)

    score, best_shape = quops_score_from_summary(plot_df, pass_column=pass_column)
    x_min = 2.0 ** math.floor(math.log2(max(1.0, float(plot_df["size"].min()))))
    x_max = 2.0 ** math.ceil(math.log2(max(2.0, float(plot_df["size"].max()))))
    if x_max <= x_min:
        x_max = x_min * 2.0
    y_min = float(plot_df["width"].min()) - 0.35
    y_max = float(plot_df["width"].max()) + 0.35
    x_grid = 2.0 ** np.linspace(math.log2(x_min), math.log2(x_max), 500)

    fig_shape, ax = plt.subplots(figsize=shape_figsize, dpi=dpi)
    cone_lower = x_grid ** (1.0 / 3.0)
    cone_upper = x_grid**0.5
    ax.fill_between(
        x_grid,
        np.clip(cone_lower, y_min, y_max),
        np.clip(cone_upper, y_min, y_max),
        where=np.clip(cone_upper, y_min, y_max) > np.clip(cone_lower, y_min, y_max),
        color="#dcecf8",
        alpha=0.65,
        linewidth=0,
        label="utility cone",
    )
    ax.plot(x_grid, cone_lower, color="#3e95d4", linewidth=1.5, alpha=0.8)
    ax.plot(x_grid, cone_upper, color="#3e95d4", linewidth=1.5, alpha=0.8)

    rng = np.random.default_rng(jitter_seed)
    plot_df = plot_df.assign(
        width_plot=plot_df["width"] + rng.uniform(-0.045, 0.045, size=len(plot_df))
    )
    pass_df = plot_df[plot_df[pass_column]]
    fail_df = plot_df[~plot_df[pass_column]]

    if not pass_df.empty:
        ax.scatter(
            pass_df["size"], pass_df["width_plot"], s=20, color="#238b45", alpha=0.75,
            linewidths=0, label="pass",
        )
    if not fail_df.empty:
        ax.scatter(
            fail_df["size"], fail_df["width_plot"], s=18, color="#d73027", alpha=0.45,
            linewidths=0, label="fail",
        )
    if best_shape is not None:
        ax.scatter(
            [best_shape[1]], [best_shape[0]], s=95, facecolors="none", edgecolors="#111111",
            linewidths=1.5, label=f"Q = {score}",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(sorted(plot_df["width"].unique()))
    ax.set_xticks(_power_of_two_ticks(x_min, x_max))
    ax.xaxis.set_major_formatter(FuncFormatter(_power_of_two_label))
    ax.set_xlabel("Size, $s$")
    ax.set_ylabel("Width, $w$")
    ax.set_title(title if label is None else f"{title}\n{label}")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig_shape.tight_layout()

    fig_polarization, ax = plt.subplots(figsize=polarization_figsize, dpi=dpi)
    for width, group in plot_df.groupby("width"):
        group = group.sort_values("size")
        ax.errorbar(
            group["size"], group["mean_polarization"], yerr=group["yerr"], marker="o",
            markersize=4, linewidth=1.2, capsize=2, alpha=0.85, label=f"w={int(width)}",
        )

    ax.axhline(threshold, color="#d73027", linestyle="--", linewidth=1.2, label=r"$1/\sqrt{e}$")
    ax.set_xscale("log", base=2)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(_power_of_two_ticks(x_min, x_max))
    ax.xaxis.set_major_formatter(FuncFormatter(_power_of_two_label))
    ax.set_xlabel("Size, $s$")
    ax.set_ylabel("Mean MCFE polarization")
    ax.set_title(
        "Polarization by shape" if label is None else f"Polarization by shape\n{label}"
    )
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig_polarization.tight_layout()

    return fig_shape, fig_polarization
