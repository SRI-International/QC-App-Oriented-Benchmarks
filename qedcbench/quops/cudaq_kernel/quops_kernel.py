"""CUDA-Q-specific QUOPS kernel: kernel building, MCFE caps, randomized
compilation, noise models, and execution via cudaq.sample.

Mirrors the API of the Qiskit quops_kernel.py so that notebooks for both frameworks
are nearly identical. Extracted from the Qperf CUDAQ src.py (2259 lines).
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, ClassVar

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_common"))

from quops_fctns import (
    RX, RY, RZ, X, Y, Z, CX, U3,
    ROTATION_GATES, TWO_QUBIT_GATES,
    adjoint_arrays, rc_rewrite_arrays, RC_STYLES,
    effective_polarization_from_counts,
    clifford_layer_arrays, sample_mcfe_cap_indices,
    final_cap_clifford_indices, target_bitstring_from_paulis,
    SINGLE_QUBIT_CLIFFORD_EULER, NUM_ONE_QUBIT_CLIFFORDS,
)

import cudaq


# ===========================================================================
# Kernel construction
# ===========================================================================

# Every single-qubit gate the QUOPS kernels can emit.
ONE_QUBIT_GATE_KEYS = ("rx", "ry", "rz", "x", "y", "z", "u3")
TWO_QUBIT_GATE_KEYS = ("cx",)


def build_kernel(gates, q0, q1, angles, width, phis=None, lams=None):
    """Build a CUDA-Q kernel from flat gate arrays and attach those arrays.

    The arrays are copied so that values captured by the compiler stay
    identical to the arrays attached to the returned kernel.
    """
    frozen_gates = [int(gate) for gate in gates]
    frozen_q0 = [int(qubit) for qubit in q0]
    frozen_q1 = [int(qubit) for qubit in q1]
    frozen_angles = [float(angle) for angle in angles]
    n = len(frozen_gates)
    frozen_phis = [0.0] * n if phis is None else [float(value) for value in phis]
    frozen_lams = [0.0] * n if lams is None else [float(value) for value in lams]

    if not (
        len(frozen_q0)
        == len(frozen_q1)
        == len(frozen_angles)
        == len(frozen_phis)
        == len(frozen_lams)
        == n
    ):
        raise ValueError("gates, q0, q1, angles, phis, and lams must have the same length")

    if n == 0:
        @cudaq.kernel
        def C(q: cudaq.qview):
            pass
    else:
        @cudaq.kernel
        def C(q: cudaq.qview):
            for i in range(n):
                g = frozen_gates[i]
                if g == RX:
                    rx(frozen_angles[i], q[frozen_q0[i]])
                elif g == RY:
                    ry(frozen_angles[i], q[frozen_q0[i]])
                elif g == RZ:
                    rz(frozen_angles[i], q[frozen_q0[i]])
                elif g == X:
                    x(q[frozen_q0[i]])
                elif g == Y:
                    y(q[frozen_q0[i]])
                elif g == Z:
                    z(q[frozen_q0[i]])
                elif g == CX:
                    x.ctrl(q[frozen_q0[i]], q[frozen_q1[i]])
                elif g == U3:
                    u3(
                        frozen_angles[i],
                        frozen_phis[i],
                        frozen_lams[i],
                        q[frozen_q0[i]],
                    )

    C.gates = frozen_gates
    C.q0s = frozen_q0
    C.q1s = frozen_q1
    C.angles = frozen_angles
    C.phis = frozen_phis
    C.lams = frozen_lams
    C.width = int(width)
    return C


def require_kernel_arrays(kernel):
    """Return ``(gates, q0s, q1s, angles, phis, lams, width)`` for a built kernel."""
    if not hasattr(kernel, "gates"):
        raise ValueError("expected a kernel built by build_kernel")
    return (
        kernel.gates,
        kernel.q0s,
        kernel.q1s,
        kernel.angles,
        kernel.phis,
        kernel.lams,
        int(kernel.width),
    )


def adjoint_kernel(kernel):
    """Return the adjoint kernel: reversed gate order with inverted rotations."""
    gates, q0s, q1s, angles, phis, lams, width = require_kernel_arrays(kernel)
    out = adjoint_arrays(gates, q0s, q1s, angles, phis, lams)
    return build_kernel(out[0], out[1], out[2], out[3], width, out[4], out[5])


# Keep the name used in the CUDAQ src.py for compatibility
adjoint_C = adjoint_kernel


def randomized_compilation(kernel, seed=None, style: str = "pauli_frame"):
    """Apply randomized compilation to a kernel.

    ``"pauli_frame"`` (default): Paulis are folded into adjacent rotations as
    U3 gates, matching the paper's reference compiler specification.

    ``"explicit_paulis"``: Explicit Pauli gates around every CNOT.
    """
    gates, q0s, q1s, angles, phis, lams, width = require_kernel_arrays(kernel)
    out = rc_rewrite_arrays(gates, q0s, q1s, angles, width, phis, lams,
                            seed=seed, style=style)
    return build_kernel(out[0], out[1], out[2], out[3], width, out[4], out[5])


# ===========================================================================
# MCFE caps
# ===========================================================================


def build_clifford_layer(clifford_indices):
    """Build a tensor product of indexed one-qubit Cliffords as a CUDA-Q kernel."""
    arrays = clifford_layer_arrays(clifford_indices)
    # clifford_layer_arrays returns (gates, q0s, q1s, thetas, phis, lams, width)
    kernel = build_kernel(arrays[0], arrays[1], arrays[2], arrays[3],
                          arrays[6], arrays[4], arrays[5])
    kernel.clifford_indices = tuple(int(i) for i in clifford_indices)
    return kernel


@dataclass(frozen=True)
class MCFECaps:
    """Compiled MCFE caps and the definite outcome they induce."""

    initial: Callable[[cudaq.qview], None]    # L
    final: Callable[[cudaq.qview], None]      # L' = P L^dag
    target: str                                # expected measurement bitstring
    clifford_indices: tuple[int, ...]
    final_clifford_indices: tuple[int, ...]
    pauli_indices: tuple[int, ...]

    @property
    def width(self) -> int:
        return len(self.clifford_indices)


def build_mcfe_caps(clifford_indices, pauli_indices) -> MCFECaps:
    """Build L, L' = P L^dag, and the resulting target bit string."""
    clifford_indices = tuple(int(i) for i in clifford_indices)
    pauli_indices = tuple(int(i) for i in pauli_indices)
    if len(clifford_indices) != len(pauli_indices):
        raise ValueError("Clifford and Pauli layers must have the same width")

    final_indices = final_cap_clifford_indices(clifford_indices, pauli_indices)
    initial = build_clifford_layer(clifford_indices)
    final = build_clifford_layer(final_indices)
    target = target_bitstring_from_paulis(pauli_indices)

    return MCFECaps(
        initial=initial,
        final=final,
        target=target,
        clifford_indices=clifford_indices,
        final_clifford_indices=final_indices,
        pauli_indices=pauli_indices,
    )


def sample_mcfe_caps(width: int, seed=None) -> MCFECaps:
    """Sample independent uniform one-qubit Cliffords and Paulis for MCFE."""
    cliffords, paulis, final, target = sample_mcfe_cap_indices(width, seed)
    return build_mcfe_caps(cliffords, paulis)


# ===========================================================================
# Composite kernels (BR / RR / REF) with MCFE caps
# ===========================================================================


@cudaq.kernel
def BR_kernel(
    width: int,
    kernel_a: Callable[[cudaq.qview], None],
    kernel_b: Callable[[cudaq.qview], None],
    kernel_c: Callable[[cudaq.qview], None],
    kernel_d: Callable[[cudaq.qview], None],
):
    """M1 mirror: L' g_ref(C^dag) C L, then measure (Supp. Eq. 30)."""
    q = cudaq.qvector(width)
    kernel_a(q)
    kernel_b(q)
    kernel_c(q)
    kernel_d(q)
    mz(q)


@cudaq.kernel
def RR_kernel(
    width: int,
    kernel_a: Callable[[cudaq.qview], None],
    kernel_b: Callable[[cudaq.qview], None],
    kernel_c: Callable[[cudaq.qview], None],
    kernel_d: Callable[[cudaq.qview], None],
):
    """M2 mirror: L' g_ref(C^dag) g_ref(C) L, then measure (Supp. Eq. 31)."""
    q = cudaq.qvector(width)
    kernel_a(q)
    kernel_b(q)
    kernel_c(q)
    kernel_d(q)
    mz(q)


@cudaq.kernel
def REF_kernel(
    width: int,
    kernel_a: Callable[[cudaq.qview], None],
    kernel_b: Callable[[cudaq.qview], None],
):
    """M3 SPAM reference: L' L, then measure (Supp. Eq. 32)."""
    q = cudaq.qvector(width)
    kernel_a(q)
    kernel_b(q)
    mz(q)


# ===========================================================================
# Compile-once mirror kernel (ALIGNMENT-LOG entry 2)
# ===========================================================================


@cudaq.kernel
def mirror_kernel(
    width: int,
    gates: list[int],
    q0: list[int],
    q1: list[int],
    theta: list[float],
    phi: list[float],
    lam: list[float],
):
    """Run a whole mirror circuit given as flat gate arrays, then measure.

    Compiled once per process. The gates are runtime data, so the compiler
    cannot cancel C against C^dag (the role of atomic_quantum_region in PRAQTICE).
    Gate ids are literals (RX, RY, RZ, X, Y, Z, CX, U3 = range(8) in quops_fctns)
    because CUDA-Q 0.13 kernels can't read module globals.
    """
    q = cudaq.qvector(width)
    for i in range(len(gates)):
        gate = gates[i]
        if gate == 0:
            rx(theta[i], q[q0[i]])
        elif gate == 1:
            ry(theta[i], q[q0[i]])
        elif gate == 2:
            rz(theta[i], q[q0[i]])
        elif gate == 3:
            x(q[q0[i]])
        elif gate == 4:
            y(q[q0[i]])
        elif gate == 5:
            z(q[q0[i]])
        elif gate == 6:
            x.ctrl(q[q0[i]], q[q1[i]])
        elif gate == 7:
            u3(theta[i], phi[i], lam[i], q[q0[i]])
    mz(q)


def mirror_arrays(*parts):
    """Concatenate gate-array parts, in execution order, into mirror_kernel arguments."""
    out = ([], [], [], [], [], [])
    for part in parts:
        for column, values in zip(out, part):
            column.extend(values)
    return out


# ===========================================================================
# Execution and result handling
# ===========================================================================


def sample_result_to_counter(result) -> Counter:
    """Normalize CUDA-Q sample result formats into a plain Counter."""
    if hasattr(result, "items"):
        return Counter({str(k).replace(" ", ""): int(v) for k, v in result.items()})
    if hasattr(result, "to_dict"):
        return Counter({str(k).replace(" ", ""): int(v) for k, v in result.to_dict().items()})
    return Counter(dict(result))


def polarization_from_result(result, width: int, target: str | None = None) -> float:
    """Return the effective polarization from a CUDA-Q sample result.

    CUDA-Q returns big-endian bit strings (qubit 0 = leftmost), which matches
    the convention used by quops_fctns. No reversal needed (unlike Qiskit).
    """
    return effective_polarization_from_counts(
        dict(sample_result_to_counter(result)), width, target=target
    )


# ===========================================================================
# Mirror circuits for the shared scan loop (quops_benchmark.py)
# ===========================================================================


def build_mirror(kind: str, width: int, parts, compile_once: bool = True):
    """Build a runnable BR, RR or REF circuit from gate-array parts in execution order.

    ``parts`` are ``(gates, q0s, q1s, angles, phis, lams)`` tuples: L, C, g(C^dag), L' for
    BR and RR, or L, L' for REF. Returns ``(kernel, args)`` for ``cudaq.sample``.

    ``compile_once=True`` (default): the whole mirror goes to ``mirror_kernel`` as arrays,
    so CUDA-Q compiles once per process (ALIGNMENT-LOG entry 2). ``False``: each part
    becomes a new kernel from ``build_kernel``, passed into BR_kernel / RR_kernel /
    REF_kernel as in the PRAQTICE code, and CUDA-Q compiles every new kernel (same
    results, about 100x slower).
    """
    if compile_once:
        return mirror_kernel, (width, *mirror_arrays(*parts))
    composite = {"br": BR_kernel, "rr": RR_kernel, "ref": REF_kernel}[kind]
    kernels = [build_kernel(g, q0, q1, a, width, p, l) for g, q0, q1, a, p, l in parts]
    return composite, (width, *kernels)


def sample_counts(circuit, shots: int = 1000, noise_model=None) -> dict[str, int]:
    """Run a ``(kernel, args)`` pair with ``cudaq.sample`` and return counts keyed q0-first.

    CUDA-Q returns big-endian bit strings (qubit 0 = leftmost), which matches
    the convention used by quops_fctns. No reversal needed (unlike Qiskit).
    """
    kernel, args = circuit
    result = cudaq.sample(kernel, *args, shots_count=shots, noise_model=noise_model)
    return dict(sample_result_to_counter(result))


# ===========================================================================
# Noise models
# ===========================================================================


def make_depolarizing_noise(p_1q: float = 0.0, p_2q: float = 0.0):
    """Build a uniform depolarizing noise model over the QUOPS gate set.

    Covers ``u3`` so that pauli-frame-compiled reference circuits see noise.
    """
    noise = cudaq.NoiseModel()
    if p_1q:
        channel = cudaq.DepolarizationChannel(float(p_1q))
        for gate in ONE_QUBIT_GATE_KEYS:
            noise.add_all_qubit_channel(gate, channel)
    if p_2q:
        for gate in TWO_QUBIT_GATE_KEYS:
            noise.add_all_qubit_channel(gate, cudaq.Depolarization2(float(p_2q)))
    return noise


def asymmetric_readout_channel(p0_to_1: float, p1_to_0: float):
    """Build a channel reproducing an asymmetric readout confusion matrix."""
    k0 = np.array(
        [[math.sqrt(1.0 - p0_to_1), 0.0], [0.0, math.sqrt(1.0 - p1_to_0)]],
        dtype=np.complex128,
    )
    k1 = np.array([[0.0, math.sqrt(p1_to_0)], [0.0, 0.0]], dtype=np.complex128)
    k2 = np.array([[0.0, 0.0], [math.sqrt(p0_to_1), 0.0]], dtype=np.complex128)
    return cudaq.KrausChannel(
        [cudaq.KrausOperator(k0), cudaq.KrausOperator(k1), cudaq.KrausOperator(k2)]
    )


@dataclass(frozen=True)
class DeviceNoiseParameters:
    """Base class carrying the modelled/unmodelled split for a device."""

    MODELLED: ClassVar[tuple[str, ...]] = ()
    UNMODELLED: ClassVar[tuple[str, ...]] = ()

    def modelled_parameters(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.MODELLED}

    def unmodelled_parameters(self) -> dict[str, float]:
        return {
            name: getattr(self, name)
            for name in self.UNMODELLED
            if getattr(self, name) not in (None, 0.0)
        }

    def describe(self) -> str:
        ignored = ", ".join(sorted(self.unmodelled_parameters())) or "none"
        return (
            f"{getattr(self, 'device', type(self).__name__)}: "
            f"models {', '.join(self.MODELLED)}; ignores {ignored}"
        )


@dataclass(frozen=True)
class QuantinuumH1NoiseParameters(DeviceNoiseParameters):
    """Published H1-1E emulator defaults."""

    device: str = "Quantinuum H1-1E"
    qubits: int = 20
    connectivity: str = "all-to-all"
    parallel_two_qubit_operations: int = 5
    p1: float = 1.8e-05
    p2: float = 0.000973
    p_meas_0_to_1: float = 0.00122
    p_meas_1_to_0: float = 0.00343
    p_init: float = 3.62e-05
    p_crosstalk_meas: float = 2.03e-05
    p_crosstalk_init: float = 2.03e-05
    p1_emission_ratio: float = 0.747
    p2_emission_ratio: float = 0.421
    quadratic_dephasing_rate: float = 0.122
    linear_dephasing_rate: float = 0.0
    coherent_to_incoherent_factor: float = 2.5
    przz_a: float = 1.651
    przz_b: float = 0.175
    przz_c: float = 1.651
    przz_d: float = 0.175
    przz_power: float = 1.0

    MODELLED: ClassVar[tuple[str, ...]] = ("p1", "p2", "p_meas_0_to_1", "p_meas_1_to_0")
    UNMODELLED: ClassVar[tuple[str, ...]] = (
        "p_init", "p_crosstalk_meas", "p_crosstalk_init",
        "p1_emission_ratio", "p2_emission_ratio",
        "quadratic_dephasing_rate", "linear_dephasing_rate",
        "coherent_to_incoherent_factor",
        "przz_a", "przz_b", "przz_c", "przz_d", "przz_power",
    )


@dataclass(frozen=True)
class QuantinuumH2NoiseParameters(DeviceNoiseParameters):
    """Nexus H2-1E backend details, captured 2026-06-04."""

    device: str = "Quantinuum H2-1E"
    backend: str = "Quantinuum/H2-1E"
    qubits: int = 56
    connectivity: str = "fully-connected"
    p1: float = 1.89e-05
    p2: float = 0.00105
    p_meas_0_to_1: float = 0.0006
    p_meas_1_to_0: float = 0.00139
    p_init: float = 4e-05
    p_crosstalk_init: float = 3.3e-06
    p_crosstalk_meas: float = 3.3e-06
    p1_emission_ratio: float = 0.54
    p2_emission_ratio: float = 0.255
    linear_dephasing_rate: float = 0.0028
    quadratic_dephasing_rate: float = 0.043
    coherent_to_incoherent_factor: float = 2.0
    przz_a: float = 1.52
    przz_b: float = 0.241
    przz_c: float = 1.52
    przz_d: float | None = None
    captured_at: str = "2026-06-04 12:53 Europe/London"

    MODELLED: ClassVar[tuple[str, ...]] = ("p1", "p2", "p_meas_0_to_1", "p_meas_1_to_0")
    UNMODELLED: ClassVar[tuple[str, ...]] = (
        "p_init", "p_crosstalk_init", "p_crosstalk_meas",
        "p1_emission_ratio", "p2_emission_ratio",
        "linear_dephasing_rate", "quadratic_dephasing_rate",
        "coherent_to_incoherent_factor",
        "przz_a", "przz_b", "przz_c", "przz_d",
    )


H1_1E = QuantinuumH1NoiseParameters()
H2_1E = QuantinuumH2NoiseParameters()


def build_device_noise_model(parameters: DeviceNoiseParameters):
    """Build the CUDA-Q noise model for the subset of parameters it can express."""
    noise = make_depolarizing_noise(p_1q=float(parameters.p1), p_2q=float(parameters.p2))
    noise.add_all_qubit_channel(
        "mz",
        asymmetric_readout_channel(
            float(parameters.p_meas_0_to_1), float(parameters.p_meas_1_to_0)
        ),
    )
    return noise


def approximate_quantinuum_h1_noise_model(parameters=None):
    """Approximate the H1-1E emulator."""
    return build_device_noise_model(parameters or H1_1E)


def approximate_quantinuum_h2_noise_model(parameters=None):
    """Approximate the H2-1E backend."""
    return build_device_noise_model(parameters or H2_1E)
