"""Qiskit-specific QUOPS kernel: builds BR / RR / REF mirror circuits from gate
arrays, runs them on AerSimulator, and provides the noise models.

Imports nothing from _common: the gate codes are defined below and must match
_common/quops_fctns.py. Circuit-level helpers for the tutorial are in quops_helpers.py
(QPerf-work only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


# Gate codes used in the gate arrays (same values as in _common/quops_fctns.py)
RX, RY, RZ, X, Y, Z, CX, U3 = range(8)


# ===========================================================================
# Circuit builders
# ===========================================================================


# Every single-qubit gate the QUOPS kernels can emit.
ONE_QUBIT_GATE_KEYS = ("rx", "ry", "rz", "x", "y", "z", "u3")
TWO_QUBIT_GATE_KEYS = ("cx",)


def build_circuit(gates, q0, q1, angles, width, phis=None, lams=None) -> QuantumCircuit:
    """Build a Qiskit QuantumCircuit from flat gate arrays.

    Supports the full gate set including U3.  Gate list metadata is attached
    so that ``randomized_compilation`` and ``adjoint_circuit`` can rewrite it.
    """
    n = len(gates)
    if phis is None:
        phis = [0.0] * n
    if lams is None:
        lams = [0.0] * n

    qc = QuantumCircuit(width)
    for i in range(n):
        g = gates[i]
        if g == RX:
            qc.rx(angles[i], q0[i])
        elif g == RY:
            qc.ry(angles[i], q0[i])
        elif g == RZ:
            qc.rz(angles[i], q0[i])
        elif g == X:
            qc.x(q0[i])
        elif g == Y:
            qc.y(q0[i])
        elif g == Z:
            qc.z(q0[i])
        elif g == CX:
            qc.cx(q0[i], q1[i])
        elif g == U3:
            qc.u(angles[i], phis[i], lams[i], q0[i])

    qc.metadata = {
        "gates": list(gates),
        "q0s": list(q0),
        "q1s": list(q1),
        "angles": [float(a) for a in angles],
        "phis": [float(p) for p in phis],
        "lams": [float(l) for l in lams],
        "width": int(width),
    }
    return qc


def require_circuit_arrays(circuit: QuantumCircuit):
    """Return ``(gates, q0s, q1s, angles, phis, lams, width)`` from circuit metadata."""
    m = circuit.metadata
    if m is None or "gates" not in m:
        raise ValueError("expected a circuit built by build_circuit")
    n = len(m["gates"])
    return (
        m["gates"],
        m["q0s"],
        m["q1s"],
        m["angles"],
        m.get("phis", [0.0] * n),
        m.get("lams", [0.0] * n),
        m["width"],
    )


# ===========================================================================
# Composite circuits (BR / RR / REF) with MCFE caps
# ===========================================================================


def build_br_circuit(width: int, L: QuantumCircuit, C: QuantumCircuit,
                     RC_C_inv: QuantumCircuit, L_prime: QuantumCircuit) -> QuantumCircuit:
    """M1 mirror: L' g_ref(C^dag) C L, then measure (Supp. Eq. 30).

    Barriers between the parts keep a transpiler from cancelling gates across them.
    """
    qc = QuantumCircuit(width, width)
    qc.compose(L, inplace=True)
    qc.barrier()
    qc.compose(C, inplace=True)
    qc.barrier()
    qc.compose(RC_C_inv, inplace=True)
    qc.barrier()
    qc.compose(L_prime, inplace=True)
    qc.measure(range(width), range(width))
    return qc


def build_rr_circuit(width: int, L: QuantumCircuit, RC_C: QuantumCircuit,
                     RC_C_inv: QuantumCircuit, L_prime: QuantumCircuit) -> QuantumCircuit:
    """M2 mirror: L' g_ref(C^dag) g_ref(C) L, then measure (Supp. Eq. 31).

    Barriers between the parts keep a transpiler from cancelling gates across them.
    """
    qc = QuantumCircuit(width, width)
    qc.compose(L, inplace=True)
    qc.barrier()
    qc.compose(RC_C, inplace=True)
    qc.barrier()
    qc.compose(RC_C_inv, inplace=True)
    qc.barrier()
    qc.compose(L_prime, inplace=True)
    qc.measure(range(width), range(width))
    return qc


def build_ref_circuit(width: int, L: QuantumCircuit, L_prime: QuantumCircuit) -> QuantumCircuit:
    """M3 SPAM reference: L' L, then measure (Supp. Eq. 32).

    A barrier between the parts keeps a transpiler from cancelling gates across them.
    """
    qc = QuantumCircuit(width, width)
    qc.compose(L, inplace=True)
    qc.barrier()
    qc.compose(L_prime, inplace=True)
    qc.measure(range(width), range(width))
    return qc


# ===========================================================================
# Execution
# ===========================================================================


def run_circuit(circuit: QuantumCircuit, shots: int = 1000,
                noise_model: NoiseModel | None = None):
    """Run a circuit on AerSimulator and return the result."""
    backend = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()
    result = backend.run(circuit, shots=shots).result()
    return result


# ===========================================================================
# Mirror circuits for the shared scan loop (quops_benchmark.py)
# ===========================================================================


def build_mirror(kind: str, width: int, parts, compile_once: bool = True) -> QuantumCircuit:
    """Build a measured BR, RR or REF circuit from gate-array parts in execution order.

    ``parts`` are ``(gates, q0s, q1s, angles, phis, lams)`` tuples: L, C, g(C^dag), L' for
    BR and RR, or L, L' for REF. ``compile_once`` is a CUDA-Q option, accepted and ignored here.
    """
    builders = {"br": build_br_circuit, "rr": build_rr_circuit, "ref": build_ref_circuit}
    circuits = [build_circuit(g, q0, q1, a, width, p, l) for g, q0, q1, a, p, l in parts]
    return builders[kind](width, *circuits)


def sample_counts(circuit: QuantumCircuit, shots: int = 1000,
                  noise_model: NoiseModel | None = None) -> dict[str, int]:
    """Run a circuit on AerSimulator and return counts keyed q0-first.

    Qiskit returns bit strings in little-endian order (qubit 0 is the rightmost
    bit), so each key is reversed to match the CUDA-Q convention.
    """
    counts = run_circuit(circuit, shots=shots, noise_model=noise_model).get_counts()
    return {k[::-1]: v for k, v in counts.items()}


# ===========================================================================
# Noise models
# ===========================================================================


def make_depolarizing_noise(p_1q: float = 0.0, p_2q: float = 0.0) -> NoiseModel:
    """Build a uniform depolarizing noise model over the QUOPS gate set.

    Covers ``u3`` so that pauli-frame-compiled reference circuits see noise.
    """
    noise = NoiseModel()
    if p_1q:
        err_1q = depolarizing_error(p_1q, 1)
        for gate in ONE_QUBIT_GATE_KEYS:
            noise.add_all_qubit_quantum_error(err_1q, gate)
    if p_2q:
        err_2q = depolarizing_error(p_2q, 2)
        for gate in TWO_QUBIT_GATE_KEYS:
            noise.add_all_qubit_quantum_error(err_2q, gate)
    return noise


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


def build_device_noise_model(parameters: DeviceNoiseParameters) -> NoiseModel:
    """Build the Qiskit noise model for the subset of parameters it can express."""
    noise = make_depolarizing_noise(p_1q=float(parameters.p1), p_2q=float(parameters.p2))
    p01 = float(parameters.p_meas_0_to_1)
    p10 = float(parameters.p_meas_1_to_0)
    readout_err = ReadoutError([[1.0 - p01, p01], [p10, 1.0 - p10]])
    noise.add_all_qubit_readout_error(readout_err)
    return noise


def approximate_quantinuum_h1_noise_model(parameters=None) -> NoiseModel:
    """Approximate the H1-1E emulator."""
    return build_device_noise_model(parameters or H1_1E)


def approximate_quantinuum_h2_noise_model(parameters=None) -> NoiseModel:
    """Approximate the H2-1E backend."""
    return build_device_noise_model(parameters or H2_1E)
