"""QUOPS Benchmark - command-line driver for the Qiskit and CUDA-Q kernels.

    python quops/quops_benchmark.py                    # Qiskit Aer, depolarizing noise (Q = 36)
    python quops/quops_benchmark.py -a cudaq           # CUDA-Q on the GPU (nvidia), depolarizing noise
    python quops/quops_benchmark.py --noise none       # noiseless (every shape passes)
    python quops/quops_benchmark.py -a cudaq -b density-matrix-cpu   # CUDA-Q, exact noise on the CPU
    python quops/quops_benchmark.py -a cudaq -n 30 -d 15 -c 2 --num_mirrors 1 --noise none   # one 30-qubit shape

The run is saved to __data/<backend_id>/QUOPS-results.json and the plots to
__images/<backend_id>/QUOPS-*.jpg and .pdf, in the current directory. To remake the
plots from the saved data: quops_benchmark.plot_results(backend_id).

The API kernels are qiskit_kernel/quops_kernel.py and cudaq_kernel/quops_kernel.py
(same file name, same functions); shared code is in _common/quops_fctns.py.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

QUOPS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(QUOPS_DIR, "_common"))  # quops_fctns.py

from quops_fctns import (
    CX, adjoint_arrays, rc_rewrite_arrays, sample_quops_arrays, shape_size,
    clifford_layer_arrays, sample_mcfe_cap_indices, effective_polarization_from_counts,
    ShapePolarizationData, enumerate_shapes, default_shape_schedule, _is_at_least,
    pass_statistics, in_utility_cone, quops_score_from_summary,
    omega_simulated, omega_projected, error_mitigation_overhead,
    BenchmarkResult, GateDurations,
    QUOPS_THRESHOLD, DEFAULT_FAMILYWISE_ALPHA,
)

# Depolarizing noise used by the benchmark notebooks (validated at Q = 36)
P_1Q, P_2Q = 0.002, 0.05

# Default backend: Qiskit always runs Aer (the id only names the image folder); CUDA-Q uses the
# GPU (with noise it simulates trajectories; -b density-matrix-cpu gives exact noise on the CPU)
DEFAULT_BACKEND = {"qiskit": "qasm_simulator", "cudaq": "nvidia"}


def import_kernel(api):
    """Load <api>_kernel/quops_kernel.py under the unique module name quops_kernel_<api>.

    Both kernels have the same file name, so a plain ``import quops_kernel`` would
    return whichever was loaded first if one session switches APIs.
    """
    name = f"quops_kernel_{api}"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(QUOPS_DIR, f"{api}_kernel", "quops_kernel.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses in the kernel look up their module here
    spec.loader.exec_module(module)
    return module


def load_kernel(api, backend_id):
    """Import the kernel module for this API and return it with a backend label for plots."""
    if api == "qiskit":
        kernel = import_kernel("qiskit")
        import qiskit, qiskit_aer
        backend_label = f"Qiskit Aer {qiskit_aer.__version__} (Qiskit {qiskit.__version__})"

    elif api == "cudaq":
        import cudaq
        cudaq.set_target(backend_id)
        kernel = import_kernel("cudaq")
        # e.g. "amd64-cu12-0.16.0" (drops the GitHub URL and commit hash)
        version = cudaq.__version__.split(" (")[0].removeprefix("CUDA-Q Version ")
        backend_label = f"CUDA-Q {version} ({cudaq.get_target().name})"

    else:
        raise ValueError(f"Unknown API '{api}': use qiskit or cudaq")

    return kernel, backend_label


############### Save and Load Data

def data_file(backend_id):
    """Path of the saved QUOPS data for a backend: __data/<backend_id>/QUOPS-results.json."""
    return os.path.join(os.getcwd(), "__data", backend_id.replace("/", "_"), "QUOPS-results.json")


def _json_value(value):
    """Convert numpy scalars (np.int64, np.bool_, ...) for json.dump."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot save {type(value).__name__} as JSON")


def save_data(result, run_info):
    """Save a QUOPS run to __data/<backend_id>/QUOPS-results.json.

    The file holds ``run_info`` (backend, API, settings, start and end time), the score
    (Q, its shape, Omega, significance used), the summary table (one row per shape tested)
    and the raw BR/RR/REF polarizations per shape. ``plot_results`` draws the plots from it,
    so they can be remade later without running the benchmark again.
    """
    data = dict(run_info)
    data.update(
        quops=result.quops,
        shape=result.shape,
        omega=result.omega,
        omega_note=result.omega_note,
        conditional_quops=result.conditional_quops,
        conditional_shape=result.conditional_shape,
        conditional_overhead=result.conditional_overhead,
        consumed_significance=result.consumed_significance,
        summary=result.summary.to_dict(orient="records"),
        shape_data=[asdict(shape) for shape in result.shape_data],
    )
    path = data_file(run_info["backend_id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_value)
    print(f"... saved {path}")


def load_data(backend_id):
    """Load a run saved by ``save_data``. Returns ``(result, run_info)``: the BenchmarkResult
    and a dict with everything else in the file (backend, API, settings, times)."""
    with open(data_file(backend_id)) as f:
        data = json.load(f)

    def as_shape(value):
        return tuple(value) if value is not None else None

    result = BenchmarkResult(
        summary=pd.DataFrame(data.pop("summary")),
        quops=data.pop("quops"),
        shape=as_shape(data.pop("shape")),
        omega=data.pop("omega"),
        omega_note=data.pop("omega_note"),
        conditional_quops=data.pop("conditional_quops"),
        conditional_shape=as_shape(data.pop("conditional_shape")),
        conditional_overhead=data.pop("conditional_overhead"),
        consumed_significance=data.pop("consumed_significance"),
        shape_data=[
            ShapePolarizationData(
                width=shape["width"], size=shape["size"], depth=shape["depth"], zeta=shape["zeta"],
                br=tuple(tuple(mirrors) for mirrors in shape["br"]),
                rr=tuple(tuple(mirrors) for mirrors in shape["rr"]),
                ref=tuple(shape["ref"]), metadata=shape["metadata"],
            )
            for shape in data.pop("shape_data")
        ],
    )
    return result, data


############### Plot Results

def plot_results(backend_id=None, api="qiskit"):
    """Plot a saved QUOPS run: read __data/<backend_id>/QUOPS-results.json and save the
    shape-scan and polarization plots as .jpg and .pdf under __images/<backend_id>/.

    Args:
        backend_id: the backend the run was saved under (default: the API's default backend)
        api: used only to pick the default backend_id (default "qiskit")
    """
    from quops_fctns import plot_quops_scan

    if backend_id is None:
        backend_id = DEFAULT_BACKEND.get(api, "")
    result, run_info = load_data(backend_id)

    # the date of the run, not of the plot
    run_date = datetime.datetime.fromisoformat(run_info["start_time"])
    label = f"Backend: {run_info['backend_label']} | {run_date.day} {run_date:%B %Y}\n{run_info['settings_label']}"
    figures = plot_quops_scan(
        result.summary,
        title=f"QUOPS Benchmark  (Q = {result.quops})",
        label=label,
    )

    folder = os.path.join(os.getcwd(), "__images", backend_id.replace("/", "_"))
    os.makedirs(folder, exist_ok=True)
    for figure, name in zip(figures, ("QUOPS-shape-scan", "QUOPS-polarization")):
        for ext in ("jpg", "pdf"):
            path = os.path.join(folder, f"{name}.{ext}")
            # tight keeps the legend, which sits outside the axes
            figure.savefig(path, bbox_inches="tight")
            print(f"... saved {path}")


############### Gate-array helpers (API-neutral)

def _frozen_arrays(gates, q0s, q1s, angles, phis=None, lams=None):
    """Return ``(gates, q0s, q1s, angles, phis, lams)`` as plain int/float lists."""
    n = len(gates)
    return (
        [int(gate) for gate in gates],
        [int(qubit) for qubit in q0s],
        [int(qubit) for qubit in q1s],
        [float(angle) for angle in angles],
        [0.0] * n if phis is None else [float(value) for value in phis],
        [0.0] * n if lams is None else [float(value) for value in lams],
    )


@dataclass(frozen=True)
class MCFECapArrays:
    """MCFE caps as gate arrays: L, L' = P L^dag, and the target bit string."""

    initial: tuple
    final: tuple
    target: str


def sample_mcfe_cap_arrays(width: int, seed=None) -> MCFECapArrays:
    """Array version of the kernels' ``sample_mcfe_caps`` (same seed gives the same caps)."""
    cliffords, paulis, final, target = sample_mcfe_cap_indices(width, seed)
    initial_arrays = clifford_layer_arrays(cliffords)
    final_arrays = clifford_layer_arrays(final)
    return MCFECapArrays(
        initial=_frozen_arrays(*initial_arrays[:6]),
        final=_frozen_arrays(*final_arrays[:6]),
        target=target,
    )


def _rc_arrays(arrays, width, seed, style):
    """Randomized compilation of a gate-array circuit (the kernels' ``randomized_compilation``)."""
    gates, q0s, q1s, angles, phis, lams = arrays
    return _frozen_arrays(*rc_rewrite_arrays(gates, q0s, q1s, angles, width, phis, lams,
                                             seed=seed, style=style))


############### Scan loop (API-neutral; moved from the kernels, ALIGNMENT-LOG entry 6)
#
# For each shape the scan creates the circuits (get_circuits), runs them (run_circuits),
# and turns the counts into polarizations (estimate_shape_polarization); ALIGNMENT-LOG
# entries 9-10. The functions take ``api`` and load that kernel module, which supplies only:
#   build_mirror(kind, width, parts, compile_once) -> a runnable BR/RR/REF circuit, where
#       parts are gate-array tuples (gates, q0s, q1s, angles, phis, lams) in execution order
#   sample_counts(circuit, shots, noise_model) -> counts dict keyed q0-first
#   make_depolarizing_noise(p_1q, p_2q) and the device noise models


def collect_reference_ensemble(
    api,
    width: int,
    *,
    num_mirrors: int = 8,
    shots: int = 1000,
    noise_model=None,
    seed=None,
    compile_once: bool = True,
) -> tuple[float, ...]:
    """Collect the width-w M3 (SPAM reference) ensemble L' L."""
    kernel = import_kernel(api)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(num_mirrors)):
        caps = sample_mcfe_cap_arrays(width, seed=int(rng.integers(0, 2**31 - 1)))
        ref_circuit = kernel.build_mirror("ref", width, (caps.initial, caps.final), compile_once=compile_once)
        counts = kernel.sample_counts(ref_circuit, shots=shots, noise_model=noise_model)
        samples.append(effective_polarization_from_counts(counts, width, target=caps.target))
    return tuple(samples)


############### Get Circuits

def get_circuits(
    num_qubits: int,
    depth: int,
    zeta: float = 1.0,
    max_circuits: int = 10,
    num_mirrors: int = 4,
    rc_style: str = "pauli_frame",
    seed=None,
    reference: Sequence[float] | None = None,
    compile_once: bool = True,
    api: str = "qiskit",
):
    """Create the BR, RR and REF mirror circuits for one circuit shape (width, depth).

    The scan calls this once per shape, so it can stop as soon as a shape fails.

    Args:
        num_qubits: circuit width w
        depth: circuit depth d (layer pairs)
        zeta: fraction of qubits in the final layer (default 1.0)
        max_circuits: random circuits for this shape (default 10)
        num_mirrors: mirror samples per circuit (default 4)
        rc_style: randomized-compilation style (default "pauli_frame")
        seed: random seed for this shape (default None)
        reference: SPAM reference polarizations to use instead of REF circuits; with it,
            no REF circuits are built (default None)
        compile_once: CUDA-Q only, passed to the kernel's ``build_mirror`` (ALIGNMENT-LOG
            entry 2); same seed, same circuits either way (default True)
        api: "qiskit" (default) or "cudaq"

    As in PRAQTICE's ``run_quops``, each mirror sample draws independent MCFE caps
    for BR, RR and REF, and gets its own REF circuit (ALIGNMENT-LOG entry 1).

    Returns ``(all_qcs, shape_info)``: ``all_qcs`` is ``{str(num_qubits): {circuit_id: circuit}}``
    with ids like ``"c3_m1_br"`` (circuit 3, mirror 1), in run order; ``shape_info`` holds
    ``targets`` (the target bit string for each id), the shape's size and gate counts.
    """
    if int(max_circuits) < 1:
        raise ValueError("max_circuits must be at least one")

    kernel = import_kernel(api)
    width = int(num_qubits)
    num_circuits = int(max_circuits)
    rng = np.random.default_rng(seed)
    circuits, targets = {}, {}

    size = shape_size(width, depth, zeta)
    one_qubit = two_qubit = 0
    reference_sizes = []

    for c in range(num_circuits):
        gates, q0, q1, angles, size = sample_quops_arrays(
            width, depth, zeta, seed=int(rng.integers(0, 2**31 - 1))
        )
        two_qubit = sum(1 for gate in gates if gate == CX)
        one_qubit = len(gates) - two_qubit
        circuit = _frozen_arrays(gates, q0, q1, angles)
        inverse = _frozen_arrays(*adjoint_arrays(*circuit))

        for m in range(int(num_mirrors)):
            caps_br, caps_rr, caps_ref = (
                sample_mcfe_cap_arrays(width, seed=int(cap_seed))
                for cap_seed in rng.integers(0, 2**31 - 1, size=3)
            )
            seeds = rng.integers(0, 2**31 - 1, size=3)
            br_inverse = _rc_arrays(inverse, width, int(seeds[0]), rc_style)
            rr_forward = _rc_arrays(circuit, width, int(seeds[1]), rc_style)
            rr_inverse = _rc_arrays(inverse, width, int(seeds[2]), rc_style)

            reference_sizes.append(len(rr_forward[0]))

            circuit_id = f"c{c}_m{m}"
            circuits[f"{circuit_id}_br"] = kernel.build_mirror(
                "br", width, (caps_br.initial, circuit, br_inverse, caps_br.final), compile_once=compile_once)
            targets[f"{circuit_id}_br"] = caps_br.target
            circuits[f"{circuit_id}_rr"] = kernel.build_mirror(
                "rr", width, (caps_rr.initial, rr_forward, rr_inverse, caps_rr.final), compile_once=compile_once)
            targets[f"{circuit_id}_rr"] = caps_rr.target
            if reference is None:
                circuits[f"{circuit_id}_ref"] = kernel.build_mirror(
                    "ref", width, (caps_ref.initial, caps_ref.final), compile_once=compile_once)
                targets[f"{circuit_id}_ref"] = caps_ref.target

    shape_info = {
        "targets": targets,
        "size": size,
        "one_qubit_gates": one_qubit,
        "two_qubit_gates": two_qubit,
        "median_reference_size": float(np.median(reference_sizes)) if reference_sizes else float("nan"),
    }
    return {str(width): circuits}, shape_info


############### Run Circuits

def run_circuits(all_qcs, num_shots: int = 1000, noise_model=None, api: str = "qiskit"):
    """Execute the circuits from ``get_circuits``, in order, with the kernel's ``sample_counts``.

    Args:
        all_qcs: circuit dict from get_circuits()
        num_shots: shots per circuit (default 1000)
        noise_model: the API's noise model object, or None for noiseless (default None)
        api: "qiskit" (default) or "cudaq"

    Returns ``(counts, tau_wall)``: counts by circuit id, and the time spent on the
    BR and RR circuits (REF is excluded from tau_wall, as in PRAQTICE).
    """
    kernel = import_kernel(api)
    counts = {}
    tau_wall = 0.0
    for circuits in all_qcs.values():
        for circuit_id, circuit in circuits.items():
            start = time.perf_counter()
            counts[circuit_id] = kernel.sample_counts(circuit, shots=num_shots, noise_model=noise_model)
            if not circuit_id.endswith("_ref"):
                tau_wall += time.perf_counter() - start
    return counts, tau_wall


def estimate_shape_polarization(
    api,
    width: int,
    depth: int,
    zeta: float,
    *,
    num_circuits: int = 10,
    num_mirrors: int = 4,
    shots: int = 1000,
    noise_model=None,
    rc_style: str = "pauli_frame",
    seed=None,
    reference: Sequence[float] | None = None,
    reference_mirrors: int = 8,
    compile_once: bool = True,
) -> ShapePolarizationData:
    """Collect the M1/M2/M3 mirror data for one circuit shape.

    Creates the shape's circuits (``get_circuits``), runs them (``run_circuits``),
    and turns the counts into polarizations.
    A supplied ``reference`` is used instead of the per-mirror REF measurements.
    """
    all_qcs, shape_info = get_circuits(
        width, depth, zeta,
        max_circuits=num_circuits, num_mirrors=num_mirrors, rc_style=rc_style,
        seed=seed, reference=reference, compile_once=compile_once, api=api,
    )
    counts, tau_wall = run_circuits(all_qcs, num_shots=shots, noise_model=noise_model, api=api)

    targets = shape_info["targets"]
    polarization = {
        circuit_id: effective_polarization_from_counts(counts[circuit_id], width, target=targets[circuit_id])
        for circuit_id in counts
    }
    circuit_range, mirror_range = range(int(num_circuits)), range(int(num_mirrors))
    br_by_circuit = [tuple(polarization[f"c{c}_m{m}_br"] for m in mirror_range) for c in circuit_range]
    rr_by_circuit = [tuple(polarization[f"c{c}_m{m}_rr"] for m in mirror_range) for c in circuit_range]
    ref_samples = [polarization[f"c{c}_m{m}_ref"] for c in circuit_range for m in mirror_range] \
        if reference is None else []

    size = shape_info["size"]
    one_qubit, two_qubit = shape_info["one_qubit_gates"], shape_info["two_qubit_gates"]
    shots_kept = 2 * int(shots) * len(circuit_range) * len(mirror_range)

    return ShapePolarizationData(
        width=int(width),
        size=int(size),
        depth=int(depth),
        zeta=float(zeta),
        br=tuple(br_by_circuit),
        rr=tuple(rr_by_circuit),
        ref=tuple(reference) if reference is not None else tuple(ref_samples),
        metadata={
            "rc_style": rc_style,
            "shots": int(shots),
            "num_mirrors": int(num_mirrors),
            "one_qubit_gates": one_qubit,
            "two_qubit_gates": two_qubit,
            "median_reference_size": shape_info["median_reference_size"],
            "tau_wall": tau_wall,
            "shots_kept": shots_kept,
            "shots_rejected": 0,
        },
    )


def run_quops_benchmark(
    api,
    shapes: Sequence[dict] | None = None,
    *,
    width_range: Sequence[int] = (2, 6),
    zeta: float | None = None,
    num_circuits: int = 10,
    num_mirrors: int = 4,
    shots: int = 1000,
    noise_model=None,
    rc_style: str = "pauli_frame",
    estimator: str = "eq45",
    threshold: float = QUOPS_THRESHOLD,
    conditional_threshold: float | None = None,
    alpha_per_test: float = DEFAULT_FAMILYWISE_ALPHA,
    familywise_alpha: float = DEFAULT_FAMILYWISE_ALPHA,
    reference_mirrors: int = 8,
    bootstrap_resamples: int = 2000,
    durations: GateDurations | None = None,
    seed=None,
    progress: Callable[[str], None] | None = print,
    compile_once: bool = True,
) -> BenchmarkResult:
    """Run the formal QUOPS benchmark of Supplementary Section V.

    Shapes are visited in increasing size, and a shape enters the capability
    region when the one-sided ``1 - alpha`` lower bound on its mean
    polarization clears ``threshold``.  A passing test costs no significance
    and a failing one charges ``alpha``; the run ends once the budget is spent.

    With the default ``alpha_per_test = familywise_alpha`` the first failure
    exhausts the budget, giving the single increasing path through shape space
    that Methods requires for ``Q`` to be a 95% confidence estimate.

    To allow N failures, pass ``alpha_per_test = familywise_alpha / N`` (``run(max_failures=N)``).

    ``api`` is "qiskit" or "cudaq" (the kernel is loaded with ``import_kernel``).
    ``compile_once`` is passed to ``estimate_shape_polarization`` (ALIGNMENT-LOG entry 2).
    """
    if conditional_threshold is not None and conditional_threshold >= threshold:
        raise ValueError("conditional_threshold must be below the main threshold")

    if shapes is None:
        shapes = enumerate_shapes(width_range, zeta)
    schedule = default_shape_schedule(shapes)

    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    collected: list[ShapePolarizationData] = []
    excluded: list[tuple[int, int]] = []
    conditionally_excluded: list[tuple[int, int]] = []
    beta = 0.0

    for shape in schedule:
        alpha = min(alpha_per_test, familywise_alpha - beta)
        if alpha <= 1e-12:  # budget spent (tolerance for alpha_per_test = familywise_alpha / N)
            break
        if any(_is_at_least(shape, failed) for failed in conditionally_excluded):
            continue

        width, depth = int(shape["width"]), int(shape["depth"])
        shape_zeta = float(shape.get("zeta", 1.0))

        # REF is now measured per mirror inside estimate_shape_polarization
        # (ALIGNMENT-LOG entry 1); the per-width reference cache is no longer used.
        data = estimate_shape_polarization(
            api,
            width,
            depth,
            shape_zeta,
            num_circuits=num_circuits,
            num_mirrors=num_mirrors,
            shots=shots,
            noise_model=noise_model,
            rc_style=rc_style,
            seed=int(rng.integers(0, 2**31 - 1)),
            compile_once=compile_once,
        )
        collected.append(data)

        statistics = pass_statistics(
            data,
            estimator=estimator,
            threshold=threshold,
            alpha=alpha,
            resamples=bootstrap_resamples,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        statistics.update(
            {
                "in_utility_cone": in_utility_cone(width, data.size),
                "median_reference_size": data.metadata["median_reference_size"],
                "one_qubit_gates": data.metadata["one_qubit_gates"],
                "two_qubit_gates": data.metadata["two_qubit_gates"],
                "tau_wall": data.metadata["tau_wall"],
                "shots_kept": data.metadata["shots_kept"],
            }
        )

        lower = statistics["bootstrap_lower"]
        eligible = not any(_is_at_least(shape, failed) for failed in excluded)
        statistics["passes"] = bool(statistics["passes"]) and eligible

        if statistics["passes"]:
            statistics["passes_conditional"] = True
        else:
            excluded.append((width, data.size))
            rescued = (
                conditional_threshold is not None
                and np.isfinite(lower)
                and lower >= conditional_threshold
            )
            statistics["passes_conditional"] = bool(rescued)
            if not rescued:
                beta += alpha
                conditionally_excluded.append((width, data.size))

        statistics["alpha"] = alpha
        statistics["consumed_significance"] = beta
        rows.append(statistics)

        if progress is not None:
            progress(
                f"w={width:2d} s={data.size:4d} zeta={shape_zeta:.2f}  "
                f"lambda={statistics['mean_polarization']:.4f} "
                f"lower={statistics['bootstrap_lower']:.4f}  "
                f"{'PASS' if statistics['passes'] else 'FAIL'}"
            )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["width", "size"]).reset_index(drop=True)

    quops, shape_at_q = quops_score_from_summary(summary, pass_column="passes")

    omega = float("nan")
    omega_note = "no passing shape inside the utility cone"
    if shape_at_q is not None and not summary.empty:
        row = summary[
            (summary["width"] == shape_at_q[0]) & (summary["size"] == shape_at_q[1])
        ].iloc[0]
        if durations is None:
            omega = omega_simulated(
                int(row["size"]),
                float(row["mean_polarization"]),
                float(row["tau_wall"]),
                int(row["shots_kept"]),
            )
            omega_note = "simulator wall-clock throughput, not a hardware rate"
        else:
            omega = omega_projected(
                int(row["size"]),
                float(row["mean_polarization"]),
                int(row["one_qubit_gates"]),
                int(row["two_qubit_gates"]),
                int(row["width"]),
                durations,
                int(row["shots_kept"]),
            )
            omega_note = "projected from the supplied gate-duration model"

    conditional_quops, conditional_shape, conditional_overhead = 0, None, float("nan")
    if conditional_threshold is not None and not summary.empty:
        conditional_quops, conditional_shape = quops_score_from_summary(
            summary, pass_column="passes_conditional"
        )
        if conditional_quops > quops:
            conditional_overhead = error_mitigation_overhead(conditional_threshold)
        else:
            conditional_quops, conditional_shape = 0, None

    return BenchmarkResult(
        summary=summary,
        quops=quops,
        shape=shape_at_q,
        omega=omega,
        omega_note=omega_note,
        conditional_quops=conditional_quops,
        conditional_shape=conditional_shape,
        conditional_overhead=conditional_overhead,
        consumed_significance=beta,
        shape_data=collected,
    )


############### Run

def get_shapes(min_qubits, max_qubits, skip_qubits=1, max_depth=None):
    """List the shapes to scan: utility-cone shapes at widths min_qubits..max_qubits
    (every skip_qubits), with depth <= max_depth (None = 3 x width, 0 = no cap)."""
    widths = range(min_qubits, max_qubits + 1, max(1, skip_qubits))
    shapes = []
    for shape in enumerate_shapes((min_qubits, max_qubits), 1.0):
        width, depth = shape["width"], shape["depth"]
        depth_cap = 3 * width if max_depth is None else max_depth
        if width in widths and (depth_cap == 0 or depth <= depth_cap):
            shapes.append(shape)
    return shapes


def run(
    # App args: the shape sweep and sampling
    min_qubits=2, max_qubits=6, skip_qubits=1, max_depth=None,
    max_circuits=10, num_mirrors=4, seed=2026, noise="depolarizing", max_failures=1,
    # Exec args: API, backend and shots
    api="qiskit", backend_id=None, num_shots=1000,
    # Benchmark args
    compile_once=True, plot_results=True,
    **kwargs,
):
    """Run the QUOPS benchmark and return its BenchmarkResult.

    App args:
        min_qubits, max_qubits: range of circuit widths to scan (default 2-6)
        skip_qubits: step between widths (default 1)
        max_depth: largest depth (layer pairs) to scan at each width; None = 3 x width
            (default), 0 = the whole utility cone. Deeper shapes are left out, never cut short.
        max_circuits: random circuits per shape (default 10)
        num_mirrors: mirror samples per circuit (default 4)
        seed: random seed (default 2026)
        noise: "depolarizing" (default; p1=0.002, p2=0.05) or "none" (noiseless)
        max_failures: failed shapes before the scan stops (default 1, the paper's method);
            each test then uses alpha = 0.05 / max_failures, so the total stays at 5%

    Exec args:
        api: "qiskit" (default) or "cudaq"
        backend_id: CUDA-Q target (default nvidia; density-matrix-cpu for exact noise);
            for Qiskit only the image folder name (default qasm_simulator)
        num_shots: shots per circuit (default 1000)

    Benchmark args:
        compile_once: CUDA-Q only; True (default) runs every mirror through one kernel
            compiled once; False builds and compiles a new kernel per circuit, as in
            the PRAQTICE code (same results, about 100x slower). Ignored for Qiskit.
        plot_results: plot the saved data and save the plots under __images/<backend_id>/
            (default True). The data is always saved to __data/<backend_id>/QUOPS-results.json;
            ``plot_results(backend_id)`` remakes the plots from it later.

    Other keyword arguments are ignored, so the same app/exec dicts can be passed to
    several benchmarks.
    """
    if noise not in ("none", "depolarizing"):
        raise ValueError(f"noise must be 'none' or 'depolarizing', got {noise!r}")
    if int(max_failures) < 1:
        raise ValueError("max_failures must be at least one")
    if backend_id is None:
        backend_id = DEFAULT_BACKEND.get(api, "")

    kernel, backend_label = load_kernel(api, backend_id)
    noise_model = kernel.make_depolarizing_noise(p_1q=P_1Q, p_2q=P_2Q) if noise == "depolarizing" else None
    print(f"QUOPS Benchmark - {backend_label}")

    # shapes in the utility cone for the width sweep, capped at max_depth
    shapes = get_shapes(min_qubits, max_qubits, skip_qubits, max_depth)
    depth_note = "3 x width" if max_depth is None else ("no cap" if max_depth == 0 else max_depth)
    if not shapes:
        raise ValueError(f"No shapes in the utility cone with depth <= {depth_note} "
                         f"for widths {min_qubits}-{max_qubits}; raise max_depth")
    print(f"Scanning up to {len(shapes)} shapes (depth <= {depth_note}), "
          f"stopping after {max_failures} failure{'s' if max_failures > 1 else ''}")

    start_time = datetime.datetime.now()
    result = run_quops_benchmark(
        api,
        shapes,
        width_range=(min_qubits, max_qubits),
        zeta=1.0,
        num_circuits=max_circuits,
        num_mirrors=num_mirrors,
        shots=num_shots,
        noise_model=noise_model,
        rc_style="pauli_frame",
        alpha_per_test=DEFAULT_FAMILYWISE_ALPHA / int(max_failures),
        seed=seed,
        compile_once=compile_once,
    )
    end_time = datetime.datetime.now()
    print()
    print(result)
    print(f"Capability region: {result.capability_region()}")

    settings_label = (f"{'noiseless' if noise_model is None else f'depolarizing p1={P_1Q} p2={P_2Q}'} | "
                      f"K={max_circuits}, M={num_mirrors}, {num_shots} shots | "
                      f"familywise alpha=0.05"
                      + (f", max {max_failures} failures" if max_failures > 1 else ""))
    save_data(result, {
        "benchmark": "QUOPS",
        "api": api,
        "backend_id": backend_id,
        "backend_label": backend_label,
        "start_time": start_time.isoformat(timespec="seconds"),
        "end_time": end_time.isoformat(timespec="seconds"),
        "settings": {
            "min_qubits": min_qubits, "max_qubits": max_qubits, "skip_qubits": skip_qubits,
            "max_depth": max_depth, "max_circuits": max_circuits, "num_mirrors": num_mirrors,
            "seed": seed, "noise": noise, "p_1q": P_1Q if noise_model is not None else 0.0,
            "p_2q": P_2Q if noise_model is not None else 0.0, "max_failures": max_failures,
            "num_shots": num_shots, "compile_once": compile_once,
        },
        "settings_label": settings_label,
    })

    if plot_results:
        # plot_results (the argument) hides the function of the same name
        globals()["plot_results"](backend_id)

    return result


#######################
# MAIN

def get_args():
    parser = argparse.ArgumentParser(description="QUOPS Benchmark")
    parser.add_argument("--api", "-a", default="qiskit", help="Programming API: qiskit or cudaq", type=str)
    parser.add_argument("--backend_id", "-b", default=None,
                        help="CUDA-Q target (default nvidia; density-matrix-cpu for exact noise); for Qiskit only the image folder name (default qasm_simulator)", type=str)
    parser.add_argument("--num_shots", "-s", default=1000, help="Number of shots", type=int)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits (min = max = N)", type=int)
    parser.add_argument("--min_qubits", "-min", default=2, help="Minimum number of qubits (circuit width)", type=int)
    parser.add_argument("--max_qubits", "-max", default=6, help="Maximum number of qubits (circuit width)", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Step between widths", type=int)
    parser.add_argument("--max_depth", "-d", default=None,
                        help="Largest depth (layer pairs) to scan at each width; default 3 x width, 0 = no cap", type=int)
    parser.add_argument("--max_failures", default=1, help="Failed shapes before the scan stops (default 1)", type=int)
    parser.add_argument("--max_circuits", "-c", default=10, help="Random circuits per shape", type=int)
    parser.add_argument("--num_mirrors", default=4, help="Mirror samples per circuit", type=int)
    parser.add_argument("--seed", default=2026, help="Random seed", type=int)
    parser.add_argument("--noise", default="depolarizing", choices=["none", "depolarizing"],
                        help=f"Noise model: depolarizing (default; p1={P_1Q}, p2={P_2Q}) or none", type=str)
    parser.add_argument("--no_compile_once", action="store_true",
                        help="CUDA-Q: build and compile a new kernel per circuit (as in PRAQTICE; same results, ~100x slower)")
    parser.add_argument("--noplot", "-nop", action="store_true", help="Do not plot results")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits

    # save plots to files without opening windows
    import matplotlib
    matplotlib.use("Agg")

    try:
        run(min_qubits=args.min_qubits, max_qubits=args.max_qubits, skip_qubits=args.skip_qubits,
            max_depth=args.max_depth, max_circuits=args.max_circuits, num_mirrors=args.num_mirrors,
            seed=args.seed, noise=args.noise, max_failures=args.max_failures, api=args.api, backend_id=args.backend_id, num_shots=args.num_shots,
            compile_once=not args.no_compile_once, plot_results=not args.noplot)
    except ValueError as error:
        sys.exit(str(error))
