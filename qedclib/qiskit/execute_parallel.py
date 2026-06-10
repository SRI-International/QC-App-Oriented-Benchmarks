# (C) Quantum Economic Development Consortium (QED-C) 2024.
# Technical Advisory Committee on Standards and Benchmarks (TAC)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################
# Parallel Execution Module - Qiskit (EXPERIMENTAL)
#
# Maps multiple circuits onto disjoint qubit regions of a single QPU
# for parallel execution, then decomposes results back to per-circuit counts.
#
# This module is loaded lazily from execute.py when parallel_execution is True.
# It accesses execute module state (sampler, backend, noise, etc.) via
# "import execute as ex".
###############################################################################

def _remove_measurements(circuit):
    """
    Return a copy of `circuit` with measurement/barrier operations removed,
    plus a list of which qubit indices were originally measured.

    The measured_qubits list is stored on the returned circuit as
    circuit._measured_qubits so CircuitExperiment can restore the
    original measurement pattern instead of using measure_all().
    """
    from qiskit import QuantumCircuit

    clean = QuantumCircuit(circuit.num_qubits, name=circuit.name)

    # Track which qubits had measurements in the original circuit
    measured_qubits = []
    for inst in circuit.data:
        op = inst.operation
        if op.name == "measure":
            qi = circuit.find_bit(inst.qubits[0]).index
            if qi not in measured_qubits:
                measured_qubits.append(qi)
            continue
        if op.name == "barrier":
            continue

        # Re-map the original instruction's qubits onto the new clean circuit.
        qargs = [clean.qubits[circuit.find_bit(q).index] for q in inst.qubits]
        clean.append(op.copy(), qargs)

    # Store the original measurement pattern for later restoration
    clean._measured_qubits = measured_qubits if measured_qubits else list(range(circuit.num_qubits))

    return clean

def _find_topology_partitions(coupling_map, circuit_width, num_partitions, gap=2,
                              backend_target=None, routing_buffer=0):
    """
    Find disjoint connected subgraphs on the device coupling map, separated
    by at least `gap` hops. Each partition has `circuit_width + routing_buffer`
    qubits — the extra qubits give the transpiler room to insert SWAP gates
    without escaping the partition.

    Scores candidates by gate error rates when available, with diameter and
    internal edge count as tiebreakers (favors compact, well-connected regions).

    Algorithm:
      1. Build undirected graph from the coupling map
      2. From each qubit, grow a connected subgraph of the target width,
         preferring neighbors with most connections to existing cluster
      3. Score each candidate by average 2-qubit gate error (from backend.target)
         plus compactness; falls back to compactness-only if no error data
      4. Greedily select non-overlapping partitions with gap separation

    Args:
        coupling_map: Qiskit CouplingMap or list of edges
        circuit_width: number of qubits the circuit uses
        num_partitions: how many partitions to find
        gap: minimum graph distance between any qubit in different partitions
        backend_target: optional Qiskit Target object (from backend.target)
            for error-rate scoring
        routing_buffer: extra qubits per partition for transpiler SWAP routing

    Returns:
        List of tuples, each tuple is a set of physical qubit indices forming
        one partition (size = circuit_width + routing_buffer).
        Length <= num_partitions (may be fewer if device is small).
        Returns empty list if coupling_map is None.
    """
    import networkx as nx

    if coupling_map is None:
        return []

    # Total qubits per partition: circuit qubits + routing buffer
    partition_size = circuit_width + routing_buffer

    # Build undirected graph from coupling map edges
    edges = coupling_map.get_edges() if hasattr(coupling_map, 'get_edges') else coupling_map
    G = nx.Graph()
    G.add_edges_from(edges)

    if partition_size == 1:
        # Trivial case: each partition is a single qubit
        nodes = sorted(G.nodes(), key=lambda n: -G.degree(n))
        partitions = []
        excluded = set()
        for n in nodes:
            if n not in excluded:
                partitions.append((n,))
                for nbr in nx.single_source_shortest_path_length(G, n, cutoff=gap):
                    excluded.add(nbr)
                if len(partitions) >= num_partitions:
                    break
        return partitions

    # Grow a connected subgraph of `size` nodes from `start`, preferring
    # the frontier node closest to the subgraph centroid (stays compact).
    # Ties broken by degree (more routing options).
    def _grow(start, size):
        subgraph = [start]
        sub_set = {start}
        frontier = set(G.neighbors(start))
        while len(subgraph) < size and frontier:
            # Score each frontier node: average graph distance to current subgraph members
            # (lower = closer to the cluster center)
            best = None
            best_score = float('inf')
            for f in frontier:
                avg_dist = sum(1 for s in subgraph if G.has_edge(f, s))
                # More direct connections to existing subgraph = better (negate to minimize)
                score = -avg_dist  # most connections first
                if score < best_score or (score == best_score and G.degree(f) > G.degree(best)):
                    best_score = score
                    best = f
            subgraph.append(best)
            sub_set.add(best)
            frontier.discard(best)
            for nbr in G.neighbors(best):
                if nbr not in sub_set:
                    frontier.add(nbr)
        return frozenset(sub_set) if len(sub_set) == size else None

    # Build edge error map from backend target (if available)
    edge_errors = {}
    if backend_target is not None:
        # Try common 2-qubit gate names
        for gate_name in ['cx', 'ecr', 'cz']:
            if gate_name in backend_target.operation_names:
                for qargs in backend_target.qargs_for_operation_name(gate_name):
                    props = backend_target[gate_name].get(qargs)
                    if props and props.error is not None:
                        edge_errors[qargs] = props.error
                        edge_errors[(qargs[1], qargs[0])] = props.error
                break  # use whichever 2q gate we find first

    # Generate candidate partitions from every starting node
    seen = set()
    candidates = []
    for start in G.nodes():
        sub = _grow(start, partition_size)
        if sub is not None and sub not in seen:
            seen.add(sub)
            sub_list = list(sub)

            # Connectivity analysis of the subgraph
            sub_G = G.subgraph(sub)
            internal_edges = sub_G.number_of_edges()
            # Diameter: max shortest path between any pair (lower = better routing)
            diameter = nx.diameter(sub_G)

            # Error score: average 2-qubit gate error on edges within subgraph
            if edge_errors:
                errs = []
                for u in sub_list:
                    for v in sub_list:
                        if u < v and G.has_edge(u, v):
                            errs.append(edge_errors.get((u, v), 0.1))
                error_score = sum(errs) / max(len(errs), 1)
            else:
                error_score = 0  # no error data, rely on connectivity alone

            # Sort priority:
            # 1. error_score (lower = better qubit quality)
            # 2. diameter (lower = shorter worst-case SWAP paths)
            # 3. -internal_edges (more edges = more routing options)
            candidates.append((error_score, diameter, -internal_edges,
                               tuple(sorted(sub_list))))

    # Sort by error, then diameter, then edge count (negated so more is better)
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))

    if candidates:
        scoring = "error+compactness" if edge_errors else "compactness-only"
        print(f"... partition candidates: {len(candidates)} found, scoring={scoring}")

    # Greedily pick non-overlapping partitions with gap separation
    selected = []
    excluded = set()
    for _err, _diam, _neg_edges, qubits in candidates:
        if any(q in excluded for q in qubits):
            continue
        selected.append(qubits)
        if edge_errors:
            print(f"...   selected {qubits}: avg_gate_err={_err:.4f}, "
                  f"diameter={_diam}, edges={-_neg_edges}")
        # Exclude all qubits within `gap` hops of this partition
        for q in qubits:
            for nbr, _dist in nx.single_source_shortest_path_length(G, q, cutoff=gap).items():
                excluded.add(nbr)
        if len(selected) >= num_partitions:
            break

    return selected


def _run_qiskit_parallel_experiment(circuits, num_shots):
    """
    Execute a batch of circuits using Qiskit's ParallelExperiment with simple
    sequential qubit allocation. Qiskit's transpiler handles layout/routing.

    Each circuit is assigned a disjoint range of physical qubits:
        circuit 0 → qubits [0, w0)
        circuit 1 → qubits [w0 + spacing, w0 + spacing + w1)
        ...

    The returned value is a list of count dictionaries, one per input circuit.
    """
    import time
    from qiskit_experiments.framework import ParallelExperiment, BaseExperiment, BaseAnalysis

    import execute as ex

    backend = ex.backend
    backend_name = ex.get_backend_name(backend)
    print(f"... parallel experiment using backend = {backend_name}")

    t0 = time.time()

    # For noisy simulator: wrap in AerSimulator with noise model so
    # ParallelExperiment's internal SamplerV2 picks up the noise.
    # For real hardware: use backend directly.
    if backend_name.endswith("qasm_simulator") and ex.noise is not None:
        from qiskit_aer import AerSimulator
        run_backend = AerSimulator(noise_model=ex.noise)
    else:
        run_backend = backend

    # Determine total qubit budget
    if hasattr(run_backend, 'num_qubits'):
        device_qubits = run_backend.num_qubits
    else:
        device_qubits = 1000  # simulator fallback

    # Qubit allocation: try topology-aware partitioning, retry with smaller gap,
    # fall back to sequential only for simulators (sequential is broken on
    # real hardware where consecutive qubits aren't necessarily connected)
    spacing = 2
    coupling_map = getattr(run_backend, 'coupling_map', None)

    # For same-width circuits, use topology partitioning if coupling map available
    widths = [c.num_qubits for c in circuits]
    backend_target = getattr(run_backend, 'target', None)
    partitions = []
    alloc_gap = spacing
    routing_buffer = 0  # ParallelExperiment requires physical_qubits == circuit size
    if coupling_map is not None and len(set(widths)) == 1:
        # Try with decreasing gap until we find enough partitions
        for try_gap in [spacing, 1, 0]:
            partitions = _find_topology_partitions(
                coupling_map, widths[0], len(circuits), gap=try_gap,
                backend_target=backend_target,
                routing_buffer=routing_buffer
            )
            alloc_gap = try_gap
            if len(partitions) >= len(circuits):
                break
            if try_gap > 0:
                print(f"... topology with gap={try_gap} found {len(partitions)} of "
                      f"{len(circuits)} needed, retrying with gap={max(try_gap-1, 0)}")

    if len(partitions) >= len(circuits):
        # Use topology-aware partitions
        physical_qubits_per_circuit = partitions[:len(circuits)]
        alloc_method = f"topology(gap={alloc_gap})"
    elif coupling_map is not None and partitions:
        # Hardware backend but not enough partitions even at gap=0:
        # use what we found and raise to trigger fallback to non-parallel
        print(f"... topology partitioning found only {len(partitions)} of "
              f"{len(circuits)} needed even at gap=0")
        print(f"... running {len(partitions)} circuits in parallel, "
              f"remaining {len(circuits) - len(partitions)} will use fallback")
        raise RuntimeError(
            f"Cannot fit {len(circuits)} circuits: only {len(partitions)} "
            f"partitions available on {device_qubits}-qubit device")
    elif coupling_map is None:
        # Simulator: sequential allocation is safe (all-to-all connectivity)
        physical_qubits_per_circuit = []
        offset = 0
        for circ in circuits:
            w = circ.num_qubits
            if offset + w > device_qubits:
                raise RuntimeError(
                    f"Circuits do not fit: need {offset + w} qubits, "
                    f"device has {device_qubits}")
            physical_qubits_per_circuit.append(tuple(range(offset, offset + w)))
            offset += w + spacing
        alloc_method = "sequential"
    else:
        # Hardware backend, no partitions found at all
        raise RuntimeError(
            f"Cannot find any {widths[0]}-qubit connected subgraphs "
            f"on {device_qubits}-qubit device")

    t1 = time.time()
    qubits_used = max(max(p) for p in physical_qubits_per_circuit) + 1 if physical_qubits_per_circuit else 0
    partition_size = len(physical_qubits_per_circuit[0]) if physical_qubits_per_circuit else 0
    buf_msg = f", routing_buffer={routing_buffer}" if routing_buffer > 0 and coupling_map is not None else ""
    print(f"... [timing] qubit allocation ({alloc_method}): {t1-t0:.3f}s "
          f"({len(circuits)} circuits, {partition_size}q partitions{buf_msg}, "
          f"{qubits_used} qubits used / {device_qubits})")
    if alloc_method.startswith("topology") and len(circuits) <= 6:
        for i, p in enumerate(physical_qubits_per_circuit):
            print(f"...   circuit {i} ({widths[i]}q) → {partition_size}q region {p}")

    # Minimal CircuitExperiment wrapper — no analysis needed
    class _NoAnalysis(BaseAnalysis):
        def _run_analysis(self, experiment_data):
            return [], []

    class _CircuitExperiment(BaseExperiment):
        def __init__(self, circuit, physical_qubits, label):
            super().__init__(
                physical_qubits=physical_qubits,
                analysis=_NoAnalysis(),
                backend=None,
            )
            # Keep the original circuit as-is, including its measurements.
            # No need to remove/re-add measurements — ParallelExperiment
            # handles the qubit remapping.
            self._circuit = circuit
            self._label = label

        def circuits(self):
            qc = self._circuit.copy()
            qc.name = self._label
            qc.metadata = {
                "component": self._label,
                "physical_qubits": self.physical_qubits,
            }
            return [qc]

    # Build experiments with original circuits.
    experiments = [
        _CircuitExperiment(
            circuit=circuits[i],
            physical_qubits=physical_qubits_per_circuit[i],
            label=getattr(circuits[i], "name", f"circuit_{i}"),
        )
        for i in range(len(circuits))
    ]

    parallel = ParallelExperiment(
        experiments=experiments,
        backend=run_backend,
        flatten_results=False,
    )
    parallel.set_transpile_options(optimization_level=1)

    t2 = time.time()
    print(f"... [timing] experiment setup: {t2-t1:.3f}s")

    # Try full-backend transpilation first (best fidelity). If the transpiler
    # routes through qubits outside our partitions (happens with deeper/wider
    # circuits), retry with pre-transpilation onto restricted coupling maps.
    try:
        expdata = parallel.run(backend=run_backend, shots=num_shots)
        expdata.block_for_results()
    except Exception as first_err:
        if "transpiled outside" not in str(first_err):
            raise

        print(f"... transpiler escaped partition bounds, retrying with restricted coupling maps")

        from qiskit.transpiler import CouplingMap
        from qiskit import transpile

        full_edges = coupling_map.get_edges() if coupling_map is not None else []
        transpiled_circuits = []
        for i, (circ, partition) in enumerate(zip(circuits, physical_qubits_per_circuit)):
            partition_set = set(partition)
            phys_to_local = {p: idx for idx, p in enumerate(partition)}
            local_edges = [
                (phys_to_local[u], phys_to_local[v])
                for u, v in full_edges
                if u in partition_set and v in partition_set
            ]
            local_coupling = CouplingMap(local_edges) if local_edges else None
            transpiled = transpile(
                circ, coupling_map=local_coupling, optimization_level=1
            )
            transpiled_circuits.append(transpiled)

        t_retry = time.time()
        print(f"... [timing] pre-transpile onto restricted maps: {t_retry-t2:.3f}s")

        experiments = [
            _CircuitExperiment(
                circuit=transpiled_circuits[i],
                physical_qubits=physical_qubits_per_circuit[i],
                label=getattr(circuits[i], "name", f"circuit_{i}"),
            )
            for i in range(len(circuits))
        ]
        parallel = ParallelExperiment(
            experiments=experiments,
            backend=run_backend,
            flatten_results=False,
        )
        parallel.set_transpile_options(optimization_level=0)

        expdata = parallel.run(backend=run_backend, shots=num_shots)
        expdata.block_for_results()

    t3 = time.time()
    print(f"... [timing] parallel.run + block_for_results: {t3-t2:.1f}s")

    # Extract per-circuit counts
    counts_list = []
    for i, child in enumerate(expdata.child_data()):
        datum = child.data(0)
        counts = datum.get("counts", datum)
        counts_list.append(counts)
        # Debug: show raw counts for first few circuits
        if i < 3:
            sample_keys = list(counts.keys())[:4] if isinstance(counts, dict) else str(type(counts))
            print(f"... [debug] child {i}: {len(counts)} entries, "
                  f"key_len={len(sample_keys[0]) if sample_keys else '?'}, "
                  f"samples={sample_keys}, "
                  f"expected_qubits={circuits[i].num_qubits}")

    print(f"... [timing] total _run_qiskit_parallel_experiment: {t3-t0:.1f}s")

    return counts_list


# ---- ORIGINAL VERSION (uses parallel_experiment/src/ for error-aware partitioning) ----
# Kept for reference — this was 125s hardware init + 58s mapping on ibm_fez (156 qubits).
# The src/ code does fidelity-aware partitioning (picks best qubit regions based on
# error rates), but the overhead is prohibitive for routine use.
# To re-enable, rename this to _run_qiskit_parallel_experiment and remove the one above.

def _run_qiskit_parallel_experiment_with_custom_partitioning(circuits, num_shots):
    """
    ORIGINAL: Execute circuits using custom partitioning from parallel_experiment/src/.
    Uses IBMQHardwareArchitecture, Floyd-Warshall distance matrices, heuristic
    partitioning, and SABRE-based initial mapping. Produces optimal qubit placement
    but takes ~180s on 156-qubit backends.
    """
    import os
    import sys
    import time
    import networkx as nx

    import execute as ex

    this_dir = os.path.dirname(os.path.abspath(__file__))
    pe_root = os.path.join(this_dir, "parallel_experiment")
    pe_src = os.path.join(pe_root, "src")

    if pe_root not in sys.path:
        sys.path.insert(0, pe_root)
    if pe_src not in sys.path:
        sys.path.insert(0, pe_src)

    from qiskit_experiments.framework import ParallelExperiment
    from qiskit.providers.fake_provider import GenericBackendV2

    from hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
    from hardware.distance_matrx import (
        get_distance_matrix_cnot_error_cost,
        get_qubit_readout_error,
    )
    from partition_process.qubit_partition import (
        partition_hardware_heuristic,
        largest_circuit_logical_degree,
        hardware_qubit_physical_degree,
    )
    from find_initial_mapping import (
        get_simultaneous_partition,
        compute_initial_mapping,
    )
    from run_parallel_experiment import (
        strip_idle_qubits,
        CircuitExperiment,
        extract_physical_qubits,
    )

    backend = ex.backend
    backend_name = ex.get_backend_name(backend)

    print(f"... parallel experiment using backend = {backend_name}")

    t0 = time.time()

    if backend_name == "qasm_simulator":
        hardware = IBMQHardwareArchitecture("ibmq_manhattan")
        run_backend = GenericBackendV2(
            num_qubits=hardware.qubit_number,
            coupling_map=None,
            basis_gates=["id", "rz", "sx", "x", "cx"],
            seed=42,
        )
    else:
        hardware = IBMQHardwareArchitecture(backend)
        run_backend = backend

    t1 = time.time()
    print(f"... [timing] hardware init: {t1-t0:.1f}s ({hardware.qubit_number} qubits)")

    hardware_graph = nx.DiGraph(hardware._coupling_graph)
    cnot_error_matrix = get_distance_matrix_cnot_error_cost(hardware)
    readout_error = get_qubit_readout_error(hardware)

    t2 = time.time()
    print(f"... [timing] floyd_warshall + error matrices: {t2-t1:.1f}s")

    total_qubits = sum(c.num_qubits for c in circuits)
    if total_qubits > hardware.qubit_number:
        raise RuntimeError("Circuits do not fit on the selected hardware model.")

    circuit_list = [_remove_measurements(c) for c in circuits]

    qubit_physical_degree, largest_physical_degree = hardware_qubit_physical_degree(
        hardware
    )
    largest_logical_degrees = [
        largest_circuit_logical_degree(c) for c in circuit_list
    ]

    accepted_circuits, multiple_partition = get_simultaneous_partition(
        circuit_list, hardware_graph, hardware,
        cnot_error_matrix, readout_error,
        qubit_physical_degree, largest_physical_degree,
        largest_logical_degrees,
        2, partition_hardware_heuristic, 0.1,
    )

    t3 = time.time()
    print(f"... [timing] partitioning: {t3-t2:.1f}s")

    if multiple_partition is None or len(accepted_circuits) != len(circuit_list):
        raise RuntimeError("Parallel partition failed or rejected some circuits.")

    circuit_partitions, per_circuit_mappings, _ = compute_initial_mapping(
        accepted_circuits, hardware, multiple_partition,
    )

    t4 = time.time()
    print(f"... [timing] initial mapping: {t4-t3:.1f}s")

    physical_qubits_per_circuit = extract_physical_qubits(
        accepted_circuits, circuit_partitions, per_circuit_mappings,
    )

    compact_circuits = [strip_idle_qubits(c) for c in accepted_circuits]

    experiments = [
        CircuitExperiment(
            circuit=compact_circuits[i],
            physical_qubits=physical_qubits_per_circuit[i],
            label=getattr(accepted_circuits[i], "name", f"circuit_{i}"),
        )
        for i in range(len(accepted_circuits))
    ]

    parallel = ParallelExperiment(
        experiments=experiments,
        backend=run_backend,
        flatten_results=False,
    )
    parallel.set_transpile_options(optimization_level=0)

    t5 = time.time()
    print(f"... [timing] experiment setup: {t5-t4:.1f}s")

    expdata = parallel.run(backend=run_backend, shots=num_shots)
    expdata.block_for_results()

    t6 = time.time()
    print(f"... [timing] parallel.run + block_for_results: {t6-t5:.1f}s")

    counts_list = []
    for i, child in enumerate(expdata.child_data()):
        datum = child.data(0)
        counts = datum.get("counts", datum)
        counts_list.append(counts)
        if i < 3:
            sample_keys = list(counts.keys())[:4] if isinstance(counts, dict) else str(type(counts))
            print(f"... [debug] child {i}: {len(counts)} entries, "
                  f"key_len={len(sample_keys[0]) if sample_keys else '?'}, "
                  f"samples={sample_keys}, "
                  f"expected_qubits={circuits[i].num_qubits}")

    print(f"... [timing] total: {t6-t0:.1f}s")

    return counts_list

def _localize_counts(counts, num_qubits):
    """
    Convert counts returned by Qiskit ParallelExperiment into the bitstring
    format expected by QED-C.

    ParallelExperiment may return child experiment counts with extra classical
    bits from the full combined circuit. QED-C expects each circuit's counts to
    contain only that circuit's local output bits.

    Example:
        For a 3-qubit circuit, ParallelExperiment may return "011000".
        QED-C expects only "011".
    """
    local = {}

    for key, value in counts.items():
        # Remove spaces that Qiskit may insert between classical registers.
        key = key.replace(" ", "")

        # Keep only the local measurement bits for this circuit.
        local_key = key[:num_qubits]

        # Merge counts in case multiple padded global keys map to the same
        # local bitstring.
        local[local_key] = local.get(local_key, 0) + value

    return local

def execute_circuits_parallel(circuits, num_shots):
    """
    Execute a list of QED-C circuits using the integrated Qiskit
    ParallelExperiment path.

    This function is the QED-C entry point for parallel circuit execution.
    Instead of calling QED-C's normal sequential execute_circuits() path, it
    attempts to run the input circuits together as one parallel experiment.

    The flow is:

        1. Receive a list of QED-C generated QuantumCircuit objects.
        2. Call _run_qiskit_parallel_experiment(), which:
             - removes final measurements,
             - partitions the hardware into disjoint qubit regions,
             - maps each circuit to one region,
             - wraps each circuit as a Qiskit Experiment,
             - runs them together with ParallelExperiment.
        3. Convert the returned counts into QED-C's expected per-circuit
           bitstring format.
        4. Wrap the counts in QED-C's ExecutionResult object.
        5. If anything fails, fall back to QED-C's original sequential execution.

    Args:
        circuits:
            List of Qiskit QuantumCircuit objects generated by QED-C.

        num_shots:
            Number of shots to execute for each circuit.

    Returns:
        A tuple of:

            ("parallel_experiment", ExecutionResult)

        on successful parallel execution, or the normal execute_circuits()
        return value if the parallel path fails.
    """
    import execute as ex

    if ex.verbose:
        print(f"... execute_circuits_parallel: {len(circuits)} circuits, {num_shots} shots")

    print(f">>> execute_circuits_parallel [qiskit]: {len(circuits)} circuits, {num_shots} shots")

    try:
        # Run the circuits through the custom multiprogramming +
        # Qiskit ParallelExperiment pipeline.
        print("Uses the integrated Qiskit ParallelExperiment workflow. If the parallel path fails, execution automatically falls back to the standard QED-C execution path.")
        counts_list = _run_qiskit_parallel_experiment(circuits, num_shots)

        # Debug: show counts before and after localization
        for i in range(min(3, len(counts_list))):
            raw = counts_list[i]
            localized = _localize_counts(raw, circuits[i].num_qubits)
            print(f"... [debug] circuit {i} ({circuits[i].num_qubits}q): "
                  f"raw={dict(list(raw.items())[:3])}, "
                  f"localized={dict(list(localized.items())[:3])}")

        # ParallelExperiment may return each child result with extra classical
        # bits from the full combined circuit. Convert each result back to the
        # local bitstring width expected by QED-C.
        counts_list = [
            _localize_counts(counts_list[i], circuits[i].num_qubits)
            for i in range(len(counts_list))
        ]

        # Convert the raw counts list into QED-C's standard result object so the
        # rest of the benchmark framework does not need to know that the circuits
        # were executed through ParallelExperiment.
        result = ex.ExecutionResult(counts_list)

        return "parallel_experiment", result

    except Exception as err:
        # If partitioning, mapping, Qiskit Experiments, or backend execution
        # fails, preserve QED-C behavior by falling back to the original
        # sequential execution path.
        import traceback
        print(f"... parallel experiment failed: {err}")
        traceback.print_exc()
        print("... falling back to normal QED-C execution")

        # Disable parallel_execution temporarily so execute_circuits() does not
        # recursively call execute_circuits_parallel().
        ex.parallel_execution = False
        try:
            return ex.execute_circuits(circuits, num_shots)
        finally:
            ex.parallel_execution = True


def execute_circuit_groups_parallel(circuit_groups, num_shots_list):
    """
    Execute circuit groups in parallel by mapping circuits from multiple groups
    onto disjoint qubit regions of a single QPU.

    When the real implementation is in place, this function will:
    1. Determine qubit allocation per group (based on max circuit width in group)
    2. For groups with the same shot count: compose one circuit from each group
       onto disjoint qubit regions and submit as a single job
    3. For groups with different shot counts: batch same-shot groups together,
       execute each batch as above
    4. Decompose combined results back to per-group, per-circuit counts

    Qubit allocation uses the widest circuit in each group to determine that
    group's region. Narrower circuits within a group use a subset of the
    allocated qubits.

    Currently a stub that calls back to execute_circuit_groups() sequentially.
    Replace the stub section below with the real implementation.

    Args:
        circuit_groups: list of lists of QuantumCircuit objects
        num_shots_list: list of ints, one per group

    Returns:
        (job_id, group_results) tuple:
        - job_id: identifier for the job
        - group_results: list of ExecutionResult, one per group
    """
    import execute as ex

    if ex.verbose:
        group_sizes = [len(g) for g in circuit_groups]
        print(f"... execute_circuit_groups_parallel: {len(circuit_groups)} groups, "
              f"sizes={group_sizes}, shots={num_shots_list}")

    #######################################################################
    # STUB: replace this section with the real parallel implementation.
    # The code below calls execute_circuit_groups() sequentially.
    # It temporarily disables parallel_execution to avoid recursion
    # back into this function.
    #######################################################################
    print(f">>> execute_circuit_groups_parallel [qiskit]: {len(circuit_groups)} groups")
    print(f"... [STUB] parallel group execution not yet implemented, executing sequentially")

    ex.parallel_execution = False
    try:
        result = ex.execute_circuit_groups(circuit_groups, num_shots_list=num_shots_list)
    finally:
        ex.parallel_execution = True

    return result
