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

# Maximum total qubits for simulator parallel execution.
# Controls how many partitions can run simultaneously: e.g., with max=16,
# two 8q circuits or three 5q circuits can run in parallel.
# Set higher (e.g., 24) if willing to wait for larger simulations.
# For hardware backends this is ignored — device size is used instead.
parallel_simulator_max_qubits = 16


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


def _find_multi_width_partitions(coupling_map, width_requests, gap=2,
                                  backend_target=None):
    """
    Find partitions for multiple circuit widths on the same device.

    Builds the device graph and error map once, generates candidate subgraphs
    for each requested width, then selects partitions using round-robin across
    widths: one partition for the largest width, one for the next largest, etc.,
    then back to the largest for a second partition, and so on until the device
    is full or all quotas are met. This ensures every width gets served before
    any width gets extra partitions.

    Args:
        coupling_map: Qiskit CouplingMap or list of edges
        width_requests: list of (width, num_partitions_wanted), sorted
            largest-first by caller
        gap: minimum graph distance between any qubit in different partitions
        backend_target: optional Qiskit Target for error-rate scoring

    Returns:
        dict: {width: [partition_tuples...]}
        Empty dict if coupling_map is None.
    """
    import networkx as nx

    if coupling_map is None:
        return {}

    # Build undirected graph from coupling map edges (once)
    edges = coupling_map.get_edges() if hasattr(coupling_map, 'get_edges') else coupling_map
    G = nx.Graph()
    G.add_edges_from(edges)

    # Build edge error map from backend target (once)
    edge_errors = {}
    if backend_target is not None:
        for gate_name in ['cx', 'ecr', 'cz']:
            if gate_name in backend_target.operation_names:
                for qargs in backend_target.qargs_for_operation_name(gate_name):
                    props = backend_target[gate_name].get(qargs)
                    if props and props.error is not None:
                        edge_errors[qargs] = props.error
                        edge_errors[(qargs[1], qargs[0])] = props.error
                break

    # Grow a connected subgraph of `size` nodes from `start`
    def _grow(start, size):
        subgraph = [start]
        sub_set = {start}
        frontier = set(G.neighbors(start))
        while len(subgraph) < size and frontier:
            best = None
            best_score = float('inf')
            for f in frontier:
                avg_dist = sum(1 for s in subgraph if G.has_edge(f, s))
                score = -avg_dist
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

    # Generate and score candidates for each width separately.
    # Each width gets its own sorted candidate list (best-first by error).
    candidates_by_width = {}
    for width, _count in width_requests:
        if width == 0:
            continue
        candidates = []
        seen = set()
        for start in G.nodes():
            sub = _grow(start, width)
            if sub is not None and sub not in seen:
                seen.add(sub)
                sub_list = list(sub)
                sub_G = G.subgraph(sub)
                internal_edges = sub_G.number_of_edges()
                diameter = nx.diameter(sub_G) if width > 1 else 0

                if edge_errors:
                    errs = []
                    for u in sub_list:
                        for v in sub_list:
                            if u < v and G.has_edge(u, v):
                                errs.append(edge_errors.get((u, v), 0.1))
                    error_score = sum(errs) / max(len(errs), 1)
                else:
                    error_score = 0

                candidates.append((error_score, diameter, -internal_edges,
                                   tuple(sorted(sub_list))))

        candidates.sort()  # best error first
        candidates_by_width[width] = candidates

    scoring = "error+compactness" if edge_errors else "compactness-only"
    cand_str = ", ".join(f"{w}q:{len(candidates_by_width.get(w, []))}"
                         for w, _ in width_requests)
    print(f"... partition candidates per width: [{cand_str}], scoring={scoring}")

    # Round-robin selection: cycle through widths (largest first per round),
    # picking one partition per width per round, until device is full or
    # all quotas met.
    quota = {width: count for width, count in width_requests}
    selected = {width: [] for width, _ in width_requests}
    excluded = set()
    pos = {width: 0 for width, _ in width_requests}  # scan position per width

    made_progress = True
    while made_progress:
        made_progress = False
        for width, _ in width_requests:  # largest-first order
            if quota[width] <= 0:
                continue
            # Scan this width's candidates for the next non-overlapping one
            cands = candidates_by_width.get(width, [])
            while pos[width] < len(cands):
                _err, _diam, _neg_edges, qubits = cands[pos[width]]
                pos[width] += 1
                if any(q in excluded for q in qubits):
                    continue
                # Found a valid partition for this width
                selected[width].append(qubits)
                quota[width] -= 1
                if edge_errors:
                    print(f"...   selected {width}q {qubits}: "
                          f"avg_gate_err={_err:.4f}, diameter={_diam}, "
                          f"edges={-_neg_edges}")
                for q in qubits:
                    for nbr in nx.single_source_shortest_path_length(
                            G, q, cutoff=gap):
                        excluded.add(nbr)
                made_progress = True
                break  # move to next width in this round

    total = sum(len(v) for v in selected.values())
    sel_str = ", ".join(f"{w}q:{len(selected[w])}" for w, _ in width_requests)
    print(f"... partitions selected: [{sel_str}], total={total}")

    return selected


def _pad_circuit(circuit, target_width):
    """
    Pad a circuit to target_width by adding idle qubits.

    The original circuit's gates and measurements stay on qubits 0..N-1.
    Extra qubits (N..target_width-1) are idle — no gates, no measurements.
    This allows a smaller circuit to run on a larger partition.
    """
    if circuit.num_qubits >= target_width:
        return circuit
    from qiskit import QuantumCircuit
    padded = QuantumCircuit(target_width, circuit.num_clbits, name=circuit.name)
    padded.compose(circuit, qubits=range(circuit.num_qubits),
                   clbits=range(circuit.num_clbits), inplace=True)
    return padded


def _group_circuits_by_width(circuits):
    """
    Group circuits by qubit width, preserving original indices.

    Returns:
        List of (width, [(orig_idx, circuit), ...]) sorted by width
        descending (largest first, so they get the best qubit regions).
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, circ in enumerate(circuits):
        groups[circ.num_qubits].append((idx, circ))
    return sorted(groups.items(), key=lambda x: -x[0])


def _assign_to_partitions(circuits, num_partitions):
    """
    Distribute circuits round-robin across partitions.

    Returns:
        partition_arrays: list of lists — partition_arrays[p] contains the
            circuits assigned to partition p.
        assignment_map: list of lists — assignment_map[p][i] is the original
            index (in the input circuits list) of the i-th circuit in partition p.
    """
    partition_arrays = [[] for _ in range(num_partitions)]
    assignment_map = [[] for _ in range(num_partitions)]
    for orig_idx, circ in enumerate(circuits):
        p = orig_idx % num_partitions
        assignment_map[p].append(orig_idx)
        partition_arrays[p].append(circ)
    return partition_arrays, assignment_map


def _run_qiskit_parallel_experiment(circuits, num_shots):
    """
    Execute circuits using Qiskit's ParallelExperiment with array batching.

    Supports both same-width and mixed-width circuit arrays. Circuits are
    grouped by width, partitions are found for each width (largest first),
    and circuits are distributed round-robin within each width group.
    ParallelExperiment composes one wide circuit per "round" (zipping the
    i-th circuit from each partition). All rounds submitted as a single job.

    Same-width example: 12 circuits of 5q, 3 partitions →
        Partition A (5q): [c0, c3, c6, c9]
        Partition B (5q): [c1, c4, c7, c10]
        Partition C (5q): [c2, c5, c8, c11]
        → 4 composite circuits, 1 job, 12 results

    Mixed-width example: 4x8q + 6x4q, device fits 2 partitions of 8q + 3 of 4q →
        Partition A (8q): [c0, c2]     Partition D (4q): [c4, c7]
        Partition B (8q): [c1, c3]     Partition E (4q): [c5, c8]
                                       Partition F (4q): [c6, c9]
        → 2 composite circuits, 1 job, 10 results

    The returned value is a list of count dictionaries, one per input circuit,
    in the original input order.
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

    # For simulators, cap to parallel_simulator_max_qubits to avoid
    # prohibitively slow simulation of wide composite circuits.
    coupling_map_check = getattr(run_backend, 'coupling_map', None)
    if coupling_map_check is None:
        device_qubits = min(device_qubits, parallel_simulator_max_qubits)
        print(f"... simulator qubit budget capped to {device_qubits} "
              f"(parallel_simulator_max_qubits={parallel_simulator_max_qubits})")

    # Qubit allocation: group circuits by width, find partitions for each width,
    # distribute circuits round-robin across partitions.
    # Works for both same-width and mixed-width input.
    spacing = 2
    coupling_map = getattr(run_backend, 'coupling_map', None)
    backend_target = getattr(run_backend, 'target', None)

    # Group circuits by width (largest first for best qubit regions)
    width_groups = _group_circuits_by_width(circuits)
    unique_widths = len(width_groups)
    width_summary = ", ".join(f"{len(g)}x{w}q" for w, g in width_groups)
    mixed_label = "mixed-width" if unique_widths > 1 else "same-width"
    print(f"... {len(circuits)} circuits ({mixed_label}): {width_summary}")

    # Flat lists built up across all width groups
    partitions = []        # qubit tuples, one per partition
    partition_arrays = []  # circuit arrays, one per partition
    assignment_map = []    # original index arrays, one per partition

    if coupling_map is not None:
        print(f"... hardware path: topology-aware partitioning on {device_qubits}-qubit device")

        # Greedy fill: find as many partitions as will fit on the device,
        # largest width first. Each width gets at least 1 partition (it must
        # fit since we go largest-first and the device is bigger than any
        # single circuit). Remaining capacity goes to more partitions of
        # any width, which reduces the number of rounds.
        #
        # Request a generous count per width — the function will return
        # however many actually fit. We ask for len(group) per width
        # (one partition per circuit = maximum parallelism) and let the
        # device constraint limit us naturally.
        width_requests = [(width, len(group)) for width, group in width_groups]

        req_str = ", ".join(f"{w}q:{c}" for w, c in width_requests)
        print(f"... partition requests (greedy fill): [{req_str}]")

        # Try with decreasing gap until every width has at least one partition
        alloc_gap = spacing
        width_partitions = {}
        for try_gap in [spacing, 1, 0]:
            width_partitions = _find_multi_width_partitions(
                coupling_map, width_requests, gap=try_gap,
                backend_target=backend_target
            )
            alloc_gap = try_gap
            # Need at least one partition, and ideally one per unique width.
            # With padding, we just need the largest width to have a partition
            # (smaller circuits can pad into it). But prefer all widths covered.
            total_found = sum(len(v) for v in width_partitions.values())
            all_covered = all(
                len(width_partitions.get(w, [])) > 0 for w, _ in width_groups
            )
            if all_covered or total_found > 0:
                break
            if try_gap > 0:
                missing = [f"{w}q" for w, _ in width_groups
                           if not width_partitions.get(w, [])]
                total_found = sum(len(v) for v in width_partitions.values())
                print(f"... gap={try_gap} found {total_found} partitions, "
                      f"missing widths={missing}, "
                      f"retrying with gap={max(try_gap-1, 0)}")

        # Check we found at least one partition overall
        if not any(width_partitions.values()):
            widths_str = ",".join(str(w) for w, _ in width_groups)
            raise RuntimeError(
                f"Cannot find any partitions for widths [{widths_str}] "
                f"on {device_qubits}-qubit device (gap={alloc_gap})")

        alloc_method = f"topology(gap={alloc_gap})"

        # Assign circuits to partitions:
        # 1. Exact-match: circuit width == partition width
        # 2. Closest-fit: circuit goes to smallest partition >= its width (padded)
        available_widths_asc = sorted(width_partitions.keys())

        # Map each circuit to its target partition width
        from collections import defaultdict
        target_groups = defaultdict(list)  # target_width -> [(orig_idx, circuit)]

        for width, group in width_groups:
            if width in width_partitions and width_partitions[width]:
                target_w = width  # exact match
            else:
                # Find smallest partition width >= circuit width
                target_w = None
                for aw in available_widths_asc:
                    if aw >= width:
                        target_w = aw
                        break
                if target_w is None:
                    raise RuntimeError(
                        f"No partition large enough for {width}q circuits "
                        f"(largest available: {available_widths_asc[-1]}q)")

            if target_w != width:
                print(f"... {len(group)} circuits of {width}q → padded into "
                      f"{target_w}q partitions")

            for orig_idx, circ in group:
                if target_w != width:
                    circ = _pad_circuit(circ, target_w)
                target_groups[target_w].append((orig_idx, circ))

        # Distribute within each target width group
        for target_w in sorted(target_groups.keys(), reverse=True):
            entries = target_groups[target_w]
            w_partitions = width_partitions[target_w]
            group_circuits = [circ for _, circ in entries]
            group_orig_indices = [idx for idx, _ in entries]
            w_arrays, w_map = _assign_to_partitions(group_circuits, len(w_partitions))
            for p_idx in range(len(w_partitions)):
                partitions.append(w_partitions[p_idx])
                partition_arrays.append(w_arrays[p_idx])
                assignment_map.append([group_orig_indices[i] for i in w_map[p_idx]])

    else:
        # Simulator: no spacing needed (all-to-all connectivity, no crosstalk).
        # Allocate partitions for each width that fits, largest first.
        # Smaller circuits that don't get their own partitions are padded
        # into larger ones (same logic as hardware path).
        print(f"... simulator path: sequential allocation on {device_qubits}-qubit device "
              f"(spacing=0)")
        offset = 0
        sim_width_partitions = {}  # width -> [partition_tuples]
        for width, group in width_groups:
            remaining = device_qubits - offset
            max_seq = remaining // max(width, 1)
            num_p = min(max(1, max_seq), len(group))
            if max_seq == 0:
                # No room for this width — will be padded into a larger partition
                print(f"... no room for {width}q partitions "
                      f"(offset={offset}), will pad into larger")
                continue

            w_partitions = []
            for _ in range(num_p):
                w_partitions.append(tuple(range(offset, offset + width)))
                offset += width
            sim_width_partitions[width] = w_partitions

        if not sim_width_partitions:
            raise RuntimeError(
                f"Cannot fit any circuits on {device_qubits}-qubit simulator")

        # Target-width mapping with padding (same as hardware path)
        from collections import defaultdict
        available_widths_asc = sorted(sim_width_partitions.keys())
        target_groups = defaultdict(list)

        for width, group in width_groups:
            if width in sim_width_partitions and sim_width_partitions[width]:
                target_w = width
            else:
                target_w = None
                for aw in available_widths_asc:
                    if aw >= width:
                        target_w = aw
                        break
                if target_w is None:
                    raise RuntimeError(
                        f"No partition large enough for {width}q circuits "
                        f"on simulator (largest: {available_widths_asc[-1]}q)")

            if target_w != width:
                print(f"... {len(group)} circuits of {width}q -> padded into "
                      f"{target_w}q partitions")

            for orig_idx, circ in group:
                if target_w != width:
                    circ = _pad_circuit(circ, target_w)
                target_groups[target_w].append((orig_idx, circ))

        for target_w in sorted(target_groups.keys(), reverse=True):
            entries = target_groups[target_w]
            w_partitions = sim_width_partitions[target_w]
            group_circuits = [circ for _, circ in entries]
            group_orig_indices = [idx for idx, _ in entries]
            w_arrays, w_map = _assign_to_partitions(group_circuits, len(w_partitions))
            for p_idx in range(len(w_partitions)):
                partitions.append(w_partitions[p_idx])
                partition_arrays.append(w_arrays[p_idx])
                assignment_map.append([group_orig_indices[i] for i in w_map[p_idx]])

        alloc_method = "sequential"

    if not partitions:
        raise RuntimeError("No partitions allocated for any circuit width")

    t1 = time.time()
    qubits_used = max(max(p) for p in partitions) + 1
    rounds = max(len(arr) for arr in partition_arrays)
    print(f"... [timing] qubit allocation ({alloc_method}): {t1-t0:.3f}s "
          f"({len(circuits)} circuits across {len(partitions)} partitions, "
          f"{rounds} rounds max, {qubits_used} qubits used / {device_qubits})")
    # Per-width summary
    for width, group in width_groups:
        w_parts = [i for i, p in enumerate(partitions) if len(p) == width]
        w_circs = sum(len(partition_arrays[i]) for i in w_parts)
        w_rounds = max((len(partition_arrays[i]) for i in w_parts), default=0)
        print(f"...   width {width}q: {len(w_parts)} partitions, "
              f"{w_circs} circuits, {w_rounds} rounds")
    if len(partitions) <= 10:
        for p_idx, partition in enumerate(partitions):
            print(f"...   partition {p_idx} ({len(partition)}q): qubits={partition}, "
                  f"{len(partition_arrays[p_idx])} circuits "
                  f"(original indices {assignment_map[p_idx]})")

    # Minimal experiment wrapper — no analysis needed.
    # _CircuitArrayExperiment holds an array of circuits per partition.
    # ParallelExperiment zips by index: composite[i] runs the i-th circuit
    # from each partition simultaneously. All composites submit as one job.
    class _NoAnalysis(BaseAnalysis):
        def _run_analysis(self, experiment_data):
            return [], []

    class _CircuitArrayExperiment(BaseExperiment):
        def __init__(self, circuits, physical_qubits, label):
            super().__init__(
                physical_qubits=physical_qubits,
                analysis=_NoAnalysis(),
                backend=None,
            )
            self._circuits = circuits
            self._label = label

        def circuits(self):
            result = []
            for i, circ in enumerate(self._circuits):
                qc = circ.copy()
                qc.name = f"{self._label}_{i}"
                qc.metadata = {
                    "component": self._label,
                    "index": i,
                    "physical_qubits": self.physical_qubits,
                }
                result.append(qc)
            return result

    # Build one experiment per partition, each holding its array of circuits.
    experiments = [
        _CircuitArrayExperiment(
            circuits=partition_arrays[p],
            physical_qubits=partitions[p],
            label=f"partition_{p}",
        )
        for p in range(len(partitions))
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
        expdata = parallel.run(backend=run_backend, sampler=getattr(ex, 'sampler', None), shots=num_shots)
        expdata.block_for_results()
    except Exception as first_err:
        if "transpiled outside" not in str(first_err):
            raise

        print(f"... transpiler escaped partition bounds, retrying with restricted coupling maps")

        from qiskit.transpiler import CouplingMap
        from qiskit import transpile

        full_edges = coupling_map.get_edges() if coupling_map is not None else []

        # Pre-transpile each partition's circuit array onto restricted coupling maps
        transpiled_partition_arrays = []
        for p, partition in enumerate(partitions):
            partition_set = set(partition)
            phys_to_local = {ph: idx for idx, ph in enumerate(partition)}
            local_edges = [
                (phys_to_local[u], phys_to_local[v])
                for u, v in full_edges
                if u in partition_set and v in partition_set
            ]
            local_coupling = CouplingMap(local_edges) if local_edges else None
            transpiled_array = [
                transpile(circ, coupling_map=local_coupling, optimization_level=1)
                for circ in partition_arrays[p]
            ]
            transpiled_partition_arrays.append(transpiled_array)

        t_retry = time.time()
        print(f"... [timing] pre-transpile onto restricted maps: {t_retry-t2:.3f}s")

        experiments = [
            _CircuitArrayExperiment(
                circuits=transpiled_partition_arrays[p],
                physical_qubits=partitions[p],
                label=f"partition_{p}",
            )
            for p in range(len(partitions))
        ]
        parallel = ParallelExperiment(
            experiments=experiments,
            backend=run_backend,
            flatten_results=False,
        )
        parallel.set_transpile_options(optimization_level=0)

        expdata = parallel.run(backend=run_backend, sampler=getattr(ex, 'sampler', None), shots=num_shots)
        expdata.block_for_results()

    t3 = time.time()
    print(f"... [timing] parallel.run + block_for_results: {t3-t2:.1f}s")

    # Extract per-circuit counts and reorder to original circuit order.
    # child_data() returns one child per partition; each child has one data
    # entry per circuit in that partition's array.
    counts_list = [None] * len(circuits)
    for partition_idx, child in enumerate(expdata.child_data()):
        for circuit_idx in range(len(assignment_map[partition_idx])):
            datum = child.data(circuit_idx)
            counts = datum.get("counts", datum)
            original_idx = assignment_map[partition_idx][circuit_idx]
            counts_list[original_idx] = counts

    # Debug: show first few results
    for i in range(min(3, len(counts_list))):
        counts = counts_list[i]
        if counts is not None and isinstance(counts, dict):
            sample_keys = list(counts.keys())[:4]
            print(f"... [debug] circuit {i} ({circuits[i].num_qubits}q): "
                  f"{len(counts)} entries, samples={sample_keys}")

    print(f"... [timing] total _run_qiskit_parallel_experiment: {t3-t0:.1f}s "
          f"({len(circuits)} circuits, {len(partitions)} partitions, {rounds} rounds)")

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
    Execute a list of QED-C circuits using Qiskit ParallelExperiment with
    array batching.

    This function is the QED-C entry point for parallel circuit execution.
    It distributes the input circuits across available hardware partitions
    using round-robin assignment, submits all circuits as a single
    ParallelExperiment job, and returns results in the original circuit order.

    The flow is:

        1. Receive a list of QED-C generated QuantumCircuit objects.
        2. Call _run_qiskit_parallel_experiment(), which:
             - finds disjoint qubit partitions (topology+error aware)
             - distributes circuits round-robin across partitions
             - wraps each partition as a _CircuitArrayExperiment
             - ParallelExperiment zips arrays into composite circuits
             - submits all composites as one job
             - extracts and reorders results to original circuit order
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
