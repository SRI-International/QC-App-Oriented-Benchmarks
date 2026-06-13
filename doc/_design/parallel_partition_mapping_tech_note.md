# Technical Note: Lightweight Qubit Partition Mapping for Qiskit ParallelExperiment

**Date:** June 10, 2026
**Authors:** Tom Lubowe, with Claude (AI assistant)
**Status:** Working implementation, optimization ongoing
**Code:** `qedclib/qiskit/execute_parallel.py` — function `_find_topology_partitions()`

---

## 1. Problem Statement

When executing multiple quantum circuits in parallel on a single QPU using Qiskit's `ParallelExperiment`, each circuit must be assigned to a disjoint set of physical qubits. The quality of this assignment directly impacts circuit fidelity. Three approaches were evaluated:

1. **Sequential allocation** — assign qubits starting from qubit 0 (fast, poor fidelity)
2. **Full error-aware partitioning** — IBMQHardwareArchitecture + Floyd-Warshall + SABRE (optimal, 183s overhead)
3. **Lightweight topology + error scoring** — our approach (good fidelity, ~1-2s overhead)

## 2. Algorithm Design

### 2.1 Architecture: Two-Level Optimization

The system uses a two-level approach:
- **Level 1 (our algorithm):** Selects good qubit neighborhoods — connected, low-error, compact, well-separated regions of the device
- **Level 2 (Qiskit transpiler):** Handles fine-grained routing within each neighborhood — initial layout, SWAP insertion, gate decomposition

This separation is key: we pick the real estate, the Qiskit transpiler arranges the furniture.

### 2.2 Partition Finding Algorithm

**Function:** `_find_topology_partitions(coupling_map, circuit_width, num_partitions, gap, backend_target)`

**Step 1: Build device graph**
- Construct undirected NetworkX graph from backend's `CouplingMap` edges

**Step 2: Grow candidate subgraphs**
- From every qubit on the device, grow a connected subgraph of `circuit_width` qubits
- Growth strategy: greedily add the frontier node with the most direct connections to the existing cluster (keeps subgraphs compact, avoids chain-shaped partitions)
- Deduplicate candidates using frozenset hashing

**Step 3: Score candidates (multi-criteria)**
- **Primary: Average 2-qubit gate error** — read from `backend.target` for ECR/CX/CZ gates on edges within the subgraph. Lower error = better qubit quality.
- **Secondary: Diameter** — maximum shortest path between any pair in the subgraph. Lower diameter = shorter worst-case SWAP paths. This differentiates T-shapes (diameter 2) from chains (diameter 3+).
- **Tertiary: Internal edge count (negated)** — more edges within the subgraph = more routing options for the transpiler, fewer SWAPs needed.

**Step 4: Greedy non-overlapping selection**
- Sort candidates by (error, diameter, -edges)
- Greedily select partitions, excluding all qubits within `gap` hops of each selected partition
- Gap provides crosstalk isolation between parallel circuits

**Step 5: Gap retry**
- If not enough partitions found at gap=2, retry with gap=1, then gap=0
- Reports progress: "topology with gap=2 found 6 of 10 needed, retrying with gap=1"

### 2.3 Transpiler Escape Handling

**Problem:** ParallelExperiment requires `physical_qubits` to exactly match circuit qubit count. For deeper or wider circuits, the Qiskit transpiler (operating on the full device coupling map) may route through qubits outside the assigned partition, causing a `QiskitError: "Component experiment has been transpiled outside of the allowed physical qubits"`.

**Solution: Try/except with pre-transpile fallback**
1. **First attempt:** Let ParallelExperiment transpile on the full backend (optimization_level=1). This gives the transpiler maximum routing freedom and best fidelity.
2. **If "transpiled outside" error:** Retry with pre-transpilation. For each circuit, build a restricted coupling map containing only edges within the partition, remap to local qubit indices (0..N-1), and transpile onto this restricted map. Then pass the pre-transpiled circuits to ParallelExperiment with optimization_level=0 (no re-transpilation).

**Trade-off:** Full-backend transpilation gives better fidelity (transpiler can find optimal routing) but may escape partition bounds. Pre-transpilation guarantees containment but constrains routing, resulting in lower fidelity for circuits that need many SWAPs.

**Observed behavior by circuit width on ibm_fez (156 qubits):**
- 4-6 qubits: Full-backend transpile usually succeeds
- 8+ qubits: Often escapes, triggering pre-transpile fallback

### 2.4 Simulator Path

For simulators (no coupling map), sequential qubit allocation is used since simulators have all-to-all connectivity. This is detected via `coupling_map = getattr(run_backend, 'coupling_map', None)`.

## 3. Test Results on ibm_fez (156 qubits)

### 3.1 QFT Benchmark

| Qubits | Circuits | 2q Gates | Non-Parallel | Parallel | % of Baseline | Notes |
|--------|----------|----------|-------------|----------|--------------|-------|
| 4 | 3 | 1.3 | 0.943 | 0.920 | 97.6% | Best result, shallow circuits |
| 4 | 2 | 24.0 | 0.877 | 0.78-0.82 | ~90% | Deeper QFT variant |
| 6 | 2 | 60.0 | 0.538 | 0.39 | 72% | Pre-transpile not triggered |
| 6 | 4 | 60.0 | 0.559 | 0.35 | 63% | 4 partitions |
| 8 | 3 | — | 0.815 | 0.738 | 91% | Top 3 partitions |
| 8 | 6 | — | — | 0.578 | — | Bad 6th partition (0.1488 error) |
| 10 | 2 | 180 | 0.001 | 0.0 | N/A | Circuit too deep for hardware |
| 12 | 2 | 264 | 0.0 | 0.0 | N/A | Circuit too deep for hardware |

### 3.2 Hamlib TFIM Benchmark (Hamiltonian Simulation)

| Qubits | Circuits | 2q Gates | Non-Parallel | Parallel | % of Baseline | Notes |
|--------|----------|----------|-------------|----------|--------------|-------|
| 6 | 1→3 | 60.0 | 0.816 (H) / 0.933 | 0.735 (NF) / 0.902 (H) | 90% | Real use case |
| 8 | 3 | 80.0 | 0.546 (NF) / 0.772 (H) | 0.453 (NF) / 0.723 (H) | 83% | Pre-transpile fallback triggered |
| 10 | 3 | 100.0 | TBD | 0.117 (NF) / 0.483 (H) | TBD | Pre-transpile fallback, 100 2q gates |
| 12 | 3 | 120.0 | TBD | 0.0 (NF) / 0.165 (H) | TBD | Pre-transpile fallback, 120 2q gates |

*(H = Hellinger fidelity, NF = Normalized fidelity)*

**Note on hamlib observable path:** The fidelity method above uses full Trotter evolution circuits (deep). The observable estimation path (`-obs`) uses much shallower Pauli measurement circuits (basis rotations + measurements) and sends circuit groups via `execute_circuit_groups`. This is the primary target for parallel execution of H2/LiH molecules but requires mixed-width group handling (not yet implemented — see Avimita's batching work).

### 3.3 Key Observations

1. **Error scoring is critical.** Pure topology-only scoring picked edge-of-chip qubits (0,1,2,3) with 0.32 fidelity. Adding `backend.target` gate error rates shifted selection to low-error regions (~0.002 avg error) achieving 0.92 fidelity.

2. **Partition shape matters.** On heavy-hex topology (max degree 3), 8+ qubit subgraphs tend to be chain-shaped. Chains have high diameter, requiring many SWAPs for long-range gates. Diameter scoring helps prefer T-shapes and stars over chains.

3. **QFT is worst-case.** QFT has O(n^2) 2-qubit gates with all-to-all connectivity. At 10+ qubits, even non-parallel execution gives 0.0 fidelity on current hardware. Hamiltonian simulation circuits (the target use case) are much shallower.

4. **The transpiler escape problem scales with circuit width/depth.** Shallow 4-qubit circuits stay within partition bounds. Deeper 8+ qubit circuits often escape, triggering the pre-transpile fallback which constrains routing and reduces fidelity.

5. **100 shots is insufficient for deep circuits.** At 180+ 2-qubit gates, the correct answer has very low probability. 1000+ shots needed to distinguish signal from noise.

6. **ibm_fez has ~6 good 8-qubit partitions at gap=2.** The 6th partition jumps to 0.1488 avg gate error (70x worse than best). The device simply doesn't have many low-error 8-qubit neighborhoods.

## 4. Approaches Tried and Rejected

### 4.1 Routing Buffer (Extra Qubits Per Partition)
**Idea:** Find partitions of `circuit_width + buffer` qubits to give the transpiler SWAP routing room.
**Result:** ParallelExperiment requires `physical_qubits` length to exactly match circuit qubit count. Error: `"The length of the layout is different than the size of the circuit: 6 <> 4"`.
**Status:** Rejected — fundamental ParallelExperiment constraint.

### 4.2 Always Pre-Transpile
**Idea:** Pre-transpile every circuit onto its partition's restricted coupling map.
**Result:** Works for all circuit sizes (no crashes), but systematically reduces fidelity because the transpiler has fewer routing options on the restricted map. 4q QFT: 0.77 (vs 0.92 without pre-transpile).
**Status:** Rejected as default, kept as fallback for circuits that escape.

### 4.3 Topology-Only Scoring (No Error Rates)
**Idea:** Score partitions by compactness (average pairwise shortest path) only, no gate error data.
**Result:** Picked edge-of-chip qubits with terrible error rates. 4q fidelity: 0.32-0.56.
**Status:** Rejected — error scoring is essential.

### 4.4 High-Degree-First Growth
**Idea:** When growing subgraphs, prefer neighboring nodes with highest degree in the device graph.
**Result:** On heavy-hex, this jumps between distant degree-3 hub nodes, producing spread-out subgraphs.
**Status:** Replaced with most-connections-to-cluster growth strategy.

## 5. Open Improvements

### 5.1 Pre-Transpile Fidelity Gap
The restricted coupling map pre-transpile reduces fidelity by 10-25% compared to full-backend transpilation. Possible improvements:
- **Higher optimization_level** (2 or 3) for pre-transpilation — more time to find better routing within the restricted map
- **Expanded restricted map** — include 1-hop neighbors of partition qubits in the restricted coupling map, then verify the transpiled circuit stays within the original partition. If it escapes, retry with optimization_level=2.
- **Iterative approach** — transpile on full backend, check which qubits were used, if they escape by only 1-2 qubits, expand the partition to include them

### 5.2 Natural Partition Count + Batching
Instead of forcing gap=0 to fit N circuits, use however many gap=2 partitions exist (e.g., 6 for 8-qubit circuits), run those in parallel, then batch the remaining circuits in a second parallel run. This keeps partition quality high.

### 5.3 Offline Partition Map Caching
Pre-compute partition layouts for known backends (ibm_fez, ibm_torino) in JSON files. Load on demand at runtime for zero computation cost. Not needed yet since on-the-fly computation is fast (~1-2s).

### 5.4 Array Batching — DONE (June 12, 2026)
Implemented `_CircuitArrayExperiment` which holds an array of circuits per partition. ParallelExperiment natively zips by circuit index across partitions, creating one composite circuit per "round" — all submitted as a single job. This supersedes the separate-job batching approach.

**Results on ibm_fez:** 27 circuits of 8q across 11 partitions (gap=0), 3 rounds, 1 job, 45s total. Hamlib TFIM with 30 circuits of 8q: comparable fidelity to sequential, 3x less billed execution time (3s vs 10s).

### 5.5 Mixed-Width Array Batching (Phase 3) — DONE (June 12, 2026)
Circuits of different widths are handled in a single code path:

1. `_find_multi_width_partitions()` allocates partitions largest-first (Phase 1: one per unique width until device is full; Phase 2: additional partitions for widths that already have one)
2. Exact-match circuits are assigned to their width's partitions
3. Unmatched circuits are distributed round-robin across ALL partitions (not just the smallest), padded via `_pad_circuit()`, balancing the load evenly

Key design decisions:
- Phase 1 **stops** at the first width that doesn't fit — avoids wasting gaps on tiny partitions that overload a single bucket
- Unmatched circuits go to all partitions equally, not just the smallest-available — prevents one partition from being overloaded with 44 circuits while others have 3

**Simulator setting:** `parallel_simulator_max_qubits` (default 16) caps the qubit budget. Spacing is 0 (no crosstalk). On hardware, device size is used with gap-based spacing.

**Tested on ibm_fez:**
- 53 circuits (BV, widths 3-20): 4 partitions (20q, 19q, 18q, 17q), 41 unmatched distributed evenly → 14 rounds, 4s billed
- 42 circuits (Hamlib, widths 2-20): 4 partitions (20q, 18q, 16q, 14q), 30 unmatched distributed → 11 rounds, 43s total
- 57 circuits (QFT, widths 2-20): 4 partitions, 15 rounds, 4s billed

### 5.6 Subgraph Diversity
Current algorithm grows subgraphs from every starting node but the growth strategy is deterministic, so many starting nodes produce the same subgraph. Could try multiple growth strategies per starting node (random restarts, different heuristics) to find more diverse candidates.

## 6. Code Structure

```
qedclib/qiskit/execute_parallel.py
│
│ Module-level settings:
│   parallel_simulator_max_qubits = 16  # Caps simulator qubit budget
│
├── _find_topology_partitions()         # Single-width partition finding
│   ├── _grow()                         # Connected subgraph growth
│   ├── Edge error extraction           # From backend.target (ECR/CX/CZ)
│   ├── Candidate scoring               # (error, diameter, -edges)
│   └── Greedy selection with gap       # Non-overlapping, gap-separated
│
├── _find_multi_width_partitions()      # Multi-width partition finding
│   ├── _grow()                         # Same growth algorithm
│   ├── Per-width candidate generation  # Candidates sorted best-first per width
│   └── Round-robin selection           # One per width per round, largest first
│
├── _group_circuits_by_width()          # Group circuits, sorted largest-first
├── _pad_circuit()                      # Add idle qubits for width mismatch
├── _assign_to_partitions()             # Round-robin distribution with reorder map
│
├── _run_qiskit_parallel_experiment()
│   ├── Backend setup                   # AerSimulator wrapper for noisy sim
│   ├── Width grouping                  # _group_circuits_by_width()
│   ├── Partition allocation            # Hardware: multi-width topology
│   │                                   # Simulator: sequential, spacing=0
│   ├── Target-width mapping + padding  # Exact match or closest larger
│   ├── Circuit distribution            # Round-robin per target width group
│   ├── _CircuitArrayExperiment         # N circuits per partition
│   ├── Full-backend transpile          # First attempt (optimization_level=1)
│   ├── Pre-transpile fallback          # Restricted coupling map retry
│   └── Result reordering              # Via assignment_map
│
├── _localize_counts()                  # Extract per-circuit bitstrings
├── execute_circuits_parallel()         # QED-C entry point
└── execute_circuit_groups_parallel()   # Group-level stub
```

## 7. Dependencies
- `networkx` — graph operations (subgraph growth, diameter, shortest paths)
- `qiskit_experiments.framework` — ParallelExperiment, BaseExperiment, BaseAnalysis
- `qiskit.transpiler.CouplingMap` — for restricted coupling map construction (fallback path)
- `qiskit.transpile` — pre-transpilation (fallback path)
- No additional dependencies beyond what qedclib already requires
