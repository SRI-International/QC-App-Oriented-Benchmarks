# Parallel Execution Flow — `_run_qiskit_parallel_experiment()`

Detailed flow through `qedclib/qiskit/execute_parallel.py`.


## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     execute_circuits_parallel()                     │
│                         (public entry point)                        │
├─────────────────────────────────────────────────────────────────────┤
│  1. Call _run_qiskit_parallel_experiment(circuits, num_shots)       │
│  2. _localize_counts() on each result (trim extra classical bits)  │
│  3. Wrap in ExecutionResult                                        │
│  4. On failure → fall back to sequential execute_circuits()        │
└─────────────────────────────────────────────────────────────────────┘
```


## Stage 1: Entry and Grouping

`_group_circuits_by_width()` sorts circuits into groups by qubit width,
largest first. Each entry preserves the original index.

```
Input: N circuits, any mix of widths
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   _group_circuits_by_width(circuits)                │
│                                                                     │
│  defaultdict(list) keyed by num_qubits                             │
│  sorted descending by width (largest first)                        │
│  each entry: (orig_idx, circuit)                                   │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────┐    ┌──────────────────────────────────────┐
│   All Same Width (1 grp) │    │       Mixed Widths (N groups)        │
│                          │    │                                      │
│  width_groups = [        │    │  width_groups = [                    │
│    (8, [(0,c0),(1,c1),   │    │    (10, [(0,c0),(3,c3),(5,c5),...]), │
│         (2,c2),...])     │    │    ( 8, [(1,c1),(4,c4),...]),         │
│  ]                       │    │    ( 6, [(2,c2),(6,c6),...]),         │
│                          │    │    ( 4, [(7,c7),(9,c9),...]),         │
│  unique_widths = 1       │    │    ( 3, [(8,c8),(10,c10),...])       │
│  label: "same-width"     │    │  ]                                   │
│                          │    │  unique_widths = 5                    │
│                          │    │  label: "mixed-width"                 │
└──────────────────────────┘    └──────────────────────────────────────┘
```


## Stage 2: Path Selection

```
                    coupling_map = backend.coupling_map
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   coupling_map is not    │    │   coupling_map is None   │
│       None               │    │                          │
│                          │    │                          │
│   HARDWARE PATH          │    │   SIMULATOR PATH         │
│   Topology-aware         │    │   Sequential allocation  │
│   partitioning           │    │   on virtual qubits      │
│                          │    │                          │
│   budget = backend       │    │   budget = min(backend,  │
│     .num_qubits          │    │     parallel_simulator_  │
│   spacing = 2 (gap hops) │    │     max_qubits)          │
│                          │    │   spacing = 0            │
│   → Stage 3 (Hardware)   │    │   → Stage 3 (Simulator)  │
└──────────────────────────┘    └──────────────────────────┘
```


## Stage 3 (Hardware): Topology-Aware Partition Finding

### Gap Retry Logic

The system requests generous partition counts (one per circuit = max parallelism)
and lets device size limit naturally. If not all widths are covered, it reduces
the gap between partitions.

```
width_requests = [(w, len(group)) for each width_group]
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Try gap = 2 (default spacing) │
              │  _find_multi_width_partitions() │
              └───────────────┬───────────────┘
                              │
                    all widths covered?
                     ╱              ╲
                   YES               NO
                    │                 │
                    ▼                 ▼
              ┌──────────┐  ┌─────────────────────┐
              │   Done   │  │  Try gap = 1         │
              └──────────┘  │  _find_multi_width_  │
                            │  partitions()        │
                            └──────────┬──────────┘
                                       │
                             all widths covered?
                              ╱              ╲
                            YES               NO
                             │                 │
                             ▼                 ▼
                       ┌──────────┐  ┌─────────────────────┐
                       │   Done   │  │  Try gap = 0         │
                       └──────────┘  │  (last resort,       │
                                     │   partitions touch)  │
                                     └──────────┬──────────┘
                                                │
                                     any partitions found?
                                      ╱              ╲
                                    YES               NO
                                     │                 │
                                     ▼                 ▼
                               ┌──────────┐    ┌─────────────┐
                               │   Done   │    │ RuntimeError │
                               └──────────┘    └─────────────┘
```

### `_find_multi_width_partitions()` — Round-Robin Selection

Builds candidates for each width independently, then picks partitions in
round-robin order: one for the largest width, one for next largest, etc.
Repeats until device is full or all quotas met.

```
┌─────────────────────────────────────────────────────────────────────┐
│  For each width: grow connected subgraphs from every node          │
│  Score each candidate: (avg_gate_error, diameter, -internal_edges) │
│  Sort candidates best-first per width                              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ROUND-ROBIN SELECTION                            │
│                                                                     │
│  Widths sorted largest-first: [10q, 8q, 6q, 4q, 3q]               │
│                                                                     │
│  Round 1:  pick best 10q │ pick best 8q │ pick best 6q │          │
│            pick best  4q │ pick best 3q │                          │
│                                                                     │
│  Round 2:  pick next 10q │ pick next 8q │ pick next 6q │          │
│            pick next  4q │ pick next 3q │                          │
│                                                                     │
│  Round 3:  pick next 10q │ ... (device running out of room)        │
│            ─ skip 8q (no candidates left that don't overlap) ─     │
│            ... continue until no progress                          │
│                                                                     │
│  Each pick excludes qubits within `gap` hops of the partition      │
└─────────────────────────────────────────────────────────────────────┘
```

### Partition Scoring Detail

```
For each candidate subgraph:

  ┌──────────────────────────────────────────────────────────────────┐
  │  1. error_score = avg 2-qubit gate error on internal edges       │
  │     (from backend.target: cx/ecr/cz error rates)                │
  │     Falls back to 0 if no error data available                  │
  │                                                                  │
  │  2. diameter = max shortest path within subgraph                 │
  │     (lower = shorter worst-case SWAP chains)                    │
  │                                                                  │
  │  3. -internal_edges = negated edge count                         │
  │     (more edges = more routing options, better)                 │
  │                                                                  │
  │  Sort key: (error_score, diameter, -internal_edges)              │
  │             lower is better on all three                         │
  └──────────────────────────────────────────────────────────────────┘
```


## Stage 3 (Simulator): Sequential Allocation

No topology and no spacing — simulators have all-to-all connectivity, so
partitions are packed contiguously. The qubit budget comes from
`parallel_simulator_max_qubits` (default 16), not the backend's qubit count.

```
┌──────────────────────────────────────────────────────────────────┐
│  Qubit budget determination                                      │
│                                                                  │
│  Hardware: device_qubits = backend.num_qubits  (e.g., 156)      │
│  Simulator: device_qubits = min(backend.num_qubits,             │
│                                  parallel_simulator_max_qubits)  │
│             default parallel_simulator_max_qubits = 16           │
│             settable: ep.parallel_simulator_max_qubits = 24      │
└──────────────────────────────────────────────────────────────────┘

device_qubits = 16 (parallel_simulator_max_qubits)
spacing = 0 (simulator — no crosstalk)

offset = 0
  │
  ├─ width_group 5q (6 circuits):
  │    max_seq = (16 - 0) / 5 = 3 → num_p = min(3, 6) = 3
  │    partition 0: qubits (0..4),   offset → 5
  │    partition 1: qubits (5..9),   offset → 10
  │    partition 2: qubits (10..14), offset → 15
  │    → 6 circuits across 3 partitions = 2 rounds
  │
  ├─ width_group 3q (4 circuits):
  │    max_seq = (16 - 15) / 3 = 0 → doesn't fit!
  │    → RuntimeError (or pad into 5q partitions on hardware)
  │
  │  With max_qubits = 24:
  │    partition 0: qubits (0..4),   partition 1: qubits (5..9),
  │    partition 2: qubits (10..14), offset → 15
  │    partition 3: qubits (15..17), partition 4: qubits (18..20),
  │    partition 5: qubits (21..23), offset → 24
  │    → all 10 circuits in 2 rounds
  │
  └─ Simple same-width example: 4 circuits of 8q, max=16
       max_seq = 16 / 8 = 2 → 2 partitions
       partition 0: qubits (0..7),  partition 1: qubits (8..15)
       → 4 circuits across 2 partitions = 2 rounds
```


## Stage 4: Circuit Distribution

### Finding Target Partition Width (Hardware Path)

Circuits prefer exact-match partitions. If no partition matches the circuit
width, the circuit is padded to fit the smallest partition >= its width.

```
available_widths_asc = sorted(width_partitions.keys())

For each (width, group) in width_groups:
  │
  ├─ width in width_partitions and has entries?
  │     YES → target_w = width  (exact match)
  │     NO  → target_w = smallest available width >= circuit width
  │            If none large enough → RuntimeError
  │
  └─ If target_w != width:
       pad each circuit with _pad_circuit(circ, target_w)
```

### `_pad_circuit()` — Adding Idle Qubits

```
┌───────────────────────────┐         ┌───────────────────────────────┐
│  Original 3q circuit      │         │  Padded to 4q                 │
│                           │         │                               │
│  q0: ─H─●─M──            │  ──►    │  q0: ─H─●─M──                │
│  q1: ───X─M──            │         │  q1: ───X─M──                │
│  q2: ─H───M──            │         │  q2: ─H───M──                │
│                           │         │  q3: ─────────  (idle)        │
│  3 qubits, 3 clbits      │         │  4 qubits, 3 clbits          │
└───────────────────────────┘         └───────────────────────────────┘

Compose original onto qubits 0..N-1 of larger QuantumCircuit.
Extra qubits have no gates and no measurements.
```

### `_assign_to_partitions()` — Round-Robin Distribution

Within each target width, circuits are distributed round-robin across
partitions of that width.

```
Example: 12 circuits of 8q, 3 partitions of 8q

  Circuit index (within group):  0  1  2  3  4  5  6  7  8  9 10 11
  Assigned to partition:         A  B  C  A  B  C  A  B  C  A  B  C

  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │  Partition A      │  │  Partition B      │  │  Partition C      │
  │  circuits:        │  │  circuits:        │  │  circuits:        │
  │    c0, c3, c6, c9 │  │    c1, c4, c7,c10│  │    c2, c5, c8,c11│
  │  assignment_map:  │  │  assignment_map:  │  │  assignment_map:  │
  │    [0, 3, 6, 9]   │  │    [1, 4, 7, 10]  │  │    [2, 5, 8, 11]  │
  └──────────────────┘  └──────────────────┘  └──────────────────┘

  4 rounds of 3 parallel circuits = 4 composite circuits
```


## Stage 5: ParallelExperiment Composition

Each partition becomes a `_CircuitArrayExperiment`. ParallelExperiment
zips them by index — composite circuit i runs the i-th circuit from
every partition simultaneously on disjoint qubit regions.

```
┌──────────────────────────────────────────────────────────────────────┐
│  _CircuitArrayExperiment per partition                               │
│                                                                      │
│  Partition A (qubits 0-9):    [cA0, cA1, cA2, cA3]                 │
│  Partition B (qubits 14-21):  [cB0, cB1, cB2, cB3]                 │
│  Partition C (qubits 26-31):  [cC0, cC1, cC2]         ← 1 fewer    │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ParallelExperiment zips by index → composite circuits              │
│                                                                      │
│  composite_0 = Partition_A[0] ║ Partition_B[0] ║ Partition_C[0]     │
│                  (qubits 0-9)   (qubits 14-21)   (qubits 26-31)    │
│                                                                      │
│  composite_1 = Partition_A[1] ║ Partition_B[1] ║ Partition_C[1]     │
│                                                                      │
│  composite_2 = Partition_A[2] ║ Partition_B[2] ║ Partition_C[2]     │
│                                                                      │
│  composite_3 = Partition_A[3] ║ Partition_B[3]                      │
│                (C has no [3], ParallelExperiment handles ragged)     │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌───────────────────────┐
                 │  ONE JOB SUBMISSION   │
                 │  shots = num_shots    │
                 │  transpile opt_level=1│
                 └───────────────────────┘
                              │
        ┌─────────────────────┴──────────────────────┐
        │  If transpiler escapes partition bounds:    │
        │  Retry with pre-transpile onto restricted   │
        │  per-partition coupling maps (opt_level=0)  │
        └─────────────────────────────────────────────┘
```


## Stage 6: Result Extraction

```
┌──────────────────────────────────────────────────────────────────────┐
│  expdata.block_for_results()                                        │
│                                                                      │
│  expdata.child_data() → one child per partition                     │
│                                                                      │
│  For each partition_idx, child:                                     │
│    For each circuit_idx in assignment_map[partition_idx]:            │
│      datum = child.data(circuit_idx)                                │
│      counts = datum["counts"]                                       │
│      original_idx = assignment_map[partition_idx][circuit_idx]       │
│      counts_list[original_idx] = counts                             │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  _localize_counts(counts, num_qubits)                               │
│                                                                      │
│  ParallelExperiment returns bitstrings with all classical bits      │
│  from the composite circuit. Trim to local width.                   │
│                                                                      │
│  Example: 3q circuit in 10q composite                               │
│    raw key:     "0110000000"  (10 bits)                             │
│    localized:   "011"         (first 3 bits only)                   │
│                                                                      │
│  Merge collisions: if multiple global keys map to the same          │
│  local key, their counts are summed.                                │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
             ┌─────────────────────────────┐
             │  Return counts_list[0..N-1] │
             │  in original circuit order  │
             └─────────────────────────────┘
```


## Complete End-to-End Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  Input: N circuits (mixed widths)                                │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1: _group_circuits_by_width()                             │
│  → sorted groups, largest width first                            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  HARDWARE            │    │  SIMULATOR               │
│                      │    │                          │
│  Stage 3: find       │    │  Stage 3: sequential     │
│  multi-width         │    │  allocation              │
│  partitions          │    │  (offset accumulates)    │
│  (round-robin,       │    │                          │
│   gap retry)         │    │                          │
└──────────┬───────────┘    └────────────┬─────────────┘
           │                             │
           └──────────┬──────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 4: For each width group:                                  │
│    - Find target partition width (exact or closest larger)       │
│    - Pad if needed (_pad_circuit)                                │
│    - Round-robin distribute (_assign_to_partitions)              │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 5: Build _CircuitArrayExperiment per partition             │
│  → ParallelExperiment zips into composite circuits               │
│  → ONE job submission                                            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 6: child_data() → per-partition results                   │
│  assignment_map reorders to original indices                     │
│  _localize_counts() trims extra bits                             │
│  → counts_list[0..N-1] in original order                         │
└──────────────────────────────────────────────────────────────────┘
```


## Worked Example

**Input:** 30 circuits — 6x10q, 8x8q, 7x6q, 5x4q, 4x3q
**Device:** ibm_fez, 156 qubits, heavy-hex topology

### Step 1: Grouping

```
_group_circuits_by_width() produces:

  width_groups = [
    (10, [(c0,c5,c10,c15,c20,c25)]),           6 circuits
    ( 8, [(c1,c6,c11,c16,c21,c26,c28,c29)]),   8 circuits
    ( 6, [(c2,c7,c12,c17,c22,c27,c3)]),         7 circuits
    ( 4, [(c4,c8,c13,c18,c23)]),                 5 circuits
    ( 3, [(c9,c14,c19,c24)]),                    4 circuits
  ]

  "mixed-width: 6x10q, 8x8q, 7x6q, 5x4q, 4x3q"
```

### Step 2: Path Selection → Hardware (coupling_map exists)

```
  width_requests = [(10,6), (8,8), (6,7), (4,5), (3,4)]
  "partition requests (greedy fill): [10q:6, 8q:8, 6q:7, 4q:5, 3q:4]"
```

### Step 3: Partition Finding (gap=2)

```
  _find_multi_width_partitions(coupling_map, width_requests, gap=2, backend_target)

  Build candidates per width:
    10q: ~400 candidates (connected 10-node subgraphs)
     8q: ~600 candidates
     6q: ~800 candidates
     4q: ~1000 candidates
     3q: ~1200 candidates

  Round-robin selection on 156-qubit device:

  ┌─────────┬────────────────────────────────────────────────────────┐
  │  Round  │  Partition selected                                    │
  ├─────────┼────────────────────────────────────────────────────────┤
  │    1    │  10q: (23,24,25,26,33,34,35,36,45,46) err=0.0031     │
  │         │   8q: (62,63,64,72,73,74,82,83)       err=0.0028     │
  │         │   6q: (97,98,99,107,108,109)           err=0.0025     │
  │         │   4q: (120,121,130,131)                err=0.0022     │
  │         │   3q: (145,146,147)                    err=0.0019     │
  ├─────────┼────────────────────────────────────────────────────────┤
  │    2    │  10q: (0,1,2,3,12,13,14,15,18,19)     err=0.0035     │
  │         │   8q: (50,51,52,53,58,59,60,61)        err=0.0032     │
  │         │   6q: (88,89,90,91,92,93)              err=0.0029     │
  │         │   4q: (138,139,140,141)                err=0.0026     │
  │         │   3q: (152,153,154)                    err=0.0023     │
  ├─────────┼────────────────────────────────────────────────────────┤
  │    3    │  10q: (device full for 10q, skip)                     │
  │         │   8q: (103,104,105,106,113,114,115,116) err=0.0038   │
  │         │   6q: (6,7,8,9,10,11)                  err=0.0033    │
  │         │   4q: (42,43,44,48)                    err=0.0030    │
  │         │   3q: (126,127,128)                    err=0.0027    │
  ├─────────┼────────────────────────────────────────────────────────┤
  │   ...   │  continues until device exhausted or quotas met       │
  └─────────┴────────────────────────────────────────────────────────┘

  Result: 10q:2, 8q:3, 6q:3, 4q:3, 3q:3 = 14 partitions
  (fewer than requested because 156 qubits with gap=2 limits packing)
```

### Step 4: Circuit Distribution

```
  3q circuits (4 circuits) have no 3q partitions? They have 3 partitions.
  Exact match → no padding needed.

  If gap=2 had failed to cover 3q, and only 4q partitions existed:
    → 3q circuits padded to 4q: _pad_circuit(circ, 4)
    → merged into 4q target group

  Actual distribution (exact match for all widths):

  10q group (6 circuits, 2 partitions):
    ┌──────────────────────┐  ┌──────────────────────┐
    │  Part 0 (10q)        │  │  Part 1 (10q)        │
    │  c0, c10, c20        │  │  c5, c15, c25        │
    │  map: [0, 10, 20]    │  │  map: [5, 15, 25]    │
    └──────────────────────┘  └──────────────────────┘
    3 rounds

  8q group (8 circuits, 3 partitions):
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Part 2 (8q) │  │  Part 3 (8q) │  │  Part 4 (8q) │
    │  c1,c16,c29  │  │  c6,c21      │  │  c11,c26     │
    │  map:[1,16,29]│  │  map:[6,21]  │  │  map:[11,26] │
    └──────────────┘  └──────────────┘  └──────────────┘
    3 rounds (Part 3,4 have only 2 in last positions → ragged)

  6q group (7 circuits, 3 partitions):
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Part 5 (6q) │  │  Part 6 (6q) │  │  Part 7 (6q) │
    │  c2,c17,c3   │  │  c7,c22      │  │  c12,c27     │
    └──────────────┘  └──────────────┘  └──────────────┘
    3 rounds

  4q group (5 circuits, 3 partitions):
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Part 8 (4q) │  │  Part 9 (4q) │  │  Part10 (4q) │
    │  c4,c23      │  │  c8          │  │  c13         │
    └──────────────┘  └──────────────┘  └──────────────┘
    2 rounds

  3q group (4 circuits, 3 partitions):
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Part11 (3q) │  │  Part12 (3q) │  │  Part13 (3q) │
    │  c9,c24      │  │  c14         │  │  c19         │
    └──────────────┘  └──────────────┘  └──────────────┘
    2 rounds
```

### Step 5: Composite Circuits

```
  ParallelExperiment zips across all 14 partitions by index:

  ┌────────────────────────────────────────────────────────────────────┐
  │  composite_0:                                                      │
  │    Part0[0] ║ Part1[0] ║ Part2[0] ║ Part3[0] ║ Part4[0] ║        │
  │    Part5[0] ║ Part6[0] ║ Part7[0] ║ Part8[0] ║ Part9[0] ║        │
  │    Part10[0] ║ Part11[0] ║ Part12[0] ║ Part13[0]                  │
  │    → 14 sub-circuits on disjoint qubit regions                    │
  │    → uses ~120 of 156 qubits                                      │
  ├────────────────────────────────────────────────────────────────────┤
  │  composite_1:                                                      │
  │    Part0[1] ║ Part1[1] ║ Part2[1] ║ Part3[1] ║ Part4[1] ║        │
  │    Part5[1] ║ Part6[1] ║ Part7[1] ║ Part8[1] ║ Part11[1]         │
  │    (Parts 9,10,12,13 have no [1])                                 │
  │    → 10 sub-circuits                                               │
  ├────────────────────────────────────────────────────────────────────┤
  │  composite_2:                                                      │
  │    Part0[2] ║ Part1[2] ║ Part2[2] ║ Part5[2]                     │
  │    → 4 sub-circuits (most partitions exhausted)                   │
  └────────────────────────────────────────────────────────────────────┘

  3 composite circuits → ONE job submission → 30 results
```

### Step 6: Result Extraction

```
  expdata.child_data() → 14 children (one per partition)

  child[0] (Part 0, 10q): 3 results → counts_list[0], [10], [20]
  child[1] (Part 1, 10q): 3 results → counts_list[5], [15], [25]
  child[2] (Part 2, 8q):  3 results → counts_list[1], [16], [29]
  ...
  child[13] (Part13, 3q): 1 result  → counts_list[19]

  _localize_counts() on each:
    raw "011000...0" (120+ bits) → "011" (3 bits for 3q circuit)

  Final: counts_list[0..29], one dict per original circuit, correct order
```

### Summary

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  30 circuits (5 widths) on 156-qubit device                     │
  │                                                                 │
  │  Without parallelism:  30 jobs                                  │
  │  With parallelism:      3 composite circuits, 1 job             │
  │                                                                 │
  │  Speedup factor: ~10x fewer rounds, 30x fewer job submissions  │
  └─────────────────────────────────────────────────────────────────┘
```
