# Parallel and Distributed Statevector Execution

When running quantum circuits on simulators or hardware, there are two ways to use multiple devices to your advantage: **parallel execution** for speed, and **distributed statevector** execution for scale.

## Parallel Execution — Run Circuits Faster

Parallel execution runs multiple circuits simultaneously on separate execution targets, reducing total wall-clock time.

- **Qiskit**: Multiple circuits are mapped onto disjoint qubit regions of a single QPU and executed simultaneously. For example, a 156-qubit device running 20-qubit circuits can execute approximately 6 at once.
- **CUDA-Q**: Circuits are distributed across multiple GPUs via MPI, with each GPU running a different circuit independently.

### Enabling Parallel Execution

**Command line:**

```bash
# Qiskit
python benchmark.py -a qiskit -p

# CUDA-Q (requires MPI)
mpirun -np 4 python -m mpi4py benchmark.py -a cudaq -p
```

**Programmatic:**

```python
import execute as ex

ex.parallel_execution = True
job_id, result = ex.execute_circuits(circuits, num_shots=1000)
```

If the system cannot parallelize (device has insufficient qubits, MPI not available, or only one GPU), execution proceeds sequentially. This is not an error — an informational message is printed and circuits run one at a time.

### Group-Level Parallel Execution

For workflows like Hamiltonian observable estimation, circuits are organized into groups where each group may require a different number of measurement shots. `execute_circuit_groups()` preserves this structure while enabling parallel execution across groups:

```python
ex.parallel_execution = True

circuit_groups = [group_a_circuits, group_b_circuits, group_c_circuits]
num_shots_list = [1000, 500, 200]   # different shots per group

job_id, group_results = ex.execute_circuit_groups(
    circuit_groups, num_shots_list=num_shots_list)
```

When parallel execution is enabled:
- **CUDA-Q**: Groups are distributed across GPUs. Each GPU processes its assigned groups sequentially, while multiple groups run simultaneously across GPUs.
- **Qiskit**: Circuits from groups with the same shot count are composed onto disjoint qubit regions for simultaneous execution.

### Qubit Width Requirements (Qiskit)

For qubit-mapped parallel execution, the device must have enough qubits to hold multiple circuits:

- **>= 2x max circuit width**: Circuits can be parallelized
- **>= 1x but < 2x**: Cannot parallelize — sequential execution with informational message
- **< 1x max circuit width**: Error — circuits are too wide for the device

Within a group, the widest circuit determines the qubit allocation. Narrower circuits use a subset of the allocated region.

### Qubit Partition Mapping (Qiskit)

When mapping multiple circuits onto disjoint qubit regions of a single QPU, the choice of which physical qubits to assign to each circuit has a significant impact on fidelity. We evaluated three approaches:

**Sequential allocation** assigns circuits to qubits starting from qubit 0: circuit 0 gets qubits [0, W), circuit 1 gets [W+gap, 2W+gap), and so on. This is fast (instant) but produces poor fidelity because it forces circuits onto whatever qubits happen to be numbered lowest, which are often at the edge of the chip with poor connectivity and high error rates. On ibm_fez (156 qubits), QFT at 4 qubits produced fidelity of 0.32–0.56 with this approach, compared to 0.94 for non-parallel execution where the transpiler freely selects the best qubits.

**Full error-aware partitioning** (using the `IBMQHardwareArchitecture` framework with Floyd-Warshall distance matrices, heuristic partition search, and SABRE-based initial mapping) achieves optimal qubit placement but takes approximately 183 seconds on ibm_fez (125s for hardware initialization + 58s for mapping). This overhead is prohibitive for routine use.

**Lightweight topology + error scoring** is the approach we implemented. The algorithm:

1. Builds an undirected graph from the backend's coupling map
2. From every qubit, grows a connected subgraph of the target circuit width by greedily adding the frontier node with the most connections to the existing cluster (keeps partitions compact)
3. Scores each candidate partition by average 2-qubit gate error rate (read directly from `backend.target`), with compactness as a tiebreaker
4. Greedily selects non-overlapping partitions, excluding all qubits within a configurable gap (default 2 hops) of each selected partition to reduce crosstalk

This approach runs in under 1 second (vs. 183s for full partitioning) because it avoids the expensive operations: no Floyd-Warshall (O(N³)), no SABRE mapping iterations, and no custom hardware initialization. The error rate data is read directly from the backend object, which is already loaded. The Qiskit transpiler handles routing within each partition at `optimization_level=1`.

**Results on ibm_fez (156 qubits):**

| Benchmark | Qubits | Non-Parallel Fidelity | Parallel Fidelity | % of Baseline |
|-----------|--------|----------------------|-------------------|--------------|
| QFT | 4 (3 circuits) | 0.943 | 0.920 | 97.6% |
| QFT | 4 (2 circuits, deeper) | 0.877 | 0.78–0.82 | ~90% |
| QFT | 8 (3 circuits) | 0.815 | 0.738 | 91% |
| QFT | 8 (27 circuits) | — | 0.545 | — |
| Hamlib TFIM | 6 (3 circuits) | 0.816 | 0.735 | 90% |
| Hamlib TFIM | 8 (3 circuits) | 0.546 | 0.453 | 83% |
| Hamlib TFIM | 8 (30 circuits) | comparable | comparable | ~100% |

For comparison, sequential allocation (qubits starting from 0) produced 0.32–0.56 fidelity, and the full error-aware partitioning approach took 183 seconds per run.

The lightweight approach achieves 83–98% of free-transpiler fidelity at negligible computational cost (~1-2 seconds). The key insight is that reading 2-qubit gate error rates from the backend target is essentially free, and even a simple scoring pass using this data dramatically improves partition quality. Scoring also considers subgraph diameter (lower = shorter SWAP paths) and internal edge count (more edges = better routing options), which helps avoid chain-shaped partitions on heavy-hex topologies.

For wider or deeper circuits where the transpiler may route through qubits outside the assigned partition, the system automatically falls back to pre-transpilation onto restricted coupling maps, trading some fidelity for guaranteed execution. See `doc/_design/parallel_partition_mapping_tech_note.md` for full technical details.

### Array Batching — Handling More Circuits Than Partitions

When the number of circuits exceeds the number of available partitions, the system uses **array batching**: circuits are distributed round-robin across partitions, and Qiskit's `ParallelExperiment` composes one wide circuit per "round" — all submitted as a single job.

For example, 30 circuits of width 8 on ibm_fez (11 partitions at gap=0):
- Each partition receives 2-3 circuits
- ParallelExperiment creates 3 composite circuits (one per round)
- All 3 composites are submitted as **one job** — one queue wait, one initialization
- Results are automatically decomposed back to 30 individual circuit results

This is significantly more efficient than submitting 30 individual jobs or even 3 separate parallel batches. In testing, Hamlib TFIM with 30 circuits of 8 qubits achieved comparable fidelity to sequential execution while using approximately **3x less billed execution time** (3 seconds vs 10 seconds on IBM Quantum).

### Mixed-Width Circuits

When the input contains circuits of different widths (common in benchmarks that sweep qubit counts, or any application that generates circuits of varying sizes), the system handles them in a single job:

1. **Group by width**, sorted largest-first (e.g., 3x20q, 3x18q, 3x16q, ...)
2. **Find partitions**, largest-first: one partition for the largest width, then the next largest, and so on until the device is full. Larger circuits get the best qubit regions.
3. **Exact-match circuits** go directly to their width's partitions.
4. **Unmatched circuits** (whose width has no partition) are padded with idle qubits and distributed round-robin across **all** available partitions, balancing the load evenly. A 3q circuit can run on a 20q partition — the extra qubits are idle.
5. **One job**: all partitions go into a single `ParallelExperiment`. Results are automatically decomposed, localized to the correct bit width, and reordered to match the original circuit order.

For example, 42 Hamlib circuits (widths 2-20, 3 each) on ibm_fez:
- 4 partitions found: 20q, 18q, 16q, 14q
- 12 exact-match circuits assigned directly
- 30 unmatched circuits (widths 2-12) distributed across all 4 partitions, padded
- 11 rounds max, **one job**, 43 seconds total

For simulators, the `parallel_simulator_max_qubits` setting (default 16) controls the qubit budget. Simulators use zero spacing between partitions (no crosstalk concerns).

```python
# Adjust simulator qubit budget if needed
import execute_parallel as ep
ep.parallel_simulator_max_qubits = 24  # allow wider parallel simulation
```

## Distributed Statevector Execution — Run Larger Circuits

Distributed statevector execution partitions and distributes a single circuit's statevector across multiple GPUs, enabling simulation of circuits that are too large for any one device.

This mode is available with **CUDA-Q only** and is enabled automatically when running under MPI. For example, 4 GPUs with 32GB each provide a combined 128GB of memory, adding approximately 2 qubits of capacity beyond what a single GPU can handle. This aligns with NVIDIA's `nvidia-mgpu` backend, which uses the cuStateVec library for multi-node, multi-GPU statevector distribution.

**Important**: Distributed statevector execution is not parallelization. Only one circuit runs at a time — the statevector is partitioned across GPUs to fit a larger problem in memory.

### Using Distributed Statevector Execution

Distributed statevector execution is the default mode when running CUDA-Q under MPI without the `--parallel` flag:

```bash
# 4 GPUs distribute the statevector (default MPI behavior)
mpirun -np 4 python -m mpi4py benchmark.py -a cudaq
```

The `--gpus_per_circuit` (`-gpc`) flag provides fine-grained control over how many GPUs participate per circuit:

```bash
# All 4 GPUs distribute the statevector (maximum capacity)
mpirun -np 4 python -m mpi4py benchmark.py -a cudaq -gpc 4

# 2 GPUs per statevector, 2 circuits run in parallel
mpirun -np 4 python -m mpi4py benchmark.py -a cudaq -gpc 2

# 1 GPU per circuit, 4 circuits run in parallel (same as -p)
mpirun -np 4 python -m mpi4py benchmark.py -a cudaq -gpc 1
```

### Programmatic Control

```python
# Distributed statevector: 4 GPUs partition the statevector for each circuit
job_id, result = ex.execute_circuits(circuits, num_shots=1000, gpus_per_circuit=4)

# Parallel: each GPU runs a different circuit
job_id, result = ex.execute_circuits(circuits, num_shots=1000, gpus_per_circuit=1)
```

## CUDA-Q Execution Modes: Our Approach vs NVIDIA's Built-In Options

NVIDIA's CUDA-Q provides two built-in multi-GPU backends that overlap with what qedclib provides. Understanding the differences helps clarify when to use which.

### NVIDIA's mgpu — Distributed Statevector

NVIDIA's `nvidia-mgpu` backend (now `nvidia` with `option="mgpu"`) partitions and distributes the statevector across GPUs using MPI. This is what qedclib uses by default when MPI is enabled — our `set_execution_target()` automatically sets the mgpu option. The `-gpc` flag controls this mode.

### NVIDIA's mqpu — Multiple Virtual QPUs

NVIDIA's `nvidia-mqpu` backend (now `nvidia` with `option="mqpu"`) treats each GPU as a separate virtual QPU. Circuits are dispatched to specific GPUs using `cudaq.sample_async(kernel, qpu_id=N)`, which returns a future. This mode does NOT require MPI — CUDA-Q detects available GPUs and manages them internally using threads.

NVIDIA provides two use cases for mqpu:
- **Parameter sweeps**: Run the same circuit with different parameters across GPUs
- **Hamiltonian distribution**: `cudaq.observe()` with `execution=cudaq.parallel.mpi` automatically distributes Hamiltonian terms across GPUs

### How qedclib's Parallel Execution Relates

| | qedclib (`-p` / `gpc=1`) | NVIDIA mqpu |
|---|---|---|
| **Target** | `nvidia` with `fp32` per MPI rank | `nvidia` with `mqpu` |
| **Distribution** | MPI rank assignment | `qpu_id` parameter or automatic |
| **Execution** | Synchronous `cudaq.sample()` | Async `cudaq.sample_async()` |
| **Who manages** | qedclib framework | User (explicit `qpu_id`) or cudaq (for `observe`) |
| **Requires MPI** | Yes (`mpiexec -np N`) | No for single node (thread mode) |
| **Multi-node** | Yes (MPI works across nodes) | Thread mode: no. MPI mode: yes. |
| **What's distributed** | Array of arbitrary circuits | Same kernel with different params, or Hamiltonian terms |

### What qedclib's Parallel Execution Provides

qedclib's `--parallel` (`-p`) option goes beyond what NVIDIA's built-in mqpu offers:

- **Multi-node and multi-GPU**: Distributes circuits across all available GPUs whether they are on the same node or spread across multiple nodes in a cluster. MPI handles cross-node communication transparently — the user calls `execute_circuits(circuit_array)` and the framework manages everything.

- **Arbitrary circuit arrays**: Distributes arrays of different circuits, each potentially with different structure and qubit widths. NVIDIA's mqpu is designed primarily for parameter sweeps (same kernel, many parameter sets) and requires explicit `qpu_id` management from the user.

- **Sampling-based observable estimation**: The `execute_circuit_groups()` function distributes groups of measurement circuits with per-group shot counts across GPUs. This enables sampling-based expectation value computation — measuring Pauli terms via circuit execution and classical post-processing — which models the behavior of real quantum hardware. NVIDIA's built-in `cudaq.observe()` with parallel distribution computes expectation values directly from the statevector, which is efficient for simulation but does not model the shot noise and measurement statistics of physical devices.

- **Framework-level transparency**: The user's code doesn't change between sequential and parallel execution — just set a flag. No need to manage `qpu_id` assignments, futures, or async patterns.

## Summary

| | Parallel (`-p`) | Distributed Statevector (default MPI) |
|---|---|---|
| **Goal** | Speed — faster completion | Scale — larger circuits |
| **Circuits running simultaneously** | Multiple | One |
| **Devices per circuit** | One | Multiple |
| **Qiskit** | Map onto disjoint qubit regions | N/A |
| **CUDA-Q** | Distribute circuits across GPUs | Distribute statevector across GPUs |
| **When to use** | Many circuits, moderate width | Few circuits, maximum width |
