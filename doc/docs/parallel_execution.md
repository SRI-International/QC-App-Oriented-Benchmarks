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

**Results on ibm_fez (156 qubits, QFT benchmark at 4 qubits, 3 circuits):**

| Approach | Partition Time | Qubits Selected | Fidelity |
|----------|---------------|-----------------|----------|
| Non-parallel (transpiler free choice) | — | transpiler picks best | 0.943 |
| Parallel + lightweight error scoring | 0.6s | (91,92,93,98), (130,131,132,133), (140,141,142,143) | 0.920 |
| Parallel + sequential allocation | instant | (0,1,2,3), (6,7,8,17), (11,18,31,32) | 0.32–0.56 |
| Parallel + full error-aware partitioning | 183s | noise-optimal regions | not tested with measurement fix |

The lightweight approach achieves within 2% of free-transpiler fidelity at negligible computational cost. The key insight is that reading 2-qubit gate error rates from the backend target is essentially free, and even a simple scoring pass using this data dramatically improves partition quality — the difference between selecting low-error qubit neighborhoods (avg gate error ~0.002) versus blindly using edge-of-chip qubits with potentially much higher error rates.

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
