# Parallel and Cooperative Execution

When running quantum circuits on simulators or hardware, there are two ways to use multiple devices to your advantage: **parallel execution** for speed, and **cooperative execution** for scale.

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

## Cooperative Execution — Run Larger Circuits

Cooperative execution uses multiple GPUs together to implement a single circuit that would be too large for any one device. The GPUs cooperate to hold and manipulate one expanded statevector.

This mode is available with **CUDA-Q only** and is enabled automatically when running under MPI. For example, 4 GPUs with 32GB each provide a combined 128GB of memory, adding approximately 2 qubits of capacity beyond what a single GPU can handle.

**Important**: Cooperative execution is not parallelization. Only one circuit runs at a time — the GPUs are sharing the workload of a problem too large for a single device.

### Using Cooperative Execution

Cooperative execution is the default mode when running CUDA-Q under MPI without the `--parallel` flag:

```bash
# 4 GPUs cooperate on each circuit (default MPI behavior)
mpirun -np 4 python -m mpi4py benchmark.py -a cudaq
```

The `--gpus_per_circuit` (`-gpc`) flag provides fine-grained control over how many GPUs cooperate per circuit:

```bash
# All 4 GPUs cooperate per circuit (maximum statevector capacity)
mpirun -np 4 python -m mpi4py benchmark.py -a cudaq -gpc 4

# 2 GPUs cooperate per circuit, 2 circuits run in parallel
mpirun -np 4 python -m mpi4py benchmark.py -a cudaq -gpc 2

# 1 GPU per circuit, 4 circuits run in parallel (same as -p)
mpirun -np 4 python -m mpi4py benchmark.py -a cudaq -gpc 1
```

### Programmatic Control

```python
# Cooperative: 4 GPUs share the workload for each circuit
job_id, result = ex.execute_circuits(circuits, num_shots=1000, gpus_per_circuit=4)

# Parallel: each GPU runs a different circuit
job_id, result = ex.execute_circuits(circuits, num_shots=1000, gpus_per_circuit=1)
```

## Summary

| | Parallel (`-p`) | Cooperative (default MPI) |
|---|---|---|
| **Goal** | Speed — faster completion | Scale — larger circuits |
| **Circuits running simultaneously** | Multiple | One |
| **Devices per circuit** | One | Multiple |
| **Qiskit** | Map onto disjoint qubit regions | N/A |
| **CUDA-Q** | Distribute across GPUs | GPUs cooperate on statevector |
| **When to use** | Many circuits, moderate width | Few circuits, maximum width |
