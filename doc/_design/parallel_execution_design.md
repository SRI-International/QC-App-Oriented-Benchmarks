# Parallel and Multi-Device Execution — Design Notes

## Two User Goals

Users reach for multi-device execution for one of two distinct reasons:

### Goal 1: Speed — "Run my circuits faster"

The user has many circuits to execute and wants to reduce total wall-clock time. The solution is to run multiple circuits simultaneously on separate execution targets.

- **Qiskit**: Map multiple circuits onto disjoint qubit regions of a single large QPU. A 156-qubit device running 20-qubit circuits can execute ~6 simultaneously.
- **CUDA-Q**: Distribute circuits across multiple GPUs via MPI. Each GPU runs a different circuit independently.

In both cases, the user just wants "go parallel." The implementation details (qubit mapping vs GPU distribution) are handled by the execution engine.

If the system can't actually parallelize (Qiskit device too small, CUDA-Q without MPI or only 1 GPU), execution falls back to sequential automatically — not an error, just an informational message.

### Goal 2: Scale — "Run larger circuits"

The user has circuits that are too wide for a single device and needs more qubits (or more GPU memory). The solution is to distribute the statevector across multiple devices.

- **Qiskit**: Not currently applicable (QPUs have fixed qubit counts; circuit cutting is a different approach).
- **CUDA-Q**: The statevector is partitioned and distributed across multiple GPUs (NVIDIA's mgpu backend). 4 GPUs with 32GB each give the memory of 128GB, adding ~2 qubits of capacity.

This is NOT parallelization — only one circuit runs at a time. The statevector is distributed across GPUs to fit a problem too large for any single device.

## Why "Parallel" Is Confusing

The CUDA-Q multi-GPU statevector distribution is sometimes called "parallel" because the GPUs work together simultaneously. But from the user's perspective, it's the opposite of parallel execution:

| | Goal 1: Speed | Goal 2: Scale |
|---|---|---|
| Circuits running simultaneously | Multiple | One |
| Devices per circuit | One | Multiple |
| User wants | Faster completion | Bigger problems |
| CUDA-Q mechanism | MPI circuit distribution (mqpu) | MPI statevector distribution (mgpu) |
| Qiskit mechanism | Qubit mapping | N/A |

We use "parallel" exclusively for Goal 1 (speed). For Goal 2, we use "distributed statevector" — aligning with NVIDIA's terminology for mgpu mode ("partition and distribute the state vector").

## CLI Interface (Design Discussion)

### For Speed (Goal 1): `--parallel` / `-p`

A simple flag that enables parallel circuit execution:

```bash
# Qiskit — maps circuits onto disjoint qubit regions
python benchmark.py -a qiskit -p

# CUDA-Q — distributes circuits across GPUs (requires MPI)
mpirun -np 4 python -m mpi4py benchmark.py -a cudaq -p
```

When `-p` is set:
- Qiskit: `execute.parallel_execution = True` → routes to qubit-mapped execution
- CUDA-Q: `execute.parallel_execution = True` → equivalent to `gpus_per_circuit=1`, distributes circuits across MPI ranks

If parallelization isn't possible (insufficient qubits, no MPI, single GPU), execution proceeds sequentially with an informational message.

### For Scale (Goal 2): `--gpus_per_circuit` / `-gpc`

Controls how many GPUs participate in distributing the statevector per circuit (CUDA-Q only):

```bash
# All 4 GPUs distribute the statevector (maximum capacity)
mpirun -np 4 python -m mpi4py benchmark.py -a cudaq -gpc 4

# 2 GPUs per statevector (2 circuits can run in parallel on 4 GPUs)
mpirun -np 4 python -m mpi4py benchmark.py -a cudaq -gpc 2
```

Note: `-gpc 1` is equivalent to `-p` for CUDA-Q.

### Interaction Between Flags

| Flags | Behavior |
|---|---|
| (none) | Sequential. CUDA-Q with MPI defaults to distributed statevector (all GPUs per circuit). |
| `-p` | Parallel circuit execution. Each device runs one circuit independently. |
| `-gpc N` | N GPUs distribute the statevector per circuit. `N=1` is parallel; `N=total` is full distribution; between is hybrid. |
| `-p -gpc N` | `-p` is redundant when `-gpc` is specified — `gpc` controls the mode precisely. |

### Default MPI Behavior (CUDA-Q)

When running under MPI without any flags, CUDA-Q defaults to distributed statevector mode (all GPUs contribute to one circuit). This is the "scale" mode, not the "speed" mode. Users who want speed must explicitly request `-p` or `-gpc 1`.

This default exists because statevector distribution is the safer choice — it works for any circuit width. Parallel distribution requires that each circuit fits on a single GPU, which may not be true for large simulations.

## Programmatic Interface

```python
import execute as ex

# Goal 1: Speed — parallel execution
ex.parallel_execution = True
job_id, result = ex.execute_circuits(circuits, num_shots=1000)

# Goal 1: Speed — parallel groups (different shot counts per group)
ex.parallel_execution = True
job_id, group_results = ex.execute_circuit_groups(
    circuit_groups, num_shots_list=[1000, 500, 200])

# Goal 2: Scale — distributed statevector (CUDA-Q only, via CLI or gpus_per_circuit)
job_id, result = ex.execute_circuits(
    circuits, num_shots=1000, gpus_per_circuit=4)
```

## Group-Level Execution

`execute_circuit_groups()` executes groups of circuits where each group can have a different shot count. This is essential for workflows like Hamiltonian observable estimation, where Pauli commuting groups are measured with shots weighted by coefficient magnitude.

```python
# 3 groups, different shot counts
circuit_groups = [
    [circuit_a1, circuit_a2],   # high-weight terms
    [circuit_b1],               # medium-weight terms  
    [circuit_c1, circuit_c2, circuit_c3],  # low-weight terms
]
num_shots_list = [1000, 500, 200]

job_id, group_results = ex.execute_circuit_groups(
    circuit_groups, num_shots_list=num_shots_list)
```

When `parallel_execution` is True:
- **CUDA-Q**: Groups distributed across GPUs. Each GPU processes its assigned groups sequentially, but multiple groups run simultaneously across GPUs.
- **Qiskit**: Circuits from different groups (with the same shot count) composed onto disjoint qubit regions for simultaneous execution.

When `parallel_execution` is False (or parallelization not available):
- Groups execute sequentially, each group's circuits passed to `execute_circuits()`.

## Qubit Width Considerations (Qiskit)

For qubit-mapped parallel execution, the device must have enough qubits to hold multiple circuits simultaneously:

- **>= 2x max circuit width**: Can parallelize (map 2+ circuits)
- **>= 1x but < 2x**: Cannot parallelize — sequential fallback with informational message
- **< 1x max circuit width**: Error — circuits too wide for the device

Within a group, the widest circuit determines the qubit allocation for that group. Narrower circuits use a subset of the allocated region.

## Current Implementation Status

| Feature | Qiskit | CUDA-Q |
|---|---|---|
| Circuit-level parallel (`-p`) | Stub (sequential fallback) | Working via MPI |
| Group-level parallel | Stub (sequential fallback) | Stub (circuit-level works within groups) |
| Distributed statevector (`-gpc N`) | N/A | Working (default MPI behavior) |
| `parallel_execution` flag | Yes | Yes |
| `execute_circuit_groups()` | Yes | Yes |

Qiskit parallel implementation (qubit mapping via ParallelExperiment) is in development.
