# qedclib Developer Tests

Internal test scripts for exercising the qedclib execution API during development.
These are not user-facing examples — see [qedclib-examples](https://github.com/quantumcomputingdata/qedclib-examples) for that.

## Prerequisites

From the repo root: `pip install -e .`

## Scripts

| Script | Purpose | Launch |
|--------|---------|--------|
| `01_basic_api.py` | Smoke test: import, configure, execute, inspect results | `python 01_basic_api.py` |
| `02_parameter_sweep.py` | Batch execution with parameterized circuits (Ising model) | `python 02_parameter_sweep.py` |
| `03_backend_switching.py` | Backend switching via `-b` flag (simulator, IBM, IonQ, IQM) | `python 03_backend_switching.py -b qasm_simulator` |
| `04_batch_scaling.py` | Batch size and qubit count scaling, serial vs batch comparison | `python 04_batch_scaling.py` |
| `05_submit_with_metrics.py` | submit_circuits with metrics collection | `python 05_submit_with_metrics.py` |
| `11_execute_parallel.py` | Circuit-level parallel execution | See below |
| `12_execute_circuit_groups.py` | Group-level parallel execution | `python 12_execute_circuit_groups.py` |

## Parallel Execution Tests

### 11_execute_parallel.py — Circuit-Level Parallel

Tests `execute.parallel_execution = True` which causes `execute_circuits()` to
run circuits in parallel rather than sequentially.

**Qiskit:** Maps circuits onto disjoint qubit regions of a single QPU.
Currently a stub (sequential fallback). No special launch requirements.

```
python 11_execute_parallel.py -a qiskit
```

**CUDA-Q:** Distributes circuits across GPUs via MPI. Requires MPI launch
with >= 2 ranks, each bound to its own GPU.

```
mpiexec -np 2 python -m mpi4py 11_execute_parallel.py -a cudaq
```

Without MPI, the parallel test is skipped and only sequential execution runs.

### 12_execute_circuit_groups.py — Group-Level Parallel

Tests `execute_circuit_groups()` with groups of varying sizes, circuit widths,
and shot counts. Currently Qiskit only.

```
python 12_execute_circuit_groups.py -a qiskit
```

## Parallel Execution Modes

The `parallel_execution` flag on the execute module provides a consistent
interface across both backends:

```python
import execute as ex
ex.parallel_execution = True    # enable parallel execution
ex.execute_circuits(circuits, num_shots=1000)
```

### How parallel_execution interacts with MPI and gpus_per_circuit (CUDA-Q)

| `parallel_execution` | MPI ranks | `gpus_per_circuit` | Behavior |
|-----------------------|-----------|--------------------|----------|
| `False` | any | `None` | Sequential on single GPU (or mgpu expansion) |
| `True` | 1 | any | Sequential (only 1 rank, can't distribute) |
| `True` | N > 1 | `None` | Distribute circuits across N GPUs |
| `False` | N > 1 | `1` | Same — `gpc=1` also triggers distribution |
| `False` | N > 1 | `None` | Sequential despite having N GPUs |
| `False` | N > 1 | `M > 1` | Hybrid statevector sharing (not yet implemented) |

**Key points:**
- `parallel_execution = True` is equivalent to `gpus_per_circuit=1`
- Both require MPI with > 1 rank to actually parallelize
- `gpus_per_circuit > 1` is for statevector sharing (pooling GPUs for one large circuit) — a different mode entirely, not tested here
- Qiskit ignores `gpus_per_circuit`; CUDA-Q ignores `parallel_execution` when `gpus_per_circuit` is explicitly set

### Qiskit parallel execution

Qiskit parallelism works differently — it maps multiple circuits onto disjoint
qubit regions of a single QPU, executing them simultaneously. No MPI needed.

The device must have enough qubits:
- **>= 2x max circuit width**: can parallelize (map 2+ circuits)
- **>= 1x but < 2x**: warning, falls back to sequential
- **< 1x max circuit width**: error, circuits too wide for device
