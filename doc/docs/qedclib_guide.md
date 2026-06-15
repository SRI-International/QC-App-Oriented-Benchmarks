# qedclib Guide

**qedclib** is a quantum program execution engine with built-in performance monitoring. It provides backend abstraction, automatic metrics collection, batched execution, and multi-GPU support for quantum computing applications.

You can use qedclib independently of the QED-C benchmarks — it is available as a standalone package on [PyPI](https://pypi.org/project/qedclib/):

```bash
pip install qedclib
```

Or install from the full repository (includes qedcbench benchmarks):

```bash
git clone https://github.com/SRI-International/QC-App-Oriented-Benchmarks.git
cd QC-App-Oriented-Benchmarks
pip install -e .
```

## Initialization

Before using qedclib, initialize it with the quantum computing API you want to use. This loads the appropriate execution backend module.

```python
import qedclib

# Initialize with the API (loads the execute module)
qedclib.initialize("qiskit")

# Access execute and metrics directly from qedclib
ex = qedclib.execute
ex.set_execution_target(backend_id="qasm_simulator")

# Metrics is always available
qedclib.metrics.verbose = True
```

After `initialize()` or `get_kernel()`, the execute module is available as `qedclib.execute`. Use `ex = qedclib.execute` for a shorter reference. The `from qedclib import metrics` shorthand also works: `import qedclib.metrics as metrics`.

## Execution Paths

qedclib provides two execution paths depending on whether you want automatic metrics collection.

### Path 1: Direct Execution

Use `execute_circuits()` for raw execution. You get back results directly and handle timing yourself.

```python
from qiskit import QuantumCircuit

# Build circuits
circuits = []
for n in [3, 5, 8]:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    circuits.append(qc)

# Execute and get results
job_id, result = ex.execute_circuits(circuits, num_shots=1000)

# Process results
for i, counts in enumerate(result.get_counts()):
    print(f"Circuit {i}: {counts}")
```

### Path 2: Metrics-Integrated Execution

Use `submit_circuits()` for automatic metrics collection. Circuits are organized as a nested dict keyed by group (typically qubit width) and circuit ID. Timing, job IDs, and result processing are handled automatically.

```python
from qedclib import metrics

# Build circuits as a nested dict {group: {circuit_id: qc}}
circuits = {}
for n_qubits in [4, 6, 8]:
    group = str(n_qubits)
    circuits[group] = {}
    for cid in range(3):
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n_qubits), range(n_qubits))
        circuits[group][str(cid)] = qc

# Submit — auto-initializes metrics if needed
ex.submit_circuits(circuits, num_shots=1000)

# Finalize and retrieve metrics
metrics.end_metrics()
metrics.finalize_all_groups()

# Per-circuit metrics: {group: {circuit_id: {metric: value}}}
cm = metrics.get_circuit_metrics()

# Group-level averages with standard deviations
gm = metrics.get_group_metrics()
for i, group in enumerate(gm["groups"]):
    print(f"Group {group}: avg_exec={gm['avg_exec_times'][i]:.4f}s "
          f"+/- {gm['std_exec_times'][i]:.4f}s")
```

### Result Handlers

A result handler is a callback that processes each circuit's results as they arrive. Use `init_execution()` to register one before calling `submit_circuits()`.

```python
def my_handler(qc, result, group, circuit_id, num_shots):
    """Called for each circuit after execution."""
    counts = result.get_counts()
    # Compute and store your own metrics (fidelity, expectation values, etc.)
    fidelity = compute_fidelity(counts, expected)
    metrics.store_metric(group, circuit_id, "fidelity", fidelity)

ex.init_execution(my_handler)
ex.submit_circuits(circuits, num_shots=1000)
```

## Metrics Flow

The full metrics-integrated workflow:

```
qedclib.initialize("qiskit")
ex = qedclib.execute

ex.set_execution_target → ex.init_execution(handler)
    → ex.submit_circuits (one or more calls)
    → metrics.end_metrics()
    → metrics.finalize_all_groups()
    → metrics.save_app_metrics(benchmark_name, method=method)
    → metrics.get_group_metrics() / get_circuit_metrics()
    → metrics.plot_metrics("Title")
```

Note: `save_app_metrics` writes benchmark results to `__data/DATA-{backend_id}.json`. It is called explicitly in each benchmark's `run_circuits()` and is separate from plotting. The `plot_metrics` function is purely for visualization.

`submit_circuits` auto-calls `metrics.init_metrics()` if not yet initialized. You can call `submit_circuits` multiple times before finalizing — all results accumulate in the metrics module.

## Backend Configuration

### Local Simulators (Qiskit)

```python
ex.set_execution_target(backend_id="qasm_simulator")
```

### IBM Hardware

```python
ex.set_execution_target(
    backend_id="ibm_sherbrooke",
    project="your-crn-or-instance",
    exec_options={"use_ibm_quantum_platform": False}
)
```

### IonQ

```python
from qiskit_ionq import IonQProvider
provider = IonQProvider()
backend = provider.get_backend("ionq_simulator")

ex.set_execution_target(
    backend_id="ionq_simulator",
    provider_backend=backend
)
```

### CUDA-Q (GPU Simulator)

```python
qedclib.initialize("cudaq")
import execute as ex

ex.set_execution_target(backend_id="nvidia")
job_id, result = ex.execute_circuits(circuits, num_shots=1000)
```

## Parallel Execution

Set `ex.parallel_execution = True` to distribute circuits across multiple execution targets for faster completion. For group-level parallel execution with per-group shot counts, use `execute_circuit_groups()`.

For full details on parallel and distributed statevector execution modes, see [Parallel Execution](parallel_execution.md).

## API Reference

### Top-Level Functions (`qedclib`)

| Function | Description |
|----------|-------------|
| `initialize(api)` | Initialize qedclib: set API and load execution backend |
| `get_api()` | Get current quantum SDK name |
| `set_api(api)` | Set default quantum SDK (called automatically by `initialize`) |
| `get_kernel(name, api, benchmark)` | Load and return a benchmark kernel module |
| `is_leader()` | True if MPI rank 0 or MPI not active |

### Execution Functions (`execute as ex`)

| Function | Description |
|----------|-------------|
| `set_execution_target(backend_id, ...)` | Configure the backend for execution |
| `init_execution(handler)` | Register a result handler callback |
| `execute_circuits(circuits, num_shots)` | Execute a list of circuits, return `(job_id, result)` |
| `execute_circuit_groups(groups, num_shots_list)` | Execute groups of circuits with per-group shot counts |
| `submit_circuits(circuits, num_shots, max_batch_size, batch_by_group)` | Execute a circuit dict with automatic metrics collection |
| `process_circuit_results(circuits_info, results, ...)` | Map batch results back to individual circuits and store metrics |
| `compute_all_circuit_metrics(circuits)` | Compute depth, gate count, and transpiled metrics for a circuit dict |

### Metrics Functions (`from qedclib import metrics`)

| Function | Description |
|----------|-------------|
| `init_metrics()` | Initialize/reset metrics tracking |
| `end_metrics()` | Record end time for the execution run |
| `store_metric(group, circuit, name, value)` | Store a custom metric value |
| `get_metric(group, circuit, name)` | Retrieve a stored metric value |
| `get_circuit_metrics()` | Return the full per-circuit metrics dict |
| `get_group_metrics()` | Return the aggregated group-level metrics dict |
| `finalize_all_groups()` | Aggregate per-circuit metrics into group averages |
| `aggregate_metrics()` | Compute group averages (called by `finalize_all_groups`) |
| `report_metrics()` | Print a summary of all group metrics |
| `save_app_metrics(name, method)` | Save metrics to `__data/DATA-{backend_id}.json` |
| `plot_metrics(title)` | Generate volumetric benchmarking plots (no side effects) |

### Group Metrics Keys

After calling `finalize_all_groups()`, `get_group_metrics()` returns a dict with parallel arrays indexed by group. Key fields:

| Key | Description |
|-----|-------------|
| `groups` | Group names (typically qubit widths) |
| `avg_elapsed_times` / `std_elapsed_times` | Wall-clock time per circuit (mean / std) |
| `avg_exec_times` / `std_exec_times` | Backend execution time per circuit (mean / std) |
| `avg_create_times` / `std_create_times` | Circuit creation time (mean / std) |
| `avg_depths` | Average algorithmic circuit depth |
| `avg_tr_depths` / `avg_tr_xis` / `avg_tr_n2qs` | Transpiled depth, xi, and 2-qubit gate counts |
| `avg_fidelities` / `std_fidelities` | Normalized fidelity (mean / std) |
| `avg_hf_fidelities` / `std_hf_fidelities` | Hellinger fidelity (mean / std) |

## Examples

See the [qedclib-examples](https://github.com/quantumcomputingdata/qedclib-examples) repository for standalone usage examples.

Test scripts demonstrating various patterns are also included in `qedclib/_tests/`:

| Script | What it demonstrates |
|--------|---------------------|
| `01_basic_api.py` | Initialize, execute circuits, get counts |
| `02_parameter_sweep.py` | Parameterized circuits, batch execution |
| `03_backend_switching.py` | Configure different backends (simulator, IBM, IonQ, IQM) |
| `04_batch_scaling.py` | Batch size scaling, serial vs batch performance |
| `05_submit_with_metrics.py` | submit_circuits with automatic metrics collection |

<br>
&copy; 2025 Quantum Economic Development Consortium (QED-C). All Rights Reserved.
