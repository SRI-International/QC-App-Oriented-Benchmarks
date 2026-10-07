# qedclib

A quantum program execution engine with built-in performance monitoring. Execute quantum circuits at scale across multiple backends with automatic metrics collection, parallel GPU execution, and detailed per-circuit timing.

## Install

```bash
pip install qedclib
```

## Quick Example

```python
import qedclib

# Initialize with the quantum computing API
qedclib.initialize("qiskit")
ex = qedclib.execute

# Configure the execution backend
ex.set_execution_target(backend_id="qasm_simulator")

# Build and execute circuits
from qiskit import QuantumCircuit

circuits = []
for n in [3, 5, 8]:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    circuits.append(qc)

job_id, result = ex.execute_circuits(circuits, num_shots=1000)

for i, counts in enumerate(result.get_counts()):
    print(f"Circuit {i}: {sorted(counts.items(), key=lambda x: -x[1])[:3]}")
```

## Features

- **Multi-backend execution** — run circuits on Qiskit simulators (Aer), IBM hardware, IonQ, IQM, and CUDA-Q (local GPU, multi-GPU, NVIDIA Quantum Cloud) through a single API
- **Automatic performance metrics** — execution time, elapsed time, circuit depth, gate counts, and fidelity metrics are collected per-circuit and aggregated per-group, with standard deviations
- **Per-circuit timing** — accurate individual circuit timing extracted from backend-specific result objects (simulator metadata, IBM execution spans, hardware elapsed time)
- **Batched execution** — memory-efficient execution for large sweeps via `max_batch_size`, with automatic create-execute alternation to control memory pressure on GPU backends
- **Multi-GPU parallel execution** — distribute circuits across GPUs using MPI with configurable modes (one GPU per circuit, multiple GPUs per circuit, or hybrid)
- **Robust job management** — automatic retry with configurable limits, job status polling with comfort indicators, result count validation, and graceful handling of cancelled or failed jobs
- **Result handlers** — plug in custom per-circuit processing (fidelity computation, expectation values, etc.) that runs automatically as results arrive
- **Volumetric benchmarking plots** — built-in visualization of performance across circuit widths and depths

## Documentation

- **[qedclib Guide](https://sri-international.github.io/QC-App-Oriented-Benchmarks/qedclib_guide/)** — API reference, execution paths, metrics flow, and backend configuration
- **[Full Documentation](https://sri-international.github.io/QC-App-Oriented-Benchmarks/)** — includes the benchmark suite, setup guides, and platform-specific instructions

## Examples

See [qedclib-examples](https://github.com/quantumcomputingdata/qedclib-examples) for standalone usage examples including parameter sweeps, backend switching, batch scaling, and metrics collection.

## Part of the QED-C Benchmarks

qedclib is the execution engine behind the [QED-C Application-Oriented Benchmarks](https://github.com/SRI-International/QC-App-Oriented-Benchmarks) suite. If you want the full benchmark suite (17 benchmarks, notebooks, and examples), clone the repository:

```bash
git clone https://github.com/SRI-International/QC-App-Oriented-Benchmarks.git
cd QC-App-Oriented-Benchmarks
pip install -e .
```

## License

Apache 2.0
