# qedclib

A quantum program execution engine with built-in performance monitoring. Execute quantum circuits at scale across multiple backends with automatic metrics collection, parallel GPU execution, and detailed per-circuit timing.

## Install

```bash
pip install qedclib
```

## Quick Example

```python
import qedclib

# Select the quantum computing API
qedclib.set_api("qiskit")

# Initialize and get the execution module
qedclib.initialize()
import execute as ex

# Execute circuits and collect metrics
circuits = [{"circuit": qc, "num_qubits": n, "secret_int": s} for qc, n, s in my_circuits]
ex.submit_circuits(circuits, num_shots=1000, backend_id="qasm_simulator")

# Plot collected metrics
qedclib.metrics.plot_metrics("My Benchmark")
```

## Features

- **Multi-backend support** — Qiskit (Aer, IBM hardware, IonQ, IQM) and CUDA-Q (simulator, multi-GPU, NVIDIA Quantum Cloud)
- **Automatic metrics** — execution time, fidelity, circuit depth, gate counts, and volumetric benchmarking plots
- **Per-circuit timing** — accurate individual circuit timing on all backends
- **Batched execution** — memory-efficient execution for large parameter sweeps on GPU backends
- **Multi-GPU parallel execution** — distribute circuits across GPUs with MPI
- **Robust job management** — automatic retry, status polling, and result validation for hardware backends

## Documentation

Full documentation is available at [sri-international.github.io/QC-App-Oriented-Benchmarks](https://sri-international.github.io/QC-App-Oriented-Benchmarks/), including the [User Guide](https://sri-international.github.io/QC-App-Oriented-Benchmarks/user_guide/) with standalone library usage examples.

## Part of the QED-C Benchmarks

qedclib is the execution engine behind the [QED-C Application-Oriented Benchmarks](https://github.com/SRI-International/QC-App-Oriented-Benchmarks) suite. If you want the full benchmark suite (17 benchmarks, notebooks, and examples), clone the repository instead:

```bash
git clone https://github.com/SRI-International/QC-App-Oriented-Benchmarks.git
cd QC-App-Oriented-Benchmarks
pip install -e .
```

## License

Apache 2.0
