# Quick Start

This guide walks you through installing the **QED-C Application-Oriented Benchmarks** and running your first benchmark. By the end, you'll have executed a quantum algorithm benchmark and seen its performance metrics.

> **Tip:** If you just want to run the standard benchmarks without cloning the repository, you can `pip install qedcbench` and then `python -m qedcbench.run_all`. See the [User Guide](user_guide.md) for details on both installation options.

## Prerequisites

- **Python 3.9 or later** (we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for environment management)
- **A quantum computing SDK** — either:
  - [Qiskit](https://qiskit.org/) 2.0+ (includes a local simulator), or
  - [CUDA-Q](https://developer.nvidia.com/cuda-quantum) (requires NVIDIA GPU)

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/SRI-International/QC-App-Oriented-Benchmarks.git
cd QC-App-Oriented-Benchmarks
```

2. **Install the benchmark suite:**

```bash
pip install -e .
```

This single command installs both **qedclib** (the execution library) and **qedcbench** (the benchmarks). No additional setup is needed for simulator-based execution.

3. **Install dependencies** (if not already installed):

```bash
pip install numpy matplotlib
pip install qiskit qiskit-aer qiskit-ibm-runtime   # for Qiskit
```

For CUDA-Q, install `cuda-quantum` instead of the Qiskit packages. See the [Setup Guides](./setup/README.md) for platform-specific instructions.

Some benchmarks have additional dependencies (e.g., `scipy` for Hamiltonian Simulation and HamLib). See individual benchmark directories for details.

## Run Your First Benchmark

The simplest way to run a benchmark is from the command line:

```bash
cd qedcbench/hidden_shift
python hs_benchmark.py --api qiskit --min_qubits 2 --max_qubits 6 --num_shots 1000
```

You should see output like:

```
... execution starting at May 06, 2026 03:22:39 UTC
************
Creating [3] circuits with num_qubits = 2
************
Creating [3] circuits with num_qubits = 4
************
Creating [3] circuits with num_qubits = 6
************
Average Circuit Algorithmic Depth, xi for the 2 qubit group = 8, 0.189
Average Normalized Transpiled Depth, xi, 2q gates for the 2 qubit group = 16, 0.091, 2.0
Average Creation, Elapsed, Execution Time for the 2 qubit group = 0.011, 0.142, 0.002 secs
Average Hellinger, Normalized Fidelity for the 2 qubit group = 0.998, 0.997
...
```

The benchmark creates quantum circuits at each qubit width, executes them on the Qiskit Aer simulator, and reports metrics including circuit depth and fidelity.

## Common Command-Line Options

All benchmarks support these standard arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--api` or `-a` | Quantum SDK to use | `qiskit` |
| `--min_qubits` | Smallest circuit size | varies |
| `--max_qubits` | Largest circuit size | varies |
| `--num_shots` | Measurement shots per circuit | 1000 |
| `--backend_id` | Backend/simulator name | `qasm_simulator` |
| `--num_circuits` or `-c` | Circuits per qubit group | 3 |

Use `--help` or `-h` on any benchmark for a full list of options.

## Run from a Jupyter Notebook

Suite notebooks provide a convenient way to run multiple benchmarks with shared configuration:

1. Start Jupyter: `jupyter notebook`
2. Open `qedcbench/benchmarks-qiskit.ipynb`
3. Run the first cell to configure the backend and execution options
4. Run individual benchmark cells

## Using qedclib as a Library

You can use the execution and metrics infrastructure independently of the benchmarks:

```python
import qedclib

# Initialize qedclib with the API to use (once, at startup)
qedclib.initialize("qiskit")

# Execute and metrics are now available
ex = qedclib.execute
ex.set_execution_target("qasm_simulator")

qedclib.metrics.init_metrics()
```

For more complete examples — including batch execution, metrics collection, hardware backends, and result analysis — see the [qedclib-examples](https://github.com/quantumcomputingdata/qedclib-examples) repository and the [qedclib Guide](./qedclib_guide.md).

## Using Benchmarks as a Package

You can import and run benchmarks programmatically:

```python
from qedcbench.hidden_shift import hs_benchmark

# Create circuits
circuits = hs_benchmark.get_circuits(min_qubits=2, max_qubits=8, num_shots=1000, api="qiskit")

# Execute them
hs_benchmark.run_circuits(circuits, backend_id="qasm_simulator")

# Plot results
hs_benchmark.plot_results()
```

## Next Steps

- See the [User Guide](./user_guide.md) for full documentation on all features, execution options, and metrics
- See the [Setup Guides](./setup/README.md) for platform-specific configuration (IBM hardware, CUDA-Q, etc.)
- See individual benchmark READMEs in `qedcbench/{benchmark}/README.md` for algorithm descriptions

<br>
&copy; 2025 Quantum Economic Development Consortium (QED-C). All Rights Reserved.
