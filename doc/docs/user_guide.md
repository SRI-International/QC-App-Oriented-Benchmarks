# User Guide

This is the complete reference for the **QED-C Application-Oriented Benchmarks** suite. It covers the repository structure, execution options, metrics, supported platforms, and how to use **qedclib** as a standalone library.

## Repository Structure

After installation (`pip install -e .`), the repository provides two Python packages:

```
QC-App-Oriented-Benchmarks/
├── qedclib/              # Execution and metrics library
│   ├── metrics.py        # Performance metrics collection and plotting
│   ├── qcb_mpi.py       # MPI support for multi-GPU execution
│   ├── qiskit/           # Qiskit execution backend
│   ├── cudaq/            # CUDA-Q execution backend
│   ├── cirq/             # Cirq execution backend (limited)
│   ├── braket/           # Braket execution backend (limited)
│   └── ocean/            # Ocean execution backend (MaxCut only)
├── qedcbench/            # The 17 benchmark applications
│   ├── hidden_shift/
│   ├── quantum_fourier_transform/
│   ├── hamlib/
│   └── ...
├── doc/                  # Documentation
└── pyproject.toml        # Package configuration
```

## Architecture

### Three-Function Benchmark API

All benchmarks expose a standard interface:

```python
from qedcbench.hidden_shift import hs_benchmark

# 1. Create circuits
circuits = hs_benchmark.get_circuits(min_qubits=2, max_qubits=8, num_shots=1000, api="qiskit")

# 2. Execute circuits
hs_benchmark.run_circuits(circuits, backend_id="qasm_simulator")

# 3. Plot results
hs_benchmark.plot_results()
```

The convenience function `run(**kwargs)` calls all three and routes arguments automatically:

```python
hs_benchmark.run(min_qubits=2, max_qubits=8, num_shots=1000, api="qiskit", backend_id="qasm_simulator")
```

### Dynamic Module Loading

The benchmarks support multiple quantum computing APIs without hard-coding imports. When you specify `--api qiskit` (or pass `api="qiskit"`), the system dynamically loads the appropriate kernel implementation and execution backend. This is handled internally by `qedclib.initialize()` — you don't need to interact with it directly when running benchmarks normally.

## Running Benchmarks

### From the Command Line

Navigate to the benchmark directory and run the benchmark script:

```bash
cd qedcbench/bernstein_vazirani
python bv_benchmark.py --api qiskit --min_qubits 2 --max_qubits 10 --num_shots 1000
```

Use `--help` or `-h` for a full list of options:

```bash
python bv_benchmark.py -h
```

### Common Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--api` | `-a` | Quantum SDK (`qiskit`, `cudaq`) | `qiskit` |
| `--min_qubits` | | Minimum circuit width | varies |
| `--max_qubits` | | Maximum circuit width | varies |
| `--num_shots` | `-s` | Shots per circuit | 1000 |
| `--num_circuits` | `-c` | Circuits per qubit group | 3 |
| `--backend_id` | `-b` | Backend/simulator name | `qasm_simulator` |
| `--method` | `-m` | Algorithm variant (where applicable) | 1 |
| `--max_batch_size` | `-mbs` | Max circuits to execute at a time (see below) | None |

### Batched Execution

By default, `run()` creates all circuits up front and then executes them as a single batch. For large sweeps this can use excessive memory, particularly on GPU backends.

When `--max_batch_size` (`-mbs`) is set, the benchmark alternates between circuit creation and execution: it creates all circuits for one qubit width at a time (up to `max_circuits`), accumulates widths until adding another would exceed the batch limit, then executes the accumulated batch before continuing. Batch boundaries always fall on qubit-width boundaries — a width's circuits are never split across batches.

```bash
# Execute at most 10 circuits at a time, creating one width at a time
python qft_benchmark.py --min_qubits 2 --max_qubits 20 --max_circuits 3 --max_batch_size 10
```

Within each batch, `max_batch_size` also controls how many circuits are submitted to the backend in a single execution call. For example, if a batch contains 9 circuits and `max_batch_size` is 10, all 9 are submitted at once. If the batch contains 12 (because a single width produced 12 circuits), they are submitted in chunks of 10 and 2.

### From Jupyter Notebooks

Suite notebooks are provided in `qedcbench/` for running multiple benchmarks with shared configuration:

1. Start Jupyter: `jupyter notebook`
2. Open a suite notebook (e.g., `qedcbench/benchmarks-qiskit.ipynb`)
3. Configure the first cell with your backend and execution options
4. Run individual benchmark cells

#### Modularized Notebooks

The modularized notebooks demonstrate the three-part API (get_circuits, run_circuits, plot_results) with explicit control over each step. A base notebook shows all three parts using QED-C defaults, and variant notebooks show how to substitute custom code for individual parts:

| Notebook | Customizes |
|----------|-----------|
| `benchmarks-qiskit-modularized.ipynb` | Baseline — all parts use QED-C defaults |
| `benchmarks-qiskit-modularized-IBM.ipynb` | **Part 2** — execution on IBM hardware |
| `benchmarks-qiskit-modularized-IQM.ipynb` | **Part 2** — execution on IQM hardware |
| `benchmarks-qiskit-modularized-MQT.ipynb` | **Part 1** — circuit generation from MQT Bench |

Each notebook defines an **execution handler** that processes results per circuit (computing fidelity, storing metrics). This is the recommended pattern for custom analysis — see Part 3 in any modularized notebook.

#### Python Script Example

A standalone script is also provided for running benchmarks programmatically:

```bash
python qedcbench/run_benchmark_example.py
```

This runs anywhere after `pip install -e .` and demonstrates the three-step API in minimal code.

### Execution Options

Additional execution options can be passed via `exec_options` dict in notebooks or as keyword arguments:

```python
exec_options = {
    "noise_model": "default",      # Built-in depolarization noise
    "optimization_level": 1,       # Qiskit transpiler optimization (0-3)
}

benchmark.run(**app_args, **exec_options)
```

### Running on Hardware

By default, benchmarks run on local simulators. To run on real hardware:

1. Configure your provider credentials (see [Setup Guides](./setup/README.md))
2. Set the `backend_id` to your hardware target
3. Reduce `max_qubits` and `num_circuits` to avoid excessive costs

```bash
python bv_benchmark.py --backend_id ibm_fez --max_qubits 6 --num_circuits 1 --num_shots 1000
```

**Important:** Hardware execution may incur billing costs. Start with small circuits and few shots.

## Understanding Results

### Metrics

As benchmarks execute, the following metrics are collected and reported for each qubit-width group:

| Metric | Description |
|--------|-------------|
| **Creation Time** | Time spent creating and transpiling circuits on the classical machine |
| **Execution Time** | Time spent running on the quantum backend (excludes queue wait) |
| **Elapsed Time** | Total wall-clock time including queue wait |
| **Hellinger Fidelity** | Similarity between measured and ideal output distributions (0 to 1) |
| **Normalized Fidelity** | Hellinger fidelity normalized to account for maximally noisy output |
| **Circuit Depth** | Number of gate layers in the algorithm |
| **Transpiled Depth** | Gate count after transpilation to a standard gate set (`rx`, `ry`, `rz`, `cx`) |

### Output Files

When benchmarks run, results are saved to the current working directory:

- `__data/DATA-{backend_id}.json` — Raw metrics data in JSON format
- `__images/{backend_id}/` — Generated plot images (JPG/PDF)

### Plotting

After execution, `plot_results()` generates bar charts showing metrics across qubit widths. Plots are displayed inline in Jupyter or saved to `__images/`.

## Using qedclib as a Library

The **qedclib** package provides execution infrastructure and metrics collection independent of any specific benchmark. You can use it for your own quantum programs.

### Basic Usage

```python
import qedclib
from qedclib import metrics

# Initialize qedclib with the API to use (once, at startup)
qedclib.initialize("qiskit")
import execute as ex
ex.set_execution_target("qasm_simulator")

# Initialize metrics tracking
metrics.init_metrics()
```

### Top-Level API

| Function | Description |
|----------|-------------|
| `qedclib.initialize(api, benchmark, kernels)` | Initialize qedclib: set API, load execution backend and kernel modules |
| `qedclib.get_api()` | Get current default SDK |
| `qedclib.set_api(api)` | Set default quantum SDK (called automatically by `initialize`) |
| `qedclib.get_kernel(name, api, benchmark)` | Load and return a kernel module |
| `qedclib.is_leader()` | True if MPI rank 0 or MPI not active |

### Metrics Module

```python
from qedclib import metrics

metrics.init_metrics()
metrics.store_metric(group_id, circuit_id, "fidelity", 0.95)
metrics.aggregate_metrics()
metrics.report_metrics()
metrics.plot_metrics("Benchmark Title")
```

## Supported Platforms

| Platform | Status | Backends |
|----------|--------|----------|
| **Qiskit** | Fully supported | Aer simulator, IBM hardware (via qiskit-ibm-runtime) |
| **CUDA-Q** | Fully supported | Local GPU simulator, NVIDIA cloud (NVQC) |
| **Cirq** | Limited (out of date) | Local simulator |
| **Braket** | Limited (out of date) | AWS simulators and hardware |
| **Ocean** | MaxCut only | D-Wave quantum annealer |

For platform-specific setup instructions, see the [Setup Guides](./setup/README.md).

## Available Benchmarks

The suite includes 17 benchmarks organized by complexity level. All benchmarks are located under `qedcbench/`.

Each benchmark directory contains its own `README.md` with algorithm-specific documentation, including:
- Algorithm description and motivation
- Circuit construction approach
- Expected results and interpretation
- Benchmark-specific options

To see the documentation for a specific benchmark:
```
qedcbench/{benchmark_name}/README.md
```

For example: `qedcbench/hidden_shift/README.md`, `qedcbench/hamlib/README.md`.

## Enabling Compiler Optimizations

The Qiskit benchmarks support compiler optimizations via `exec_options`:

```python
exec_options = {
    "optimization_level": 3,    # Qiskit transpiler optimization level
}
```

Custom transformers (in `qedclib/transformers/`) and third-party tools can also be enabled. See the suite notebook's configuration cells for examples of available optimizations including:
- `qiskit_passmgr` — custom Qiskit pass manager
- `tket_optimiser` — Quantinuum t|ket⟩ optimization
- `trueq_rc` — True-Q randomized compiling

## Upgrading from v1.x

If you have existing code that uses the previous repository structure, here is what changed and how to update:

**Benchmarks have moved** from the top level into `qedcbench/`:
```
# Old:  cd hidden_shift && python hs_benchmark.py
# New:  cd qedcbench/hidden_shift && python hs_benchmark.py
```

**Imports changed** from `_common` to `qedclib`:
```python
# Old:
from _common import metrics
from _common import qcb_mpi as mpi

# New:
from qedclib import metrics
from qedclib import qcb_mpi as mpi
```

The `_common` directory no longer exists. All imports must use `qedclib`.

**Package imports** now use the `qedcbench` prefix:
```python
# Old:
from hidden_shift import hs_benchmark

# New:
from qedcbench.hidden_shift import hs_benchmark
```

**Installation** is now done with `pip install -e .` at the repo root, replacing the old `setup.py`.

For full compatibility with v1.x, use branch [master-260411-v1.2.2](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/tree/master-260411-v1.2.2).

## Developer Notes

This section is for developers working on the repository source code.

### Setup

After cloning the repository, install in editable mode from the repo root:

```bash
pip install -e .
```

This registers both `qedclib` and `qedcbench` as importable packages that point to your local source. Changes to `.py` files take effect immediately without re-installing. You only need to re-run `pip install -e .` if you modify `pyproject.toml`.

### Repository Layout

```
QC-App-Oriented-Benchmarks-master/
├── pyproject.toml              # Package definition (provides qedclib + qedcbench)
├── qedclib/                    # Execution library
│   ├── __init__.py             # Public API
│   ├── api.py                  # Dynamic loader, init, kernel discovery
│   ├── metrics.py              # Metrics collection and plotting
│   ├── batched.py              # Batched execution (batched_run)
│   ├── qiskit/execute.py       # Qiskit execution backend
│   └── cudaq/execute.py        # CUDA-Q execution backend
├── qedcbench/                  # Benchmarks (17 benchmarks + notebooks)
│   ├── __init__.py             # Registers benchmark root with qedclib
│   ├── {benchmark}/            # Each benchmark has its own directory
│   ├── benchmarks-*.ipynb      # Suite and modularized notebooks
│   └── run_benchmark_example.py
├── doc/                        # Documentation (mkdocs)
│   ├── docs/                   # Source .md files (this user guide, etc.)
│   ├── mkdocs.yml              # Navigation config
│   └── site/                   # Generated HTML (python -m mkdocs build)
└── server/app.py               # FastAPI doc server
```

### Documentation

Documentation source is in `doc/docs/`. To build:

```bash
cd doc
python -m mkdocs build
```

To serve locally for preview:

```bash
cd doc
python -m mkdocs serve
```

Or use the FastAPI server (serves the built site with additional features):

```bash
python server/app.py
```

### Key Conventions

- **No `__init__.py` in API subdirectories** (`qedclib/qiskit/`, `qedclib/cudaq/`, etc.) — this avoids namespace conflicts with vendor packages.
- **Benchmark directories keep `__init__.py`** — needed for `from qedcbench.{name} import ...` imports.
- **`import execute as ex`** — after `initialize()` or `get_kernel()`, the execute module is importable by bare name. Never use `from qedclib.qiskit import execute` directly.
- **Absolute paths** — use `os.path.dirname(os.path.abspath(__file__))` in benchmark files, never relative paths.
- **`sys.path.insert` lines** — keep these in benchmark files; they're needed for benchmark-local imports (e.g., `hamlib/_common/`).

<br>
&copy; 2025 Quantum Economic Development Consortium (QED-C). All Rights Reserved.
