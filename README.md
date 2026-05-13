# Application-Oriented Performance Benchmarks for Quantum Computing

This repository contains a collection of prototypical application- or algorithm-centric benchmark programs designed for characterizing the end-user perception of the performance of current-generation Quantum Computers.

The repository is maintained by members of the Quantum Economic Development Consortium (QED-C) Technical Advisory Committee on Standards and Performance Metrics (Standards TAC).

**Important Note:** The examples maintained in this repository are not intended to be viewed as "performance standards". Rather, they are offered as simple "prototypes", designed to make it as easy as possible for users to execute simple "reference applications" across multiple quantum computing APIs and platforms.

## Quick Start

Install and run a standard set of benchmarks on a local simulator:

```bash
pip install qedcbench
python -m qedcbench.run_all
```

This runs 5 benchmarks (Hidden Shift, Bernstein-Vazirani, QFT, Phase Estimation, Amplitude Estimation) from 2 to 8 qubits and displays a combined volumetric positioning plot at the end.

### Customize the run

```bash
python -m qedcbench.run_all -b ibm_sherbrooke -max 6 -s 100     # IBM hardware
python -m qedcbench.run_all -b ionq -max 6                       # IonQ (requires QISKIT_IONQ_API_TOKEN)
python -m qedcbench.run_all -b iqm -max 6                        # IQM (requires IQM_API_TOKEN)
python -m qedcbench.run_all --list                                # show all available benchmarks
python -m qedcbench.run_all --benchmarks hidden_shift,grovers     # run specific benchmarks
```

Common arguments:

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--api` | `-a` | Quantum SDK (`qiskit`, `cudaq`) | `qiskit` |
| `--backend_id` | `-b` | Backend name | `qasm_simulator` |
| `--min_qubits` | `-min` | Minimum circuit width | `2` |
| `--max_qubits` | `-max` | Maximum circuit width | `8` |
| `--max_circuits` | `-c` | Circuits per qubit group | `3` |
| `--num_shots` | `-s` | Shots per circuit | `100` |

### Run individual benchmarks

After installing, you can also run benchmarks individually:

```bash
cd qedcbench/hidden_shift
python hs_benchmark.py --api qiskit --min_qubits 2 --max_qubits 8
```

## Cloning the Repository

For full access to source code, notebooks, and the ability to modify benchmarks:

```bash
git clone https://github.com/SRI-International/QC-App-Oriented-Benchmarks.git
cd QC-App-Oriented-Benchmarks
pip install -e .
```

This installs both packages in editable mode — changes to `.py` files take effect immediately. The repository includes Jupyter notebooks for interactive exploration:

```bash
cd qedcbench
jupyter notebook benchmarks-qiskit.ipynb
```

> **Note:** If you have existing code that depends on the v1.x repository structure, use branch **[master-260411-v1.2.2](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/tree/master-260411-v1.2.2)** for compatibility. See the [User Guide](https://sri-international.github.io/QC-App-Oriented-Benchmarks/user_guide/#upgrading-from-v1x) for migration details.

## Documentation

**[Full Documentation Site](https://sri-international.github.io/QC-App-Oriented-Benchmarks/)** — User guide, benchmark descriptions, qedclib API reference, and setup guides.

| Document | Description |
|----------|-------------|
| [User Guide](https://sri-international.github.io/QC-App-Oriented-Benchmarks/user_guide/) | Complete reference for running benchmarks |
| [Benchmarks](https://sri-international.github.io/QC-App-Oriented-Benchmarks/benchmarks/) | All 17 benchmarks with algorithm descriptions |
| [qedclib Guide](https://sri-international.github.io/QC-App-Oriented-Benchmarks/qedclib_guide/) | Execution engine API, metrics, and backend configuration |
| [Quick Start](https://sri-international.github.io/QC-App-Oriented-Benchmarks/quick_start/) | First-time setup walkthrough |
| [Release Notes](https://sri-international.github.io/QC-App-Oriented-Benchmarks/release_notes/) | Version history and changes |
| [PAL](https://sri-international.github.io/QC-App-Oriented-Benchmarks/known_issues/) | Problems, Anomalies, and Limitations |
| [About](https://sri-international.github.io/QC-App-Oriented-Benchmarks/about/) | Project background, structure, and credits |
| [Setup Guides](https://sri-international.github.io/QC-App-Oriented-Benchmarks/setup/README/) | Platform-specific installation (Qiskit, CUDA-Q, etc.) |

**Standalone execution engine:** `pip install qedclib` — use the execution and metrics library in your own projects without the benchmarks. See [qedclib on PyPI](https://pypi.org/project/qedclib/).

## Benchmark Complexity Levels

```
Level 1: Deutsch-Jozsa, Bernstein-Vazirani, Hidden Shift
Level 2: Quantum Fourier Transform, Grover's Search
Level 3: Phase Estimation, Amplitude Estimation, HHL Linear Solver
Level 4: Monte Carlo, Hamiltonian Simulation, HamLib, VQE, Shor's Algorithm
Level 5: MaxCut, Hydrogen Lattice, Image Recognition
```

## Publications

&nbsp;&nbsp;&nbsp;&nbsp;[Application-Oriented Performance Benchmarks for Quantum Computing](https://arxiv.org/abs/2110.03137) (Oct 2021)

&nbsp;&nbsp;&nbsp;&nbsp;[Optimization Applications as Quantum Performance Benchmarks](https://arxiv.org/abs/2302.02278) (Feb 2023)

&nbsp;&nbsp;&nbsp;&nbsp;[Quantum Algorithm Exploration using Application-Oriented Performance Benchmarks](https://arxiv.org/abs/2402.08985) (Feb 2024)

&nbsp;&nbsp;&nbsp;&nbsp;[A Comprehensive Cross-Model Framework for Benchmarking the Performance of Quantum Hamiltonian Simulations](https://arxiv.org/abs/2409.06919) (Sep 2024)

&nbsp;&nbsp;&nbsp;&nbsp;[A Practical Framework for Assessing the Performance of Observable Estimation in Quantum Simulation](https://arxiv.org/abs/2504.09813) (Apr 2025)

&nbsp;&nbsp;&nbsp;&nbsp;[Platform-Agnostic Modular Architecture for Quantum Benchmarking](https://arxiv.org/abs/2510.08469) (2025)

## Implementation Status

![Application-Oriented Benchmarks - Implementation Status](./doc/docs/images/proto_benchmarks_status.png)

<br>
&copy; 2025 Quantum Economic Development Consortium (QED-C). All Rights Reserved.
