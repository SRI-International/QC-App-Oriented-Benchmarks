# Benchmark Reference

Each benchmark application includes its own documentation describing the algorithm, circuit construction, and expected results. This page provides quick links to all benchmark documentation.

## Level 1: Introductory

| Benchmark | Description |
|-----------|-------------|
| [Deutsch-Jozsa](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/deutsch_jozsa/README.md) | Determines if a function is constant or balanced |
| [Bernstein-Vazirani](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/bernstein_vazirani/README.md) | Finds a hidden bit string encoded in a function |
| [Hidden Shift](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/hidden_shift/README.md) | Finds a hidden shift between two functions |

## Level 2: Intermediate

| Benchmark | Description |
|-----------|-------------|
| [Quantum Fourier Transform](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/quantum_fourier_transform/README.md) | Quantum analog of the discrete Fourier transform |
| [Grover's Search](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/grovers/README.md) | Unstructured database search with quadratic speedup |

## Level 3: Advanced

| Benchmark | Description |
|-----------|-------------|
| [Phase Estimation](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/phase_estimation/README.md) | Estimates eigenvalues of unitary operators |
| [Amplitude Estimation](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/amplitude_estimation/README.md) | Estimates probability amplitudes |
| [HHL Linear Solver](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/hhl/README.md) | Solves linear systems of equations |

## Level 4: Application

| Benchmark | Description |
|-----------|-------------|
| [Monte Carlo](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/monte_carlo/README.md) | Quantum Monte Carlo sampling |
| [Hamiltonian Simulation](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/hamiltonian_simulation/README.md) | Trotterized quantum time evolution |
| [HamLib Simulation](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/hamlib/README.md) | Hamiltonian simulation using the HamLib library |
| [VQE](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/vqe/README.md) | Variational Quantum Eigensolver for ground state energy |
| [Shor's Algorithm](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/shors/README.md) | Integer factoring via quantum order finding |

## Level 5: Hybrid Variational

| Benchmark | Description |
|-----------|-------------|
| [MaxCut](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/maxcut/README.md) | Graph optimization using QAOA |
| [Hydrogen Lattice](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/hydrogen_lattice/README.md) | Molecular ground state energy estimation |
| [Image Recognition](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/image_recognition/README.md) | Quantum-enhanced image classification |

## Other

| Benchmark | Description |
|-----------|-------------|
| [Quantum Reinforcement Learning](https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/qedcbench/quantum_reinforcement_learning/README.md) | Reinforcement learning with parameterized quantum circuits |

## Programming Examples

Ready-to-use examples are provided in `qedcbench/` showing how to run benchmarks from code:

| Example | Description |
|---------|-------------|
| `run_benchmark_example.py` | Minimal Python script — runs anywhere after `pip install -e .` |
| `benchmarks-qiskit-modularized.ipynb` | Notebook: baseline three-part flow (get_circuits → run_circuits → plot_results) |
| `benchmarks-qiskit-modularized-IBM.ipynb` | Notebook: custom execution on IBM hardware |
| `benchmarks-qiskit-modularized-IQM.ipynb` | Notebook: custom execution on IQM hardware |
| `benchmarks-qiskit-modularized-MQT.ipynb` | Notebook: custom circuit generation from MQT Bench |

The modularized notebooks illustrate how to substitute custom code for any of the three parts. See the [User Guide](./user_guide.md#from-jupyter-notebooks) for details.
