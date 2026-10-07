# qedcbench — Benchmark Modules

This directory contains the 17 application-oriented quantum computing benchmark modules. Each benchmark implements a three-function API: `get_circuits()`, `run_circuits()`, and `plot_results()`.

## Running Benchmarks

**Run the standard suite:**

```bash
python -m qedcbench.run_all                          # 5 defaults, simulator
python -m qedcbench.run_all -b ibm_sherbrooke -max 6  # IBM hardware
python -m qedcbench.run_all --list                    # show all benchmarks
```

**Run an individual benchmark:**

```bash
cd qedcbench/hidden_shift
python hs_benchmark.py --min_qubits 2 --max_qubits 8
```

**Import programmatically:**

```python
from qedcbench.hidden_shift import hs_benchmark
hs_benchmark.run(min_qubits=2, max_qubits=6, num_shots=100)
```

## Available Benchmarks

| Benchmark | Directory | Level |
|-----------|-----------|-------|
| Deutsch-Jozsa | `deutsch_jozsa/` | 1 |
| Bernstein-Vazirani | `bernstein_vazirani/` | 1 |
| Hidden Shift | `hidden_shift/` | 1 |
| Quantum Fourier Transform | `quantum_fourier_transform/` | 2 |
| Grover's Search | `grovers/` | 2 |
| Phase Estimation | `phase_estimation/` | 3 |
| Amplitude Estimation | `amplitude_estimation/` | 3 |
| HHL Linear Solver | `hhl/` | 3 |
| Monte Carlo | `monte_carlo/` | 4 |
| Hamiltonian Simulation | `hamiltonian_simulation/` | 4 |
| HamLib | `hamlib/` | 4 |
| VQE | `vqe/` | 4 |
| Shor's Algorithm | `shors/` | 4 |
| MaxCut | `maxcut/` | 5 |
| Hydrogen Lattice | `hydrogen_lattice/` | 5 |
| Image Recognition | `image_recognition/` | 5 |
| Quantum Reinforcement Learning | `quantum_reinforcement_learning/` | 5 |

## Key Files

- `run_all.py` — CLI benchmark runner (`python -m qedcbench.run_all`)
- `benchmarks-qiskit.ipynb` — Jupyter notebook for interactive benchmarking
- `run_benchmark_example.py` — Example script showing programmatic usage

## More Information

- [User Guide](https://sri-international.github.io/QC-App-Oriented-Benchmarks/user_guide/) — Complete reference for running and configuring benchmarks
- [qedcbench-examples](https://github.com/quantumcomputingdata/qedcbench-examples) — Standalone examples for use after `pip install qedcbench`
- [Top-level README](../README.md) — Installation options and overview
