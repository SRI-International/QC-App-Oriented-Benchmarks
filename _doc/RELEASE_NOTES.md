# Release Notes

The **QED-C Application-Oriented Benchmarks** suite is continually evolving, with issues fixed and enhancements made to add features.
This section presents a brief record of relevant changes made to each version that has been released. Latest version is at the top.

## Version 2.0 Release Notes 

This version constitutes a major refactor of the entire benchmark repository.  For backwards compatibility with externally developed work that uses the repo, use version 1.2.1, the last satable release of the earlier benchmark repository structure.

### Release 2.0.0 - 15 May 2026

#### Three-Function Benchmark API

All 16 benchmarks now expose a standard three-function interface for independent circuit creation, execution, and plotting:
```python
all_qcs, metrics = benchmark.get_circuits(min_qubits=2, max_qubits=8, ...)
benchmark.run_circuits(all_qcs, backend_id="qasm_simulator", ...)
benchmark.plot_results(...)
```
The convenience `run(**kwargs)` function calls all three and routes arguments automatically. Variational benchmarks (maxcut, hydrogen_lattice, image_recognition, VQE, QRL) support `get_circuits` for method 1 (fidelity); method 2 optimizer loops remain in a dedicated `run_method2()`.

#### Array-Based Execution Path

The execution engine (`_common/qiskit/execute.py` and `_common/cudaq/execute.py`) now uses a batch-oriented API:
- `execute_circuits(circuits, num_shots)` — execute an array of circuits
- `process_circuit_results(circuits_info, results)` — map results back to metrics
- `submit_circuits(circuits_dict, num_shots)` — convenience wrapper for the above
- `compute_all_circuit_metrics(all_qcs)` — batch circuit depth/gate metrics

The older single-circuit pipelined execution functions (`submit_circuit`, `throttle_execution`, `finalize_execution`) remain for backward compatibility but are no longer used by any benchmark.

#### Per-Circuit Execution Timing (CUDA-Q)

Each circuit executed via `execute_circuits()` in CUDA-Q now records its own execution time individually, rather than dividing the batch total evenly. This provides accurate per-circuit timing without requiring `max_batch_size=1`.

#### Noise Model via exec_options

Both Qiskit and CUDA-Q now support setting the noise model through `exec_options`, eliminating the need to import the execute module directly:
```python
exec_options = {"noise_model": "default"}   # built-in depolarization model
exec_options = {"noise_model": None}        # no noise (noiseless simulation)
exec_options = {"noise_model": my_model}    # custom noise model object
```

#### Simplified Init API

New helper functions in `_common/qedc_init.py` reduce boilerplate:
- `qedc_set_api("qiskit")` — set the default API once
- `qedc_get_kernel("kernel_name")` — one-line kernel loading with auto-detection
- `qedc_is_leader()` — MPI leader check abstraction

#### Updated Benchmark Notebooks

`benchmarks-qiskit.ipynb` and `benchmarks-cudaq.ipynb` now use grouped `app_args` and `exec_args` dictionaries, matching the pattern used in the HamLib observables notebooks. Each benchmark cell is two lines:
```python
from quantum_fourier_transform import qft_benchmark
qft_benchmark.run(method=1, **app_args, **exec_args)
```

#### Streamlined Notebook Configuration

The `benchmarks-qiskit.ipynb` setup cell now presents backend configurations (simulator, IBM Cloud, IBM Platform, IonQ, BlueQubit) as self-contained blocks — uncomment the one you need. Noise model options are consolidated inline. The separate custom options cell has been removed.

#### Simplified Thin Wrappers

Benchmark wrappers (e.g. `maxcut/maxcut_benchmark.py`) now use `run(**kwargs)` forwarding instead of manually listing all parameters. The `api` argument selects the implementation; `backend_id` defaults to `"qasm_simulator"` if not provided.

#### Qiskit 2.x Estimator Migration

The `hydrogen_lattice` and `hamlib` benchmarks have been updated for the Qiskit 2.x Estimator API:
- `Estimator` → `StatevectorEstimator` / `BackendEstimatorV2`
- New call format: `estimator.run([(qc, operator)])`
- Result extraction: `result[0].data.evs.item()`

#### Windows Compatibility

Fixed `UnicodeEncodeError` on Windows terminals (cp1252) by replacing the Greek xi character (ξ) with ASCII "xi" in metrics output.

#### Bug Fixes

- Fixed `get_appname_from_title()` crash when plot title missing expected delimiter
- Fixed `run_objective_function()` in hydrogen_lattice to work with newer scipy (COBYLA minimum function evaluation requirement)
- Fixed test imports for restructured package layout (hydrogen_lattice, hamiltonian_simulation)
- Removed duplicate benchmark files (`grovers/qiskit/grovers_benchmark.py`, `amplitude_estimation/qiskit/ae_benchmark.py`)

---

## Version 1.2 Release Notes 

This is the last stable version of the earlier benchmark repository structure. For convenience in running externally developed work with no changes, a branch names master-260414 contains source code for this version.

### Release 1.2.1 - 14 April 2026

- **Parallel Execution Logic modifed to use GPU Cluster Size** The pair of options that was implemented earlier to support parallel execution of circuits in hamlib has been modifed. The options **--parallel_mode** and **--num_gpus** have been removed and replaced with the **--gpus_per_cluster (-gpc)** option, which indicates the number of GPUs that cluster together to execute a single circuit. E.g., if you have 16 GPUs and set -gpc to 1, then you could execute 16 circuits in parallel. with 1 GPU used for each execution. The default is to use ALL GPUs in a cluster if MPI is enabled. This is the option which provides more qubits by distributing the state vector across all of the GPUs. The "hybrid" mode in which you might set -gpc to a number other than 1 is not yet implemented.
    
- **Refactor of API-specific Execution Logic** This set of changes consolidated and normalized the result object returned from execution of a quantum circuit across all APIs.  The ExecutionResult object provides the get_counts() method which returns a dictionary of measurment results.  The counts dict can be populated in serverl methods unique to the API.

### Release 1.2.0 - 11 April 2026

- **Added Support for Parallel Circuit Execution** The Hamlib benchmark on CUDA-Q now supports an execution mode in which a subset of circuits generated for the selected Hamiltonian simulation are executed in parallel in order to reduce the total execution time.

### Release 1.1.0 - 25 January 2026

- **Completed Refactoring of API-specific kernel Files** All benchmarks are now structured so that API-specific code, e.g for Qiskit or CUDA-Q, is contained in a correspondingly named sub-directory. The primary benchmark file resides at the benchmark directory level and is API-agnostic.  The --app (-a) command line argument is used to specify the name of the API, cudaq or qiskit, from which the appropriate kernel code will be dynamically imported.

### Release 1.0.0 - Prior to 12 November 2025

- **Multiple Years of Development and Testing** Since 2019, QED-C has been working to develop, test, and polish the complete set of benchmark programs.

<br>
&nbsp;&copy;&nbsp;2025 Quantum Economic Development Consortium (QED-C). All Rights Reserved.
