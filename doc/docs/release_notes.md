# Release Notes

The **QED-C Application-Oriented Benchmarks** suite is continually evolving, with issues fixed and enhancements made to add features.
This section presents a brief record of relevant changes made to each version that has been released. Latest version is at the top.

## Version 2.0 Release Notes 

This version constitutes a major refactor of the entire benchmark repository. For the previous repository structure, use branch **master-260411-v1.2.2**.

### Release 2.0.5 - 27 May 2026

- **Bug fix: backend switching** — Fixed stale IBM session/sampler state persisting when switching between backends (e.g., IBM → IonQ), causing gate validation errors. State is now fully reset on each call to `set_execution_target`.
- **Bug fix: IBM API credentials** — The execution engine now reads `IBM_API_TOKEN` from the environment alongside `IBM_INSTANCE`, instead of always using the saved token. Warns when instance is set without a matching token, and provides a clear error message on authentication failure (403).
- **Bug fix: cloud simulator batching** — `max_batch_size=1` (for cancel responsiveness) is now applied only to local simulators. Cloud backends like `ionq_simulator` batch all circuits together as expected.
- **Hardened CUDA-Q execution** — Added error handling and `KeyboardInterrupt` propagation around `cudaq.sample()` in all execution paths. Improved signal handling in `run_all.py` for more reliable Ctrl-C termination during GPU execution.
- **Server UI: cancel status** — Benchmarks that have not yet started when the user cancels a run now show "CANCELLED" instead of remaining as "pending".
- **Fixed HamLib notebook imports** for v2 repository structure.
- **Documentation**: clarified Qiskit/CUDA-Q install steps for pip users.
- Updated PAL (known issues) documentation for accuracy.

### Release 2.0.4 - 19 May 2026

- **Benchmark Runner Web UI (experimental)**: Added a preliminary browser-based interface for executing benchmarks. Select benchmarks, configure run parameters (API, backend, qubits, shots), and launch execution from the browser. Live progress updates, a console output panel streaming benchmark output in real time, and volumetric positioning plots displayed after completion. Includes backend presets dropdown, data file management, and run cancellation support. To try it: `cd server && uvicorn app:app --port 8088` (requires `pip install fastapi uvicorn`).
- Fixed IBM job tags exceeding the 8-tag limit on large circuit batches.
- Added cancellation support in qiskit and cudaq execute modules (`request_cancel()`).

### Release 2.0.3 - 18 May 2026

- **Documentation improvements**: User Guide now presents both installation options (pip install vs clone) up front. Quick Start mentions pip install as an alternative. Dependency requirements (numpy, matplotlib, Qiskit/CUDA-Q packages) documented in both guides.
- Added `qedcbench/README.md` with benchmark listing and usage examples for repo browsers.
- Updated `qedclib/README.md` example to use the `qedclib.execute` import pattern.
- Created starter examples in [qedclib-examples](https://github.com/quantumcomputingdata/qedclib-examples) and [qedcbench-examples](https://github.com/quantumcomputingdata/qedcbench-examples) repos for pip install users.

### Release 2.0.2 - 12 May 2026

- **Metrics save separated from plotting**: `plot_metrics()` no longer saves data as a side effect. Each benchmark's `run_circuits()` now calls `metrics.save_app_metrics(benchmark_name, method=method)` explicitly after `finalize_all_groups()`. This makes the save visible and controllable.
- **Simplified data file keys**: Data file keys changed from `"Benchmark Results - Quantum Fourier Transform (1) - Qiskit"` to `"Quantum Fourier Transform (1)"`. The API name is now stored as a separate `"api"` field in the data record. **Breaking change**: existing `__data/DATA-*.json` files must be regenerated.
- **Simplified imports**: After `initialize()` or `get_kernel()`, the execute module is available as `qedclib.execute`. Use `ex = qedclib.execute` instead of `import execute as ex`. Metrics are always available as `qedclib.metrics`. All standard benchmarks updated to use this pattern.
- Added CLI benchmark runner (`python -m qedcbench.run_all`) for running a standard set of benchmarks from the command line with a combined volumetric plot.
- Created dedicated qedclib Guide documentation page with API reference, execution paths, and metrics workflow.
- Updated README to support both `pip install` and repository cloning workflows.
- Published `qedcbench` package to PyPI (`pip install qedcbench`).

### Release 2.0.1 - 9 May 2026

- General performance improvements to Qiskit execution logic
- Added `metrics.get_circuit_metrics()` and `metrics.get_group_metrics()` accessor functions for clean retrieval of collected metrics data.
- Added test script `05_submit_with_metrics.py` demonstrating the submit/finalize/get metrics flow.
- Execution engine cleanup: threaded `job.result()`, timing function extraction, sampler timing fix.

### Release 2.0.0 - 5 May 2026

#### Repository Restructure: qedclib + qedcbench

The repository has been split into two packages, both installable with a single `pip install -e .`:
- **qedclib** — the execution and metrics library (formerly `_common/`). Import as `import qedclib`.
- **qedcbench** — the 17 benchmark applications (moved into `qedcbench/` directory). Import as `from qedcbench.hidden_shift import hs_benchmark`.

Code that previously used `from _common import metrics` should change to `from qedclib import metrics`.

#### Three-Function Benchmark API

All 16 benchmarks now expose a standard three-function interface for independent circuit creation, execution, and plotting:
```python
all_qcs, metrics = benchmark.get_circuits(min_qubits=2, max_qubits=8, ...)
benchmark.run_circuits(all_qcs, backend_id="qasm_simulator", ...)
benchmark.plot_results(...)
```
The convenience `run(**kwargs)` function calls all three and routes arguments automatically. Variational benchmarks (maxcut, hydrogen_lattice, image_recognition, VQE, QRL) support `get_circuits` for method 1 (fidelity); method 2 optimizer loops remain in a dedicated `run_method2()`.

#### Array-Based Execution Path

The execution engine (`qedclib/qiskit/execute.py` and `qedclib/cudaq/execute.py`) now uses a batch-oriented API:
- `execute_circuits(circuits, num_shots)` — execute an array of circuits
- `process_circuit_results(circuits_info, results)` — map results back to metrics
- `submit_circuits(circuits_dict, num_shots)` — convenience wrapper for the above
- `compute_all_circuit_metrics(all_qcs)` — batch circuit depth/gate metrics

The older single-circuit pipelined execution functions (`submit_circuit`, `throttle_execution`, `finalize_execution`) remain for backward compatibility but are no longer used by any benchmark.

#### Auto-Warmup (CUDA-Q)

The first call to `execute_circuits()` in CUDA-Q now automatically runs a tiny 1-qubit circuit to prime the JIT compiler before executing the user's circuits. This eliminates the JIT compilation overhead from the first benchmark circuit's timing, without requiring any flags or special handling. The per-benchmark `--warmup` CLI option has been removed. To disable auto-warmup, set `execute.auto_warmup = False`.

#### Batched Execution

When `max_batch_size` (`-mbs`) is set, `run()` alternates between circuit creation and execution rather than creating all circuits up front. Circuits are created one qubit width at a time (up to `max_circuits` per width) and accumulated until adding another width would exceed the batch limit. The accumulated batch is then executed before continuing to the next set of widths. Batch boundaries always align with qubit-width boundaries — a width's circuits are never split across batches.

This reduces peak memory usage for large sweeps, particularly on GPU backends where many simultaneous state vectors can cause out-of-memory errors.

```bash
python qft_benchmark.py --min_qubits 2 --max_qubits 20 -c 3 --max_batch_size 10
```

#### Per-Circuit Execution Timing

Each circuit now records its own execution time individually:

- **CUDA-Q**: Times each `cudaq.sample()` call directly, rather than dividing the batch total evenly.
- **Qiskit (Aer simulator)**: Extracts per-experiment `time_taken` from results.
- **Qiskit (IBM hardware)**: Divides `execution_spans` duration evenly across circuits in the batch. Per-width timing requires batched execution (`-mbs`).
- **Qiskit (IonQ, IQM)**: Falls back to `elapsed_time / N` when backends report `time_taken=0`.

Elapsed-time overhead (queue wait, network) is distributed proportionally by execution time across circuits in a batch.

#### Job Status & Error Handling

The execution engine now includes robust job management for hardware backends:

- **Retry logic**: `job.result()` is retried up to 40 times at 15-second intervals, with early detection of cancelled or errored jobs.
- **Status polling**: Job status is checked before blocking on results, with comfort dots printed while queued or running (in verbose mode).
- **Result validation**: Warns if the number of results returned doesn't match the number of circuits submitted, and pads missing results. Warns if actual shot count differs from requested.
- **Empty counts guard**: Circuits that return empty measurement counts (e.g., from cancelled jobs) are skipped during result processing to avoid division-by-zero errors.

These improvements apply to both `qedclib/qiskit/execute.py` and `qedclib/cudaq/execute.py`.

#### Noise Model via exec_options

Both Qiskit and CUDA-Q now support setting the noise model through `exec_options`, eliminating the need to import the execute module directly:
```python
exec_options = {"noise_model": "default"}   # built-in depolarization model
exec_options = {"noise_model": None}        # no noise (noiseless simulation)
exec_options = {"noise_model": my_model}    # custom noise model object
```

#### Simplified Init API

New helper functions in `qedclib` reduce boilerplate:
- `qedc_set_api("qiskit")` — set the default API once
- `qedc_get_kernel("kernel_name")` — one-line kernel loading with auto-detection
- `qedc_is_leader()` — MPI leader check abstraction

#### Updated Benchmark Notebooks

`benchmarks-qiskit.ipynb` and `benchmarks-cudaq.ipynb` now use grouped `app_args` and `exec_args` dictionaries, matching the pattern used in the HamLib observables notebooks. Each benchmark cell is two lines:
```python
from quantum_fourier_transform import qft_benchmark
qft_benchmark.run(method=1, **app_args, **exec_args)
```

#### Modularized Notebooks

A new set of modularized notebooks splits benchmark execution into three independent parts (Get Circuits, Run Circuits, Plot Results), allowing any part to be customized while reusing the others:

- **benchmarks-qiskit-modularized.ipynb** — baseline, all parts use QED-C defaults
- **benchmarks-qiskit-modularized-IBM.ipynb** — custom Part 2 for execution on IBM hardware
- **benchmarks-qiskit-modularized-IQM.ipynb** — custom Part 2 for execution on IQM hardware
- **benchmarks-qiskit-modularized-MQT.ipynb** — custom Part 1 for circuit generation from MQT Bench

Results analysis uses an `execution_handler` callback pattern via `ex.init_execution(handler)` and `ex.process_circuit_results()`, replacing manual iteration over result indices.

#### Braket & Cirq Notebooks (Unmaintained)

The Braket and Cirq suite notebooks have been renamed to `zz-benchmarks-braket.ipynb` and `zz-benchmarks-cirq.ipynb` and marked as unmaintained. These APIs are not actively tested; community contributions are welcome.

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
