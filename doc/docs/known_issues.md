# PAL — Problems, Anomalies, and Limitations

As with most complex and evolving software, the **QED-C Application-Oriented Benchmarks** have several issues you may want to be aware of. This section describes known **P**roblems, **A**nomalies, and **L**imitations. Being aware of these in advance can help avoid frustration.

## Limitations

These are fundamental limitations in the current implementation. Some may be relaxed in future versions.

#### Platform Support

- **Cirq and Braket support is out of date.** The benchmark implementations for Google Cirq and Amazon Braket have not been updated for the v2.0 architecture. They may still work for some benchmarks but are not actively tested.

- **Ocean support is limited to MaxCut only.** The D-Wave Ocean integration supports only the MaxCut benchmark.

#### Execution

- **CUDA-Q mgpu mode: OOM at 18+ qubits with multiple circuits.** On Perlmutter (post-May 2026 maintenance), even sequential `cudaq.sample()` calls don't fully release state vector memory between circuits when using mgpu with 8 GPUs. This appears to be a cudaq/cusvsim regression. Workaround: use pipelined execution (when available) or reduce circuit count.

- **Resumable job persistence is not implemented.** If a hardware job completes after the Python process exits, results cannot currently be recovered and matched to expected distributions for fidelity computation. This feature is deferred pending a group discussion on design.

- **Dynamic kernel loader assumes repo structure.** The dynamic loading system (`qedclib._init_engine`) expects benchmark kernels to be located at `qedcbench/{benchmark}/{api}/`. Benchmarks cannot currently be run from arbitrary locations outside the repository without modification.

#### Timing

- **IBM hardware returns batch-level timing only.** Per-circuit execution time is not available from IBM when circuits are submitted as a batch. All circuits in a batch receive equal `exec_time` (total divided by N). Per-width timing requires pipelined execution (Task 1, not yet implemented).

- **IonQ and IQM return zero execution time.** Both backends report `time_taken=0.0` in results. Timing falls back to `elapsed_time / N`, which includes queue wait time.

## Anomalies

These are not bugs per se, but behaviors worth knowing about.

#### Module Loading

- **`import execute` vs `from qedclib.qiskit import execute` are different module instances.** The dynamic loader injects `execute` into `sys.modules` after `initialize()` is called. Importing directly from `qedclib.qiskit` bypasses this and creates a separate module object. Settings changed on one (e.g., `execute.verbose = True`) are not reflected in the other. Always use `import execute as ex` after initialization.

#### Benchmark Behavior

- **Verbose flag gets reset by benchmarks.** Each benchmark's `run_circuits()` sets `ex.verbose = verbose` using its module-level `verbose` variable. To enable verbose output, set `benchmark.verbose = True` before calling `run()`, not by setting `execute.verbose` directly.

- **Random seeds reset per qubit group.** Most benchmarks call `np.random.seed(0)` at the start of each qubit-width group. This ensures reproducibility but means the same random circuits are generated regardless of which groups are selected.

## Known Problems

These are specific bugs or issues that may be fixed in future releases.

#### Import Patterns

- **Some old-style benchmark files use wrong import pattern.** Files in `{benchmark}/qiskit/` and `{benchmark}/cirq/` subdirectories (pre-v2 implementations) may use `from qedclib.qiskit import execute` directly. This bypasses the dynamic loader and creates a separate module instance. These files are functional but should be migrated to the standard `import execute` pattern after `initialize()`.

- **QRL module identity mismatch.** The quantum reinforcement learning benchmark imports metrics in two different ways, which can cause the metrics module to appear as two separate instances.

#### Notebooks

- **HHL notebook has hardcoded qubit limits.** The HHL benchmark notebook restricts qubit counts in a way that may not reflect the actual capability of the algorithm implementation.

<br>
&copy; 2025 Quantum Economic Development Consortium (QED-C). All Rights Reserved.
