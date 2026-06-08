# PAL — Problems, Anomalies, and Limitations

As with most complex and evolving software, the **QED-C Application-Oriented Benchmarks** have several issues you may want to be aware of. This section describes known **P**roblems, **A**nomalies, and **L**imitations. Being aware of these in advance can help avoid frustration.

## Limitations

These are fundamental limitations in the current implementation. Some may be relaxed in future versions.

#### Platform Support

- **Cirq and Braket support is out of date.** The benchmark implementations for Google Cirq and Amazon Braket have not been updated for the v2.0 architecture. They may still work for some benchmarks but are not actively tested.

- **Ocean support has not been tested with the v2.0 framework.** The D-Wave Ocean integration (MaxCut benchmark only) was functional in v1.2 but has not been updated or tested with the new repository structure. This will be addressed in a near-term release.

#### Execution

- **CUDA-Q multi-GPU (mgpu) fails on certain circuit structures.** When running Hamiltonian observable estimation with distributed statevector execution on real multi-GPU hardware (e.g., 4x A100 on Perlmutter), circuits containing multi-qubit Pauli basis rotation gates (YY, XXX, YYXXX, etc.) fail with `requested size is too big` or gate-grouping errors in `cusvsim`. Single-qubit X/Z terms and simpler Hamiltonians like TFIM work fine. This is a CUDA-Q 0.13.0 runtime issue, not a qedclib bug. **Workaround**: use `--parallel` (`-pm`) to run in single-GPU-per-rank mode instead of distributed statevector mode. See [CUDA-Q Multi-GPU Issues](issues_cudaq_mgpu.md) for detailed analysis and reproduction steps.

- **CUDA-Q execution may fail with multiple circuits per group.** When using `-c 2` or higher (multiple circuit repetitions per qubit width), execution on CUDA-Q can fail once the total number of circuits across all groups exceeds some threshold. The root cause is under investigation. Workaround: use `-c 1` or reduce the qubit range.

- **Resumable job persistence is not implemented.** If a hardware job completes after the Python process exits, results cannot currently be recovered and matched to expected distributions for fidelity computation. This feature is deferred pending a group discussion on design.

- **Dynamic kernel loader assumes repo structure.** The dynamic loading system (`qedclib._init_engine`) expects benchmark kernels to be located at `qedcbench/{benchmark}/{api}/`. Benchmarks cannot currently be run from arbitrary locations outside the repository without modification.

#### Timing

- **Elapsed time on hardware backends may include queue wait time.** When executing on any hardware backend (IBM, IonQ, IQM, etc.), the reported elapsed time includes time spent waiting in the provider's job queue, not just circuit execution time. The auto-warmup option may reduce queue wait for subsequent batches by keeping the session active, but this has not yet been fully tested.

## Anomalies

These are not bugs per se, but behaviors worth knowing about.

#### Module Loading

- **Always use `import execute as ex` after calling `initialize()`.** Do not import execute directly from `qedclib.qiskit` (e.g., `from qedclib.qiskit import execute`) — this bypasses the dynamic loader and creates a separate module instance, so settings and state will not be shared correctly.

#### Benchmark Behavior

- **Verbose flag gets reset by benchmarks.** Each benchmark's `run_circuits()` sets `ex.verbose = verbose` using its module-level `verbose` variable. To enable verbose output, set `benchmark.verbose = True` before calling `run()`, not by setting `execute.verbose` directly.

- **Random seeds reset per qubit group.** Most benchmarks call `np.random.seed(0)` at the start of each qubit-width group. This ensures reproducibility but means the same random circuits are generated regardless of which groups are selected.

## Known Problems

These are specific bugs or issues that may be fixed in future releases.

#### Import Patterns

- **Some old-style benchmark files use wrong import pattern.** Files in `{benchmark}/qiskit/` and `{benchmark}/cirq/` subdirectories (pre-v2 implementations) may use `from qedclib.qiskit import execute` directly. This bypasses the dynamic loader and creates a separate module instance. These files are functional but should be migrated to the standard `import execute` pattern after `initialize()`.

- **QRL module identity mismatch.** The quantum reinforcement learning benchmark imports metrics in two different ways, which can cause the metrics module to appear as two separate instances.

#### Benchmark-Specific

- **Some benchmarks have built-in qubit maximums.** Grover's, Shor's, HHL, and VQE impose default qubit limits to avoid runaway execution times. These can be relaxed by editing the default values in the benchmark source files.

<br>
&copy; 2025 Quantum Economic Development Consortium (QED-C). All Rights Reserved.
