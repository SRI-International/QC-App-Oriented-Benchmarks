# About

## Project Origin

The **QED-C Application-Oriented Benchmarks** were developed by the Quantum Economic Development Consortium (QED-C) Technical Advisory Committee on Standards and Performance Metrics (Standards TAC). The project began in 2019 with the goal of providing standardized, application-relevant benchmarks for comparing quantum computing platforms.

Unlike gate-level or volume-based benchmarks, these benchmarks measure performance on complete quantum algorithms — providing an end-user perspective on what today's quantum computers can actually accomplish.

## Repository Structure

```
QC-App-Oriented-Benchmarks/
├── README.md               # Project overview and links
├── pyproject.toml          # Package configuration (pip install -e .)
│
├── qedclib/                # Execution and metrics library
│   ├── __init__.py         # Top-level API: initialize(), set_api(), get_kernel(), etc.
│   ├── metrics.py          # Metrics collection, aggregation, and plotting
│   ├── qcb_mpi.py         # MPI support for multi-GPU execution
│   ├── job_store.py        # Job persistence across sessions
│   ├── _init_engine.py     # Dynamic module loading system
│   ├── qiskit/             # Qiskit execution backend (no __init__.py)
│   ├── cudaq/              # CUDA-Q execution backend
│   ├── cirq/               # Cirq execution backend
│   ├── braket/             # Braket execution backend
│   ├── ocean/              # Ocean execution backend
│   ├── custom/             # Custom noise models
│   ├── executors/          # Third-party executors (BlueQubit, cuQuantum, Fire Opal)
│   ├── transformers/       # Compiler optimizations (tket, TrueQ, Qiskit PassMgr)
│   └── postprocessors/     # Error mitigation (mthree)
│
├── qedcbench/              # Benchmark applications (17 benchmarks)
│   ├── __init__.py         # Registers benchmark root with qedclib
│   ├── hidden_shift/       # Each has README.md + API subdirs
│   ├── hamlib/
│   ├── quantum_fourier_transform/
│   └── ...
│
├── doc/                    # Documentation
│   ├── index.md            # Documentation overview
│   ├── quick_start.md      # First-time user guide
│   ├── user_guide.md       # Complete reference
│   ├── release_notes.md    # Version history
│   ├── known_issues.md     # PAL (Problems, Anomalies, Limitations)
│   ├── about.md            # This file
│   ├── setup/              # Platform-specific setup guides
│   └── _design/            # Internal design documents
│
```

## Two Packages, One Install

The repository provides two importable Python packages from a single installation:

- **qedclib** — The execution and metrics library. Useful independently for researchers who want metrics tracking, backend abstraction, and MPI support for their own quantum programs.

- **qedcbench** — The 17 benchmark applications. Depends on qedclib. Provides standardized benchmarks across multiple quantum platforms.

Install both with: `pip install -e .`

## Related Repositories

- **QC-App-Benchmarks-Data** — Benchmark execution scripts, data collection results, and plotting programs for generating publication figures.

- **qhpctools** — HPC utilities for running benchmarks on NERSC/Slurm clusters (GPU allocation scripts, job monitoring).

## Project Lead

This project was created and is maintained by **Thomas Lubinski**, former Chair of the QED-C Technical Advisory Committee on Standards and Performance Metrics and currently Sub-committee Lead for Quantum Computing. Tom has led the design, development, and evolution of the benchmarking framework since its inception in 2019, including authoring the core execution library (qedclib), directing the research agenda, and mentoring summer interns whose work has contributed to multiple published papers using this repository.

## Contributors

The benchmarks and supporting research have benefited from contributions by members of the QED-C Standards TAC, summer research interns (2019–present), and the broader quantum computing community.

## License

Copyright 2025 Quantum Economic Development Consortium (QED-C). All Rights Reserved.

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

## Version

Current version: **2.0.6** (June 2026)

See [Release Notes](./release_notes.md) for version history.
