# Release Notes

The **QED-C Application-Oriented Benchmarks** suite is continually evolving, with issues fixed and enhancements made to add features.
This section presents a brief record of relevant changes made to each version that has been released. Latest version is at the top.

---

## Version 2.0 Release Notes 

This version constitutes a major refactor of the entire benchmark repository.  For backwards compatibility with externally developed work that uses the repo, use version 1.2.1, the last satable release of the earlier benchmark repository structure.

### Release 2.0.0 - 15 May 2026

TBD ...

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
