# Release Notes

The **QED-C Application-Oriented Benchmarks** suite is continually evolving, with issues fixed and enhancements made to add features.
This section presents a brief record of relevant changes made to each version that has been released. Latest version is at the top.

### Release 1.2.1 - 13 April 2026

- **Refactor of API-specific Execution Logic** This set of changes consolidated and normalized the result object returned from execution of a quantum circuit across all APIs.  The ExecutionResult object provides the get_counts() method which returns a dictionary of measurment results.  The counts dict can be populated in serverl methods unique to the API.

### Release 1.2.0 - 11 April 2026

- **Added Support for Parallel Circuit Execution** The Hamlib benchmark on CUDA-Q now supports an execution mode in which a subset of circuits generated for the selected Hamiltonian simulation are executed in parallel in order to reduce the total execution time.

### Release 1.1.0 - 25 January 2026

- **Completed Refactoring of API-specific kernel Files** All benchmarks are now structured so that API-specific code, e.g for Qiskit or CUDA-Q, is contained in a correspondingly named sub-directory. The primary benchmark file resides at the benchmark directory level and is API-agnostic.  The --app (-a) command line argument is used to specify the name of the API, cudaq or qiskit, from which the appropriate kernel code will be dynamically imported.

### Release 1.0.0 - Prior to 12 November 2025

- **Multiple Years of Development and Testing** Since 2019, QED-C has been working to develop, test, and polish the complete set of benchmark programs.

<br>
&nbsp;&copy;&nbsp;2025 Quantum Economic Development Consortium (QED-C). All Rights Reserved.
