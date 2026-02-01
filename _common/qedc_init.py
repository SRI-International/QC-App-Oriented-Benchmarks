###############################################################################
# (C) Quantum Economic Development Consortium (QED-C) 2021.
# Technical Advisory Committee on Standards and Benchmarks (TAC)
#

"""
QED-C Benchmarks Initialization Module

This module provides infrastructure for dynamically loading API-specific benchmark implementations.

=================================================================================================
ARCHITECTURE OVERVIEW
=================================================================================================

1. MOTIVATION FOR DYNAMIC LOADING
----------------------------------
The QED-C benchmarks support multiple quantum computing APIs (Qiskit, Cirq, CUDA-Q, Braket, etc.).
Rather than hard-coding imports for each API, we use dynamic loading to:

  - Allow users to specify the API at runtime via command-line arguments
  - Load only the API-specific code needed for that execution
  - Maintain a clean directory structure where implementations are organized by API:
      hidden_shift/
        qiskit/
          hs_kernel.py
        cirq/
          hs_kernel.py
        cudaq/
          hs_kernel.py


2. USAGE IN BENCHMARK PROGRAMS
-------------------------------
Benchmark programs must call qedc_benchmarks_init() AFTER determining the API from command-line
arguments but BEFORE importing any benchmark-specific modules that depend on dynamically loaded code.

Required benchmark structure:

    # At the very top - ensure project root is in sys.path
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

    from _common.qedc_init import qedc_benchmarks_init

    # ... argument parsing to get args.api ...

    # Initialize - loads API-specific modules
    qedc_benchmarks_init(args.api, "benchmark_name", ["kernel_module"])

    # NOW safe to import the dynamically loaded modules
    import kernel_module
    import execute

The initialization function must be called before other imports because those imports may depend
on modules that only exist after dynamic loading completes.


3. RUNNING WITHOUT PIP INSTALL
-------------------------------
The benchmarks can run directly from a cloned repository without pip installation. The 2-line
path setup at the top of each benchmark file ensures the project root is in sys.path:

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

To run from anywhere outside the repository, add the repository path manually:

    import sys
    sys.path.insert(0, "/path/to/QC-App-Oriented-Benchmarks")

Then execute the benchmark normally. This simulates an installed package without requiring
pip install.


4. REQUIREMENTS FOR KERNEL AUTHORS
-----------------------------------
To work within this system, kernel and common files must use appropriate import styles:

  - For installed packages (qiskit, cirq, cudaq, etc.):
      from qiskit import QuantumCircuit  # Absolute import finds real package

  - For top-level _common:
      from _common import metrics  # Works because top-level _common is in sys.path
      from _common.qiskit import execute

  - For modules loaded via qedc_benchmarks_init:
      import hs_kernel  # Available after init, injected into sys.modules
      import execute    # Available after init

Note: The API subdirectories (qiskit/, cudaq/, etc.) under benchmarks and _common are NOT
Python packages (no __init__.py), so relative imports like "from . import something" will
not work from within those directories. Use absolute imports instead.

=================================================================================================
"""

from importlib import import_module
import sys


def qedc_benchmarks_init(api: str, benchmark_name: str, module_names: list[str]) -> None:
    """
    Assigns the modules to sys.modules dictionary only if it doesn't currently exist.  

    Args:
        api: the api to run the benchmark on.
        benchmark_name: the name of the benchmark.
        module_names: the name of the modules to import. 
    """
    if api is None: api = "qiskit"

    # Add all modules from the list
    for module_name in module_names:
        module_path = f"{benchmark_name}.{api}.{module_name}"
        if sys.modules.get(module_name) is None:
            sys.modules[module_name] = import_module(module_path)
    
    # Add execute module
    if sys.modules.get("execute") is None:
        path_to_execute = f"_common.{api}.execute"
        sys.modules["execute"] = import_module(path_to_execute)