###############################################################################
# (C) Quantum Economic Development Consortium (QED-C) 2021.
# Technical Advisory Committee on Standards and Benchmarks (TAC)
#

"""
QED-C Benchmarks Initialization Module

This module provides infrastructure for dynamically loading API-specific benchmark implementations
while managing Python's import system to avoid namespace collisions.

=================================================================================================
ARCHITECTURE OVERVIEW
=================================================================================================

1. MOTIVATION FOR DYNAMIC LOADING
----------------------------------
The QED-C benchmarks support multiple quantum computing APIs (Qiskit, Cirq, CuQA, Braket, etc.).
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

Dynamic loading via importlib.import_module() allows us to construct module paths at runtime
(e.g., "hidden_shift.qiskit.hs_kernel") and inject them into sys.modules for use throughout
the benchmark.


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
    
    # NOW safe to import benchmark modules that depend on dynamically loaded code
    from benchmark_name._common import utilities

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

=================================================================================================
THE NAMESPACE COLLISION PROBLEM
=================================================================================================

4. API AND _COMMON DIRECTORY SHADOWING
---------------------------------------
Our directory structure creates namespace collisions:

  a) API folder shadowing:
     - We have local folders named "qiskit/", "cirq/", etc.
     - Kernel files need to import from the REAL installed qiskit/cirq packages
     - Python's import system finds the local "qiskit/" folder first, causing:
       
       from qiskit import QuantumCircuit  # ERROR: Finds hidden_shift/qiskit/ instead!

  b) _common folder shadowing:
     - We have _common/ at the project root AND within benchmark directories
     - Benchmark code needs to import from top-level _common/
     - Python finds the local benchmark/_common/ first, causing:
       
       from _common import metrics  # ERROR: Finds hamlib/_common/ instead!


SOLUTION: Temporary sys.path Manipulation
------------------------------------------
During dynamic imports, we temporarily remove directories from sys.path that would cause
shadowing. The isolated_import_context() context manager:

  1. Saves the current sys.path
  2. Removes:
     - Current directory variants ('', '.', cwd)
     - The benchmark directory (e.g., hidden_shift/)
     - The API subdirectory (e.g., hidden_shift/qiskit/)
  3. Performs the dynamic imports (finds real qiskit package, top-level _common)
  4. Restores sys.path to its original state

This is safe because:
  - Changes are temporary and automatically restored via try/finally
  - Project root remains in sys.path, so our imports still work
  - Only affects the import_module() calls, not the rest of the program


REQUIREMENTS FOR KERNEL AUTHORS
--------------------------------
To work within this system, kernel and common files must use appropriate import styles:

  - For installed packages (qiskit, cirq, etc.):
      from qiskit import QuantumCircuit  # Absolute import finds real package
  
  - For top-level _common:
      from _common import metrics  # Works because top-level is in sys.path
      from _common.qiskit import execute
  
  - For local files in same directory:
      from . import constants  # Relative import (dot = current directory)
  
  - For benchmark's _common (from within API subdirectory):
      from .._common import utilities  # Relative import (.. = parent directory)

Relative imports (with dots) are resolved based on the module's package position, not sys.path,
so they work reliably regardless of path manipulation.

=================================================================================================
"""

from importlib import import_module
import sys
from pathlib import Path
from contextlib import contextmanager

# Store project root for use in path calculations
_project_root = Path(__file__).parent.parent.resolve()


@contextmanager
def isolated_import_context(benchmark_name: str, api: str):
    """
    Context manager that temporarily modifies sys.path to prevent namespace collisions
    during dynamic imports.
    
    This removes directories that would cause local folders (qiskit/, _common/) to shadow
    installed packages or top-level modules, while preserving the ability to import from
    the project structure.
    
    Args:
        benchmark_name: Name of the benchmark directory (e.g., "hidden_shift")
        api: Name of the API being used (e.g., "qiskit", "cirq")
    
    Yields:
        None - the modified sys.path is available during the with block
    
    Example:
        with isolated_import_context("hidden_shift", "qiskit"):
            # During this block:
            # - from qiskit import X finds the real qiskit package
            # - from _common import Y finds top-level _common/
            # - import_module("hidden_shift.qiskit.kernel") still works
            module = import_module("hidden_shift.qiskit.hs_kernel")
    """
    # Save original sys.path for restoration
    original_sys_path = sys.path.copy()
    
    # Calculate paths that would cause shadowing
    benchmark_dir = _project_root / benchmark_name
    
    # Paths to remove:
    # - '' and '.' represent the current directory
    # - str(Path.cwd()) is the absolute path to current directory
    # - benchmark_dir would make local qiskit/ shadow real qiskit
    # - benchmark_dir/api is the specific API subdirectory
    paths_to_remove = {
        '',
        '.',
        str(Path.cwd()),
        str(benchmark_dir),
        str(benchmark_dir / api)
    }
    
    # Create cleaned sys.path without problematic directories
    sys.path = [p for p in sys.path if p not in paths_to_remove]
    
    try:
        # Yield control to the with block - imports happen here
        yield
    finally:
        # Always restore original sys.path, even if an exception occurred
        # This ensures we don't leave the import system in a broken state
        sys.path = original_sys_path


def qedc_benchmarks_init(api: str, benchmark_name: str, module_names: list[str]) -> None:
    """
    Dynamically load API-specific benchmark modules and inject them into sys.modules.
    
    This function loads kernel implementations and the execute module for the specified API,
    making them available for import throughout the benchmark program. It uses temporary
    sys.path manipulation to avoid namespace collisions with local directories.
    
    IMPORTANT: This function must be called AFTER argument parsing (to get the API name)
    but BEFORE importing any benchmark modules that depend on the dynamically loaded code.
    
    Args:
        api: The quantum computing API to use. One of: "qiskit", "cirq", "cudaq", "braket",
             or None (defaults to "qiskit").
        benchmark_name: The name of the benchmark directory (e.g., "hidden_shift", "hamlib").
                       Must match the actual directory name in the repository.
        module_names: List of module names to dynamically load from benchmark_name/api/.
                     Example: ["hs_kernel"] will load hidden_shift/qiskit/hs_kernel.py
    
    Returns:
        None - modules are injected into sys.modules and become available for import
    
    Side Effects:
        - Adds entries to sys.modules for each loaded module (using bare names like "hs_kernel")
        - Adds "execute" to sys.modules, pointing to _common/{api}/execute.py
    
    Example:
        # In hidden_shift/hs_benchmark.py
        qedc_benchmarks_init("qiskit", "hidden_shift", ["hs_kernel"])
        
        # After this call, these imports work:
        import hs_kernel  # Points to hidden_shift/qiskit/hs_kernel.py
        import execute    # Points to _common/qiskit/execute.py
    """
    # Default to qiskit if no API specified
    if api is None:
        api = "qiskit"
    
    # Use context manager to temporarily modify sys.path during imports
    # This prevents local qiskit/ and _common/ folders from shadowing real packages
    with isolated_import_context(benchmark_name, api):
        
        # Dynamically load each requested kernel module
        for module_name in module_names:
            # Check if already loaded to avoid redundant imports
            if sys.modules.get(module_name) is None:
                # Construct full module path: benchmark_name.api.module_name
                # Example: "hidden_shift.qiskit.hs_kernel"
                module_path = f"{benchmark_name}.{api}.{module_name}"
                
                # Load the module
                print(f"... import_module({module_path})")
                module = import_module(module_path)
                
                # Inject into sys.modules with bare name for convenient importing
                # This allows "import hs_kernel" instead of "import hidden_shift.qiskit.hs_kernel"
                sys.modules[module_name] = module
        
        # Load the API-specific execute module from top-level _common
        if sys.modules.get("execute") is None:
            # Construct path to execute module: _common.api.execute
            # Example: "_common.qiskit.execute"
            path_to_execute = f"_common.{api}.execute"
            
            print(f"... import_module({path_to_execute})")
            module = import_module(path_to_execute)
            
            # Inject with bare name "execute"
            sys.modules["execute"] = module
    
    # Context manager automatically restores sys.path here
    # Subsequent imports in the benchmark program work normally