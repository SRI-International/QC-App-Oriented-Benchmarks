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
import inspect
import sys
from pathlib import Path
from importlib import invalidate_caches

# Store project root for use in path calculations
_project_root = Path(__file__).parent.parent.resolve()

# Default API, set via qedc_set_api()
_current_api = None


def qedc_set_api(api):
    """
    Set the default quantum computing API for all subsequent benchmark operations.

    Call this once at startup (e.g., in the first notebook cell) to avoid passing
    api= to every function call.

    Args:
        api: The API name ("qiskit", "cudaq", "cirq", "braket", etc.)

    Example:
        qedc_set_api("qiskit")
    """
    global _current_api
    _current_api = api


def qedc_get_api():
    """
    Get the current default API. Returns "qiskit" if not explicitly set.
    """
    return _current_api or "qiskit"


def qedc_get_kernel(kernel_name, api=None, benchmark_name=None):
    """
    Initialize and return a kernel module in one call.

    Auto-detects the benchmark directory from the calling file's location unless
    benchmark_name is provided explicitly.

    Args:
        kernel_name: Name of the kernel module to load (e.g., "qft_kernel").
        api: API to use. If None, uses the value set by qedc_set_api() (default "qiskit").
        benchmark_name: Benchmark directory name. If None, auto-detected from caller's directory.

    Returns:
        The loaded kernel module, ready to use.

    Example:
        kernel = qedc_get_kernel("qft_kernel")
        qc = kernel.QuantumFourierTransform(4, 3, [1,1,0,0], method=1)
    """
    if api is None:
        api = qedc_get_api()
    if benchmark_name is None:
        caller_file = inspect.stack()[1].filename
        benchmark_name = Path(caller_file).parent.name

    qedc_benchmarks_init(api, benchmark_name, [kernel_name])
    return sys.modules[kernel_name]


def qedc_is_leader():
    """
    Return True if this process is the MPI leader (rank 0) or MPI is not active.

    Use this to guard output that should only appear once in multi-GPU runs
    (e.g., drawing circuits, plotting results).
    """
    from qedclib import qcb_mpi as mpi
    return mpi.leader()


def qedc_benchmarks_init(api: str, benchmark_name: str = None, module_names: list[str] = None) -> None:
    """
    Dynamically load API-specific benchmark modules and inject them into sys.modules.

    This function sets up sys.path to find kernel implementations and the execute module
    for the specified API, then loads them and makes them available for import throughout
    the benchmark program.

    Can be called with just an API name to initialize the execute module without loading
    any benchmark-specific kernels. This is useful when you need to access execute settings
    (e.g., execute.verbose) before running any benchmarks.

    IMPORTANT: This function must be called AFTER argument parsing (to get the API name)
    but BEFORE importing any benchmark modules that depend on the dynamically loaded code.

    Args:
        api: The quantum computing API to use. One of: "qiskit", "cirq", "cudaq", "braket",
             or None (defaults to "qiskit").
        benchmark_name: The name of the benchmark directory (e.g., "hidden_shift", "hamlib").
                       Must match the actual directory name in the repository.
                       If None, only the common execute module is loaded (no kernels).
        module_names: List of module names to dynamically load from benchmark_name/api/.
                     Example: ["hs_kernel"] will load hidden_shift/qiskit/hs_kernel.py

    Returns:
        None - modules are injected into sys.modules and become available for import

    Side Effects:
        - Modifies sys.path to include API-specific directories
        - Adds entries to sys.modules for each loaded module (using bare names like "hs_kernel")
        - Adds "execute" to sys.modules, pointing to _common/{api}/execute.py

    Example:
        # Initialize just the execute module (e.g., to set verbose before running benchmarks)
        qedc_benchmarks_init("qiskit")
        import execute
        execute.verbose = True

        # In hidden_shift/hs_benchmark.py
        qedc_benchmarks_init("qiskit", "hidden_shift", ["hs_kernel"])

        # After this call, these imports work:
        import hs_kernel  # Points to hidden_shift/qiskit/hs_kernel.py
        import execute    # Points to _common/qiskit/execute.py
    """
    # Initialize MPI early (before any prints) so non-leader ranks get stdout suppressed
    from qedclib import qcb_mpi as mpi
    mpi.init()

    # Print version on first initialization
    import qedclib
    if not getattr(qedc_benchmarks_init, '_initialized', False):
        print(f"QED-C Quantum Circuit Execution Library (qedclib {qedclib.__version__})")
        qedc_benchmarks_init._initialized = True

    # Default to qiskit if no API specified
    if api is None:
        api = "qiskit"

    # Set the default API so get_kernel() and other functions pick it up
    qedc_set_api(api)

    if module_names is None:
        module_names = []

    # Calculate common directories (always needed)
    common_dir = _project_root / "qedclib"
    common_api_dir = common_dir / api

    # Add common directories to sys.path
    for path in [str(common_dir), str(common_api_dir)]:
        if path not in sys.path:
            sys.path.insert(0, path)

    # If a benchmark was specified, set up benchmark-specific paths and load kernels
    if benchmark_name is not None:
        # Look for benchmarks in the qedcbench package directory
        import qedclib
        if qedclib._benchmark_root is not None:
            benchmark_dir = Path(qedclib._benchmark_root) / benchmark_name
        else:
            benchmark_dir = _project_root / "qedcbench" / benchmark_name
        api_dir = benchmark_dir / api

        if str(api_dir) not in sys.path:
            sys.path.insert(0, str(api_dir))

        # For cudaq, may need to reset caches
        if api == "cudaq":
            reset_module_caches(api, benchmark_name, module_names)

        # Dynamically load each requested kernel module
        for module_name in module_names:
            # Check if already loaded to avoid redundant imports
            if sys.modules.get(module_name) is None:
                # Load the module (path is already set up, so bare name works)
                module = import_module(module_name)

                # Inject into sys.modules with bare name for convenient importing
                sys.modules[module_name] = module

    # Load the API-specific execute module from qedclib
    if sys.modules.get("execute") is None:
        module = import_module("execute")
        sys.modules["execute"] = module

    # Make execute accessible as qedclib.execute
    import qedclib
    qedclib.execute = sys.modules["execute"]


def reset_module_caches(api: str, benchmark_name: str, module_names: list[str] = None) -> None:
    """
    Reset the module caches to normalize execution.

    This block of code seems to be required only for cudaq.
    If benchmark launched as benchmark_name/<bn>_benchmark.py, it fails to find the module.
    If launched using -m <benchmark_name>.<bn>_benchmark it would succeed.
    This code normalizes the sys.path stack so both work the same.
    """
    # 1) Normalize sys.path so top-level package import works like `-m`
    repo_root = str(Path(__file__).resolve().parents[1])      # repo root
    pkg_dir = str(Path(repo_root) / benchmark_name)           # e.g. .../bernstein_vazirani
    sys.path = [repo_root] + [p for p in sys.path if p not in (repo_root, pkg_dir)]

    # 2) Remove poisoned/partial module entries (this fixes __spec__ is None)
    if module_names is None:
        module_names = []
    to_clear = {benchmark_name,
                f"{benchmark_name}.{api}"} | {
                f"{benchmark_name}.{api}.{m}" for m in module_names}
    for name in to_clear:
        sys.modules.pop(name, None)

    # 3) Refresh import caches
    invalidate_caches()
