###############################################################################
# (C) Quantum Economic Development Consortium (QED-C) 2021.
# Technical Advisory Committee on Standards and Benchmarks (TAC)
#
"""
What this does:

Module-level code (runs on import): Removes '', '.', cwd and adds project root first

This fixes: from _common import metrics finding top-level _common ✓


Context manager (runs during dynamic imports): ALSO removes benchmark directories

This fixes: from qiskit import QuantumCircuit finding real qiskit ✓
"""

# _common/qedc_init.py

from importlib import import_module
import sys
from pathlib import Path
from contextlib import contextmanager  

_project_root = Path(__file__).parent.parent.resolve() 
    
@contextmanager
def isolated_import_context(benchmark_name: str, api: str):
    """
    Remove benchmark directories temporarily during dynamic imports.
    """
    original_sys_path = sys.path.copy()
    
    benchmark_dir = _project_root / benchmark_name
    #additional_removals = {str(benchmark_dir), str(benchmark_dir / api)}
    paths_to_remove = {'', '.', str(Path.cwd()), str(benchmark_dir), str(benchmark_dir / api)}
    
    sys.path = [p for p in sys.path if p not in paths_to_remove]
    
    try:
        yield
    finally:
        sys.path = original_sys_path


def qedc_benchmarks_init(api: str, benchmark_name: str, module_names: list[str]) -> None:
    """
    Dynamically imports benchmark kernel modules from the "api" directory..
    """
    if api is None:
        api = "qiskit"
    
    # Use context manager ONLY for the dynamic imports
    with isolated_import_context(benchmark_name, api):
        for module_name in module_names:
            if sys.modules.get(module_name) is None:
                module_path = f"{benchmark_name}.{api}.{module_name}"
                #print(f"... import_module({module_path})")
                sys.modules[module_name] = import_module(module_path)
        
        if sys.modules.get("execute") is None:
            path_to_execute = f"_common.{api}.execute"
            #print(f"... import_module({path_to_execute})")
            sys.modules["execute"] = import_module(path_to_execute)
            