###############################################################################
# (C) Quantum Economic Development Consortium (QED-C) 2021.
# Technical Advisory Committee on Standards and Benchmarks (TAC)
#

from importlib import import_module
import sys
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def temp_sys_path_removal(entries_to_remove):
    """
    Temporarily remove the given entris from the path list and restore them at the end.
    """
    original_sys_path = sys.path.copy()
    sys.path = [p for p in sys.path if p not in entries_to_remove]
    try:
        yield
    finally:
        sys.path = original_sys_path
        

def qedc_benchmarks_init(api: str, benchmark_name: str, module_names: list[str]) -> None:
    """
    Assigns the modules to sys.modules dictionary only if it doesn't currently exist.  

    Args:
        api: the api to run the benchmark on.
        benchmark_name: the name of the benchmark.
        module_names: the name of the modules to import. 
    """
    if api is None: api = "qiskit"

    # Temporarily remove current directory to avoid name conflicts
    entries_to_remove = ['', '.', str(Path().resolve())]
    with temp_sys_path_removal(entries_to_remove):
    
        for module_name in module_names:
            module_path = f"{benchmark_name}.{api}.{module_name}"
            #module = import_module(module_path)
            #sys.modules[module_name] = module  # Bare alias like qft_kernel = module
            
            if sys.modules.get(module_name) is None:
                sys.modules[module_name] = import_module(module_path)

        # Add execute module
        if sys.modules.get("execute") is None:
            path_to_execute = f"_common.{api}.execute"
            #sys.modules["execute"] = import_module(path_to_execute)
            sys.modules["execute"] = import_module(path_to_execute)
                 
    
