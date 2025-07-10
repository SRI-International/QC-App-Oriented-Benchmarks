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

