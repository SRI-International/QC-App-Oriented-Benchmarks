from importlib import import_module
from types import ModuleType


def qedc_benchmarks_init(api: str, benchmark_name: str, module_names: list[str]) -> list[ModuleType]:
    """
    Args:
        api: the api to run the benchmark on.
        benchmark_name: the name of the benchmark.
        module_names: the name of the modules to import.
    
    Returns:
        A list of the modules in the order of module_names. 
    """
    if api is None: api = "qiskit"

    modules = []

    for module_name in module_names:
        module_path = f"{benchmark_name}.{api}.{module_name}"
        modules.append(import_module(module_path))
    
    return modules

def get_execute_module(api: str):
    """
    Args:
        api: the api to run the benchmark on.
    
    Returns:
        The execute module for the api.
    """

    if api is None: api = "qiskit"

    path_to_execute = f"_common.{api}.execute"
    
    return import_module(path_to_execute)
