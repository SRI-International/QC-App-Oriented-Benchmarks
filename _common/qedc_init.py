from importlib import import_module

def get_from_kernel(benchmark_name: str, kernel_name: str, api: str, to_get: list[str]):
    """
    Args:
        benchmark_name: the name of the benchmark.
        kernel_name: the name of the kernel file for the benchmark.
        api: the api to run the benchmark on. 
        to_get: list of classes/method. to get from the kernel. 
    
    Returns:
        A list of the to_get classes/methods.
        Note that the order will be the same as the to_get list. 
    """
    kernel_path = f"{benchmark_name}.{api}.{kernel_name}"

    kernel = import_module(kernel_path)

    output = []

    for name in to_get:
        output.append(getattr(kernel, name))
    
    return output

def get_execute_module(api: str):
    """
    Args:
        api: the api to run the benchmark on.
    
    Returns:
        The execute module for the api.
    """

    path_to_execute = f"_common.{api}.execute"
    return import_module(path_to_execute)
