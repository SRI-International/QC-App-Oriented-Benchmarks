import inspect
from qiskit.providers.ibmq.runtime import UserMessenger
import time
def main(backend, user_messenger=UserMessenger(), **kwargs):

    # Stores all default parameters in dictionary
    argspec = inspect.getfullargspec(run)
    _args = {x: y for (x, y) in zip(argspec.args, argspec.defaults)}

    _args["provider_backend"] = backend
    _args["backend_id"] = backend.name()

    # Removing saving and plotting because they will not work on runtime servers 
    _args["save_res_to_file"] = False
    _args["save_final_counts"] = False
    _args["plot_results"] = False

    args = {**_args, **kwargs}

    start = time.perf_counter()
    run(**args)
    _wall_time = time.perf_counter() - start

    # Remove provider_backend becuase it is not JSON Serializable
    args['provider_backend'] = None
    # Remove instances because we already have them stored and they are large
    args.pop('_instances')

    final_out = dict(
        circuit_metrics=circuit_metrics,
        circuit_metrics_final_iter=circuit_metrics_final_iter,
        circuit_metrics_detail = circuit_metrics_detail,
        circuit_metrics_detail_2 = circuit_metrics_detail_2,
        benchmark_inputs=args,
        wall_time=_wall_time,
    )

    return final_out