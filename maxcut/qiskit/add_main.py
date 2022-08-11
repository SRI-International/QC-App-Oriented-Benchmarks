import inspect
from qiskit.providers.ibmq.runtime import UserMessenger
def main(backend, user_messenger=UserMessenger(), **kwargs):

    argspec = inspect.getfullargspec(run)
    _args = {x: y for (x, y) in zip(argspec.args, argspec.defaults)}

    _args["provider_backend"] = backend
    _args["backend_id"] = backend.name()
    _args["save_res_to_file"] = False
    _args["save_final_counts"] = False
    _args["plot_results"] = True

    args = {**_args, **kwargs}

    run(**args)
    args.pop('provider_backend')
    args.pop('_instances')

    final_out = dict(
        circuit_metrics=circuit_metrics,
        circuit_metrics_final_iter=circuit_metrics_final_iter,
        circuit_metrics_detail = circuit_metrics_detail,
        circuit_metrics_detail_2 = circuit_metrics_detail_2,
        benchmark_inputs=args
    )

    return final_out