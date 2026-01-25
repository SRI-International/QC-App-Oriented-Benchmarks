'''
MaxCut Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.

This is a thin wrapper that delegates to the API-specific implementation.
'''

# Add benchmark home dir to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from _common.qedc_init import qedc_benchmarks_init

# Benchmark Name
benchmark_name = "MaxCut"


def run(min_qubits=3, max_qubits=6, skip_qubits=2,
        max_circuits=1, num_shots=100,
        method=1, rounds=1, degree=3, alpha=0.1, thetas_array=None,
        parameterized=False, do_fidelities=True,
        max_iter=30, score_metric='fidelity', x_metric='cumulative_exec_time', y_metric='num_qubits',
        fixed_metrics={}, num_x_bins=15, y_size=None, x_size=None, use_fixed_angles=False,
        objective_func_type='approx_ratio', plot_results=True,
        save_res_to_file=False, save_final_counts=False, detailed_save_names=False, comfort=False,
        backend_id=None, provider_backend=None, eta=0.5,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None,
        min_annealing_time=1, max_annealing_time=200,
        api=None,
        _instances=None):

    # Determine which API to use
    selected_api = api if api else "qiskit"

    # Configure the QED-C Benchmark package for use with the given API
    qedc_benchmarks_init(selected_api, "maxcut", ["maxcut_benchmark"])

    # Import the actual benchmark module (now available after qedc_init)
    import maxcut_benchmark as maxcut_impl

    # Use default backend_id if None passed
    if backend_id is None:
        backend_id = "qasm_simulator"

    # Build common parameters
    kwargs = dict(
        min_qubits=min_qubits, max_qubits=max_qubits,
        max_circuits=max_circuits, num_shots=num_shots,
        method=method, degree=degree, alpha=alpha, thetas_array=thetas_array,
        parameterized=parameterized, do_fidelities=do_fidelities,
        max_iter=max_iter, score_metric=score_metric, x_metric=x_metric, y_metric=y_metric,
        fixed_metrics=fixed_metrics, num_x_bins=num_x_bins, y_size=y_size, x_size=x_size,
        objective_func_type=objective_func_type, plot_results=plot_results,
        save_res_to_file=save_res_to_file, save_final_counts=save_final_counts,
        detailed_save_names=detailed_save_names, comfort=comfort,
        backend_id=backend_id, provider_backend=provider_backend, eta=eta,
        hub=hub, group=group, project=project, exec_options=exec_options,
        _instances=_instances,
    )

    # Add API-specific parameters
    if selected_api == "qiskit":
        kwargs.update(
            skip_qubits=skip_qubits,
            rounds=rounds,
            use_fixed_angles=use_fixed_angles,
            context=context,
        )
    elif selected_api == "ocean":
        kwargs.update(
            min_annealing_time=min_annealing_time,
            max_annealing_time=max_annealing_time,
        )

    # Delegate to the implementation
    maxcut_impl.run(**kwargs)


def load_data_and_plot(folder=None, backend_id=None, api=None, **kwargs):
    """
    Load data from a previous run and regenerate plots.
    Delegates to the API-specific implementation.
    Assumes run() or qedc_benchmarks_init() was already called.
    """
    # Import the actual benchmark module (assumes paths already set up)
    import maxcut_benchmark as maxcut_impl

    # Delegate to the implementation
    maxcut_impl.load_data_and_plot(folder=folder, backend_id=backend_id, **kwargs)


#######################
# MAIN

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="MaxCut Benchmark")
    parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits (min = max = N)", type=int)
    parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=6, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=2, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=1, help="Maximum circuit repetitions", type=int)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument("--rounds", "-r", default=1, help="Number of QAOA rounds", type=int)
    parser.add_argument("--degree", "-d", default=3, help="Degree of graph", type=int)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    if args.num_qubits > 0:
        args.min_qubits = args.max_qubits = args.num_qubits

    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        method=args.method,
        rounds=args.rounds,
        degree=args.degree,
        backend_id=args.backend_id,
        exec_options={"noise_model": None} if args.nonoise else {},
        api=args.api
    )
