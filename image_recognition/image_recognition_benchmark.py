'''
Image Recognition Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.

This is a thin wrapper that delegates to the qiskit implementation.
'''

# Add benchmark home dir to path, so the benchmark can be run without pip installing.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from _common.qedc_init import qedc_benchmarks_init

# Benchmark Name
benchmark_name = "Image Recognition"


def run(
    min_qubits=2, max_qubits=4, skip_qubits=2, max_circuits=3, num_shots=100,
    method=2,
    radius=None, thetas_array=None, parameterized=False, parameter_mode=1, do_fidelities=True,
    minimizer_function=None,
    minimizer_tolerance=1e-3, max_iter=300, comfort=False,
    line_x_metrics=["iteration_count", "cumulative_exec_time", "iteration_count"],
    line_y_metrics=["train_loss", "train_accuracy", "test_accuracy"],
    bar_y_metrics=["average_exec_times", "train_accuracy", "test_accuracy"],
    bar_x_metrics=["num_qubits"],
    score_metric=["train_accuracy"],
    x_metric=["cumulative_exec_time", "cumulative_elapsed_time"],
    y_metric="num_qubits",
    fixed_metrics={},
    num_x_bins=15,
    y_size=None,
    x_size=None,
    show_results_summary=True,
    plot_results=True,
    plot_layout_style="grid",
    show_elapsed_times=True,
    use_logscale_for_times=False,
    save_res_to_file=True, save_final_counts=False, detailed_save_names=False,
    backend_id=None,
    provider_backend=None, hub="ibm-q", group="open", project="main",
    exec_options=None,
    _instances=None,
    ansatz_type='qcnn uniform',
    reps=1,
    batch_size=50,
    backend_id_train='statevector_simulator',
    test_pass_count=30,
    test_size=50,
    train_size=200,
    api=None,
    get_circuits=False,
    draw_circuits=True,
):

    # Configure the QED-C Benchmark package for use with the given API
    # Note: image_recognition only has qiskit implementation
    qedc_benchmarks_init(api if api else "qiskit", "image_recognition", ["image_recognition_benchmark"])

    # Import the actual benchmark module (now available after qedc_init)
    import image_recognition_benchmark as img_impl

    # Use default backend_id if None passed
    if backend_id is None:
        backend_id = "qasm_simulator"

    # Delegate to the implementation
    return img_impl.run(
        min_qubits=min_qubits, max_qubits=max_qubits, skip_qubits=skip_qubits,
        max_circuits=max_circuits, num_shots=num_shots,
        method=method,
        radius=radius, thetas_array=thetas_array, parameterized=parameterized,
        parameter_mode=parameter_mode, do_fidelities=do_fidelities,
        minimizer_function=minimizer_function,
        minimizer_tolerance=minimizer_tolerance, max_iter=max_iter, comfort=comfort,
        line_x_metrics=line_x_metrics,
        line_y_metrics=line_y_metrics,
        bar_y_metrics=bar_y_metrics,
        bar_x_metrics=bar_x_metrics,
        score_metric=score_metric,
        x_metric=x_metric,
        y_metric=y_metric,
        fixed_metrics=fixed_metrics,
        num_x_bins=num_x_bins,
        y_size=y_size,
        x_size=x_size,
        show_results_summary=show_results_summary,
        plot_results=plot_results,
        plot_layout_style=plot_layout_style,
        show_elapsed_times=show_elapsed_times,
        use_logscale_for_times=use_logscale_for_times,
        save_res_to_file=save_res_to_file, save_final_counts=save_final_counts,
        detailed_save_names=detailed_save_names,
        backend_id=backend_id,
        provider_backend=provider_backend, hub=hub, group=group, project=project,
        exec_options=exec_options,
        _instances=_instances,
        ansatz_type=ansatz_type,
        reps=reps,
        batch_size=batch_size,
        backend_id_train=backend_id_train,
        test_pass_count=test_pass_count,
        test_size=test_size,
        train_size=train_size,
        get_circuits=get_circuits,
        draw_circuits=draw_circuits
    )


def load_data_and_plot(folder=None, backend_id=None, **kwargs):
    """
    Load data from a previous run and regenerate plots.
    Delegates to the API-specific implementation.
    Assumes run() or qedc_benchmarks_init() was already called.
    """
    # Import the actual benchmark module (assumes paths already set up)
    import image_recognition_benchmark as img_impl

    # Delegate to the implementation
    img_impl.load_data_and_plot(folder=folder, backend_id=backend_id, **kwargs)


#######################
# MAIN

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Image Recognition Benchmark")
    parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits (min = max = N)", type=int)
    parser.add_argument("--min_qubits", "-min", default=2, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=4, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=2, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--method", "-m", default=2, help="Algorithm Method", type=int)
    parser.add_argument("--max_iter", "-i", default=300, help="Maximum iterations", type=int)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--noplot", "-nop", action="store_true", help="Do not plot results")
    parser.add_argument("--nodraw", "-nod", action="store_true", help="Do not draw circuit diagram")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    if args.num_qubits > 0:
        args.min_qubits = args.max_qubits = args.num_qubits

    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        method=args.method,
        max_iter=args.max_iter,
        backend_id=args.backend_id,
        exec_options={"noise_model": None} if args.nonoise else {},
        api=args.api,
        draw_circuits=not args.nodraw, plot_results=not args.noplot
    )
