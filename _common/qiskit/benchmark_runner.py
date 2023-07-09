import importlib
import ast
import argparse
import os
import sys

benchmark_algorithms = [
    "amplitude-estimation",
    "bernstein-vazirani",
    "deutsch-jozsa",
    "grovers",
    "hamiltonian-simulation",
    "hidden-shift",
    "maxcut",
    "monte-carlo",
    "phase-estimation",
    "quantum-fourier-transform",
    "shors",
    "vqe",
]

# Add algorithms to path:
for algorithm in benchmark_algorithms:
    sys.path.insert(1, os.path.join(f"{algorithm}", "qiskit"))

import ae_benchmark
import bv_benchmark
import dj_benchmark
import grovers_benchmark
import hamiltonian_simulation_benchmark
import hs_benchmark
import maxcut_benchmark
import mc_benchmark
import pe_benchmark
import qft_benchmark
import shors_benchmark
import vqe_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking")

    # Universal arguments: These arguments are used by all algorithms in the benchmarking suite.
    parser.add_argument("--algorithm", default="quantum-fourier-transform", help="Benchmarking algorithm to run.", type=str)
    parser.add_argument("--min_qubits", default=2, help="Minimum number of qubits.", type=int)
    parser.add_argument("--max_qubits", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--max_circuits", default=3, help="Maximum number of circuits", type=int)
    parser.add_argument("--num_shots", default=100, help="Number of shots.", type=int)
    parser.add_argument("--backend_id", default="qasm_simulator", help="Backend simulator or hardware string", type=str)
    parser.add_argument("--provider_backend", default=None, help="Provider backend name.", type=str)
    parser.add_argument("--hub", default="ibm-q", help="Computing group hub.", type=str)
    parser.add_argument("--group", default="open", help="Group status", type=str)
    parser.add_argument("--project", default="main", help="Project", type=str)
    parser.add_argument("--provider_module_name", default=None, help="Hardware Provider Module Name", type= str)
    parser.add_argument("--provider_class_name", default=None, help="Hardware Provider Class Name", type= str)
    parser.add_argument("--noise_model", default=None, help="Custom Noise model defined in Custom Folder", type= str)
    parser.add_argument("--exec_options", default={}, help="Additional execution options", type=ast.literal_eval)

    # Additional arguments required by other algorithms.
    parser.add_argument("--epsilon", default=0.05, help="Used for Monte-Carlo", type=float)
    parser.add_argument("--degree", default=2, help="Used for Monte-Carlo", type=int)
    parser.add_argument("--use_mcx_shim", default=False, help="Used for Grovers", type=bool)
    parser.add_argument("--use_XX_YY_ZZ", default=False, help="Used for Hamiltonian-Simulation", type=bool)
    parser.add_argument("--num_state_qubits", default=1, help="Used for amplitude-estimation and Monte-Carlo", type=int)
    parser.add_argument("--method", default=1, help="Used for Bernstein-Vazirani, MaxCut, Monte-Carlo, QFT, Shor, and VQE", type=int)

    # Additional arguments required (only for MaxCut).
    parser.add_argument("--rounds", default=1, help="Used for MaxCut", type=int)
    parser.add_argument("--alpha", default=0.1, help="Used for MaxCut", type=float)
    parser.add_argument("--thetas_array", default=None, help="Used for MaxCut", type=list)
    parser.add_argument("--parameterized", default=False, help="Used for MaxCut", type=bool)
    parser.add_argument("--do_fidelities", default=True, help="Used for MaxCut", type=bool)
    parser.add_argument("--max_iter", default=30, help="Used for MaxCut", type=int)
    parser.add_argument("--score_metric", default="fidelity", help="Used for MaxCut", type=str)
    parser.add_argument("--x_metric", default="cumulative_exec_time", help="Used for MaxCut", type=str)
    parser.add_argument("--y_metric", default="num_qubits", help="Used for MaxCut", type=str)
    parser.add_argument("--fixed_metrics", default={}, help="Used for MaxCut", type=ast.literal_eval)
    parser.add_argument("--num_x_bins", default=15, help="Used for MaxCut", type=int)
    parser.add_argument("--x_size", default=None, help="Used for MaxCut", type=int)
    parser.add_argument("--y_size", default=None, help="Used for MaxCut", type=int)
    parser.add_argument("--use_fixed_angles", default=False, help="Used for MaxCut", type=bool)
    parser.add_argument("--objective_func_type", default='approx_ratio', help="Used for MaxCut", type=str)
    parser.add_argument("--plot_results", default=True, help="Used for MaxCut", type=bool)
    parser.add_argument("--save_res_to_file", default=False, help="Used for MaxCut", type=bool)
    parser.add_argument("--save_final_counts", default=False, help="Used for MaxCut", type=bool)
    parser.add_argument("--detailed_save_names", default=False, help="Used for MaxCut", type=bool)
    parser.add_argument("--comfort", default=False, help="Used for MaxCut", type=bool)
    parser.add_argument("--eta", default=0.5, help="Used for MaxCut", type=float)
    parser.add_argument("--_instance", default=None, help="Used for MaxCut", type=str)

    # Grouping for "common options"
    # i.e. show_plot_images etc. (others in metrics.py)

    args = parser.parse_args()

    # For Inserting the Noise model default into exec option as it is function call 
    if args.noise_model is not None :
        
        if args.noise_model == 'None':
           args.exec_options["noise_model"] = None 
        else:
            module,method = args.noise_model.split(".")
            module = importlib.import_module(f"custom.{module}")
            method = method.split("(")[0]
            custom_noise = getattr(module, method)
            noise = custom_noise()
            args.exec_options["noise_model"] = noise
    
    # Provider detail update using provider module name and class name
    if args.provider_module_name is not None and args.provider_class_name is not None:
        provider_class = getattr(importlib.import_module(args.provider_module_name), args.provider_class_name)
        provider = provider_class()
        provider_backend = provider.get_backend(args.backend_id)
        args.provider_backend = provider_backend
     
    algorithm = args.algorithm

    # Parsing universal arguments.
    universal_args = {
        "min_qubits": args.min_qubits,
        "max_qubits": args.max_qubits,
        "num_shots": args.num_shots,
        "backend_id": args.backend_id,
        "provider_backend": args.provider_backend,
        "hub": args.hub,
        "group": args.group,
        "project": args.project,
        "exec_options": args.exec_options,
    }

    # Parsing additional arguments used in some algorithms.
    additional_args = {
        "epsilon": args.epsilon,
        "degree": args.degree,
        "use_mcx_shim": args.use_mcx_shim,
        "use_XX_YY_ZZ": args.use_XX_YY_ZZ,
        "num_state_qubits": args.num_state_qubits,
        "method": args.method,
    }

    # Parsing arguments for MaxCut
    maxcut_args = {
        "rounds": args.rounds,
        "alpha": args.alpha,
        "thetas_array": args.thetas_array,
        "parameterized": args.parameterized,
        "do_fidelities": args.do_fidelities,
        "max_iter": args.max_iter,
        "score_metric": args.score_metric,
        "x_metric": args.x_metric,
        "y_metric": args.y_metric,
        "fixed_metrics": args.fixed_metrics,
        "num_x_bins": args.num_x_bins,
        "x_size": args.x_size,
        "y_size": args.y_size,
        "use_fixed_angles": args.use_fixed_angles,
        "objective_func_type": args.objective_func_type,
        "plot_results": args.plot_results,
        "save_res_to_file": args.save_res_to_file,
        "save_final_counts": args.save_final_counts,
        "detailed_save_names": args.detailed_save_names,
        "comfort": args.comfort,
        "eta": args.eta,
        "_instance": args._instance,
    }

    if algorithm == "amplitude-estimation":
        universal_args["num_state_qubits"] = additional_args["num_state_qubits"]
        ae_benchmark.run(**universal_args)

    elif algorithm == "bernstein-vazirani":
        universal_args["method"] = additional_args["method"]
        bv_benchmark.run(**universal_args)

    elif algorithm == "deutsch-jozsa":
        dj_benchmark.run(**universal_args)

    elif algorithm == "grovers":
        universal_args["use_mcx_shim"] = additional_args["use_mcx_shim"]
        grovers_benchmark.run(**universal_args)

    elif algorithm == "hamiltonian-simulation":
        universal_args["use_XX_YY_ZZ"] = additional_args["use_XX_YY_ZZ"]
        hamiltonian_simulation_benchmark.run(**universal_args)

    elif algorithm == "hidden-shift":
        hs_benchmark.run(**universal_args)

    elif algorithm == "maxcut":
        maxcut_args = {}
        maxcut_args.update(universal_args)
        maxcut_args.update(maxcut_args)
        maxcut_args["method"] = additional_args["method"]
        maxcut_args["degree"] = additional_args["degree"]
        maxcut_benchmark.run(**maxcut_args)

    elif algorithm == "monte-carlo":
        universal_args["epsilon"] = additional_args["epsilon"]
        universal_args["method"] = additional_args["method"]
        universal_args["degree"] = additional_args["degree"]
        mc_benchmark.run(**universal_args)

    elif algorithm == "phase-estimation":
        pe_benchmark.run(**universal_args)

    elif algorithm == "quantum-fourier-transform":
        universal_args["method"] = additional_args["method"]
        qft_benchmark.run(**universal_args)

    elif algorithm == "shors":
        universal_args["method"] = additional_args["method"]
        shors_benchmark.run(**universal_args)

    elif algorithm == "vqe":
        universal_args["method"] = additional_args["method"]
        vqe_benchmark.run(**universal_args)

    else:
        raise ValueError(f"Algorithm {algorithm} not supported.")
