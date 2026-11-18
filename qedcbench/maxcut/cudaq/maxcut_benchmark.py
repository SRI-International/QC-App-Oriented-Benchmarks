"""
MaxCut Benchmark Program - CUDA-Q

This module implements methods 1 and 2 of the QED-C MaxCut benchmark using CUDA-Q.
It mirrors the Qiskit QAOA circuit: an H layer, one cost-unitary RZZ layer per
edge, and one RX mixer layer per round.
"""

import datetime
import json
import logging
import math
import os
import time
from typing import List

import cudaq
import numpy as np
from scipy.optimize import minimize

from maxcut._common import common
from qedclib import metrics
from qedclib import qcb_mpi as mpi
from qedclib.cudaq import execute as ex


benchmark_name = "MaxCut"

np.random.seed(0)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s - %(levelname)s:%(message)s",
)

maxcut_inputs = dict()
verbose = False
print_sample_circuit = True
do_compute_expectation = True

QC_ = None

MAX_QUBITS = 40

# CUDA-Q sample bitstrings are ordered with qubit 0 at string index 0.
reverseStep = 1

iter_dist = {"cuts": [], "counts": [], "sizes": []}
iter_size_dist = {"unique_sizes": [], "unique_counts": [], "cumul_counts": []}
saved_result = {}
minimizer_loop_index = 0
opt_ts = 0
_parameterized_circuit_cache = {}


@cudaq.kernel
def _apply_qaoa(qubits: cudaq.qview, edge_i: List[int], edge_j: List[int],
                num_edges: int, betas: List[float], gammas: List[float],
                rounds: int):
    for i_qubit in range(qubits.size()):
        h(qubits[i_qubit])

    for i_round in range(rounds):
        gamma = gammas[i_round]
        beta = betas[i_round]

        for i_edge in range(num_edges):
            source = edge_i[i_edge]
            target = edge_j[i_edge]
            x.ctrl(qubits[source], qubits[target])
            rz(-gamma, qubits[target])
            x.ctrl(qubits[source], qubits[target])

        for i_qubit in range(qubits.size()):
            rx(2.0 * beta, qubits[i_qubit])


@cudaq.kernel
def maxcut_kernel(num_qubits: int, edge_i: List[int], edge_j: List[int],
                  num_edges: int, betas: List[float], gammas: List[float],
                  rounds: int):
    qubits = cudaq.qvector(num_qubits)
    _apply_qaoa(qubits, edge_i, edge_j, num_edges, betas, gammas, rounds)
    mz(qubits)


@cudaq.kernel
def maxcut_state_kernel(num_qubits: int, edge_i: List[int], edge_j: List[int],
                        num_edges: int, betas: List[float],
                        gammas: List[float], rounds: int):
    qubits = cudaq.qvector(num_qubits)
    _apply_qaoa(qubits, edge_i, edge_j, num_edges, betas, gammas, rounds)


def _normalize_thetas(thetas_array, rounds):
    if thetas_array is None:
        thetas_array = 2 * rounds * [1.0]

    p = len(thetas_array) // 2
    if rounds < p:
        p = rounds
        thetas_array = thetas_array[:2 * rounds]
    elif rounds > p:
        rounds = p
        print(f"WARNING: rounds is greater than length of thetas_array/2; using rounds={rounds}")

    betas = [float(theta) for theta in thetas_array[:p]]
    gammas = [float(theta) for theta in thetas_array[p:]]
    return betas, gammas, p


def _edge_lists(edges):
    edge_i = [int(edge[0]) for edge in edges]
    edge_j = [int(edge[1]) for edge in edges]
    return edge_i, edge_j, len(edges)


def _bound_circuit(num_qubits, edge_i, edge_j, num_edges, betas, gammas, rounds):
    return [
        maxcut_kernel,
        [num_qubits, edge_i, edge_j, num_edges, betas, gammas, rounds],
        {"counts_dict": True},
    ]


def MaxCut(num_qubits, secret_int, edges, rounds, thetas_array, parameterized,
           measured=True):
    if parameterized:
        return MaxCut_param(num_qubits, secret_int, edges, rounds, thetas_array)

    betas, gammas, p = _normalize_thetas(thetas_array, rounds)
    edge_i, edge_j, num_edges = _edge_lists(edges)

    if do_compute_expectation:
        logger.info("Computing expectation")
        compute_expectation(num_qubits, secret_int, edge_i, edge_j,
                            num_edges, betas, gammas, p)

    circuit = _bound_circuit(num_qubits, edge_i, edge_j, num_edges, betas,
                             gammas, p)

    global QC_
    if QC_ is None or num_qubits <= 6:
        if num_qubits < 9:
            QC_ = circuit

    return circuit, None


def MaxCut_param(num_qubits, secret_int, edges, rounds, thetas_array):
    # CUDA-Q decorated kernels are parameterized by their Python arguments. This
    # mirrors Qiskit's parameterized path by reusing a static kernel/topology
    # handle and carrying the current beta/gamma values as separate bindings.
    betas, gammas, p = _normalize_thetas(thetas_array, rounds)
    edge_i, edge_j, num_edges = _edge_lists(edges)

    if do_compute_expectation:
        logger.info("Computing expectation")
        compute_expectation(num_qubits, secret_int, edge_i, edge_j,
                            num_edges, betas, gammas, p)

    cache_key = (num_qubits, tuple(edge_i), tuple(edge_j), p)
    if cache_key not in _parameterized_circuit_cache:
        _parameterized_circuit_cache[cache_key] = [
            maxcut_kernel,
            [num_qubits, edge_i, edge_j, num_edges, [], [], p],
            {
                "counts_dict": True,
                "parameterized_indices": {"betas": 4, "gammas": 5},
            },
        ]

    circuit = _parameterized_circuit_cache[cache_key]
    params = {"betas": betas, "gammas": gammas}

    global QC_
    if QC_ is None or num_qubits <= 6:
        if num_qubits < 9:
            QC_ = _bound_circuit(num_qubits, edge_i, edge_j, num_edges,
                                 betas, gammas, p)

    return circuit, params


############### Expectation Tables

expectations = {}


def _state_index_to_cudaq_key(index, num_qubits):
    return format(index, f"0{num_qubits}b")[::-1]


def compute_expectation(num_qubits, secret_int, edge_i, edge_j, num_edges,
                        betas, gammas, rounds):
    state = cudaq.get_state(maxcut_state_kernel, num_qubits, edge_i, edge_j,
                            num_edges, betas, gammas, rounds)
    statevector = np.asarray(state, dtype=np.complex128)

    counts = {}
    for index, amplitude in enumerate(statevector):
        probability = float(np.abs(amplitude) ** 2)
        if probability > 1e-15:
            counts[_state_index_to_cudaq_key(index, num_qubits)] = probability

    id = f"_{num_qubits}_{secret_int}"
    expectations[id] = counts


def get_expectation(num_qubits, degree, num_shots):
    id = f"_{num_qubits}_{degree}"
    if id not in expectations:
        return None

    counts = expectations[id]
    scaled_counts = {bitstring: round(probability * num_shots)
                     for bitstring, probability in counts.items()}
    del expectations[id]
    return scaled_counts


############### Result Data Analysis

expected_dist = {}


def analyze_and_print_result(qc, result, num_qubits, num_shots, secret_int=None):
    global expected_dist

    counts = result.get_counts(qc)
    expected_dist = get_expectation(num_qubits, secret_int, num_shots)
    if expected_dist is None:
        expected_dist = counts

    if verbose:
        print(f"For width {num_qubits} problem {secret_int}\n  measured: {counts}\n  expected: {expected_dist}")

    fidelity = metrics.polarization_fidelity(counts, expected_dist)
    return counts, fidelity


def compute_cutsizes(results, nodes, edges):
    counts_dict = results.get_counts()
    cuts = list(counts_dict.keys())
    counts = list(counts_dict.values())
    sizes = [common.eval_cut(nodes, edges, cut, reverseStep) for cut in cuts]
    return cuts, counts, sizes


def get_size_dist(counts, sizes):
    # dict.fromkeys preserves insertion order; set iteration order varies
    # across processes under hash randomization and can desync MPI ranks.
    unique_sizes = list(dict.fromkeys(sizes))
    unique_counts = [0] * len(unique_sizes)

    for i_size, size in enumerate(unique_sizes):
        corresp_counts = [
            counts[ind] for ind, this_size in enumerate(sizes)
            if this_size == size
        ]
        unique_counts[i_size] = sum(corresp_counts)

    size_count_pairs = sorted(
        [[size, count] for size, count in zip(unique_sizes, unique_counts)],
        key=lambda pair: pair[0],
    )
    unique_sizes = [pair[0] for pair in size_count_pairs]
    unique_counts = [pair[1] for pair in size_count_pairs]
    cumul_counts = np.cumsum(unique_counts)
    return unique_counts, unique_sizes, cumul_counts.tolist()


def compute_sample_mean(counts, sizes, **kwargs):
    counts, sizes = np.array(counts), np.array(sizes)
    return -np.sum(counts * sizes) / np.sum(counts)


def compute_cvar(counts, sizes, alpha=0.1, **kwargs):
    counts, sizes = np.array(counts), np.array(sizes)
    sort_inds = np.argsort(-sizes)
    sizes = sizes[sort_inds]
    counts = counts[sort_inds]

    num_avgd = math.ceil(alpha * np.sum(counts))
    cvar_sum = 0
    counts_so_far = 0
    for count, size in zip(counts, sizes):
        if counts_so_far + count >= num_avgd:
            cts_to_consider = num_avgd - counts_so_far
            cvar_sum += cts_to_consider * size
            break

        counts_so_far += count
        cvar_sum += count * size

    return -cvar_sum / num_avgd


def compute_gibbs(counts, sizes, eta=0.5, **kwargs):
    counts, sizes = np.array(counts), np.array(sizes)
    largest_size = max(sizes)
    shifted_sizes = sizes - largest_size
    return (
        -eta * largest_size
        - np.log(np.sum(counts / np.sum(counts) * np.exp(eta * shifted_sizes)))
    )


def compute_best_cut_from_measured(counts, sizes, **kwargs):
    return -np.max(sizes)


def compute_quartiles(counts, sizes):
    counts, sizes = np.array(counts), np.array(sizes)

    sort_inds = np.argsort(sizes)
    sizes = sizes[sort_inds]
    counts = counts[sort_inds]
    num_shots = np.sum(counts)

    q_vals = [0.25, 0.5, 0.75]
    ct_vals = [math.floor(q_val * num_shots) for q_val in q_vals]

    cumsum_counts = np.cumsum(counts)
    locs = np.searchsorted(cumsum_counts, ct_vals)
    return sizes[locs]


def uniform_cut_sampling(num_qubits, degree, num_shots, _instances=None):
    instance_filename = os.path.join(
        os.path.dirname(__file__),
        "..",
        "_common",
        common.INSTANCE_DIR,
        f"mc_{num_qubits:03d}_{degree:03d}_000.txt",
    )
    nodes, edges = common.read_maxcut_instance(instance_filename, _instances)

    unif_cuts = np.random.randint(2 ** num_qubits, size=num_shots).tolist()
    unif_cuts_uniq = list(dict.fromkeys(unif_cuts))
    unif_counts = [unif_cuts.count(cut) for cut in unif_cuts_uniq]
    unif_cuts = unif_cuts_uniq

    def int_to_bs(numb):
        bitstring = format(numb, "b")
        return "0" * (num_qubits - len(bitstring)) + bitstring

    unif_cuts = [int_to_bs(i_cut) for i_cut in unif_cuts]
    unif_sizes = [
        common.eval_cut(nodes, edges, cut, reverseStep) for cut in unif_cuts
    ]
    unique_counts_unif, unique_sizes_unif, cumul_counts_unif = get_size_dist(
        unif_counts, unif_sizes
    )

    return (
        unif_cuts,
        unif_counts,
        unif_sizes,
        unique_counts_unif,
        unique_sizes_unif,
        cumul_counts_unif,
    )


def store_final_iter_to_metrics_json(num_qubits, degree, restart_ind,
                                     num_shots, converged_thetas_list, opt,
                                     iter_size_dist, iter_dist,
                                     parent_folder_save, dict_of_inputs,
                                     save_final_counts, save_res_to_file,
                                     _instances=None):
    _, _, _, unique_counts_unif, unique_sizes_unif, cumul_counts_unif = (
        uniform_cut_sampling(num_qubits, degree, num_shots, _instances)
    )
    unif_dict = {
        "unique_counts_unif": unique_counts_unif,
        "unique_sizes_unif": unique_sizes_unif,
        "cumul_counts_unif": cumul_counts_unif,
    }

    metrics.store_props_final_iter(num_qubits, restart_ind, "optimal_value", opt)
    metrics.store_props_final_iter(num_qubits, restart_ind, None, iter_size_dist)
    metrics.store_props_final_iter(
        num_qubits, restart_ind, "converged_thetas_list", converged_thetas_list
    )
    metrics.store_props_final_iter(num_qubits, restart_ind, None, unif_dict)

    if save_res_to_file:
        dump_to_json(
            parent_folder_save,
            num_qubits,
            restart_ind,
            iter_size_dist,
            iter_dist,
            dict_of_inputs,
            converged_thetas_list,
            opt,
            unif_dict,
            save_final_counts=save_final_counts,
        )


def dump_to_json(parent_folder_save, num_qubits, restart_ind, iter_size_dist,
                 iter_dist, dict_of_inputs, converged_thetas_list, opt,
                 unif_dict, save_final_counts=False):
    if not mpi.leader():
        return

    if not os.path.exists(parent_folder_save):
        os.makedirs(parent_folder_save)

    store_loc = os.path.join(
        parent_folder_save, f"width_{num_qubits}_restartInd_{restart_ind}.json"
    )
    all_restart_ids = list(metrics.circuit_metrics[str(num_qubits)].keys())
    ids_this_restart = [
        circuit_id for circuit_id in all_restart_ids
        if int(circuit_id) // 1000 == restart_ind
    ]
    iterations_dict_this_restart = {
        circuit_id: metrics.circuit_metrics[str(num_qubits)][circuit_id]
        for circuit_id in ids_this_restart
    }

    dict_to_store = {
        "iterations": iterations_dict_this_restart,
        "general_properties": dict_of_inputs,
        "converged_thetas_list": converged_thetas_list,
        "optimal_value": opt,
        "unif_dict": unif_dict,
        "final_size_dist": iter_size_dist,
    }
    if save_final_counts:
        dict_to_store["final_counts"] = iter_dist

    with open(store_loc, "w") as outfile:
        json.dump(dict_to_store, outfile)


def get_random_angles(rounds, restarts):
    theta_min = [0] * 2 * rounds
    theta_max = [np.pi] * rounds + [2 * np.pi] * rounds
    thetas = np.random.uniform(
        low=theta_min, high=theta_max, size=(restarts, 2 * rounds)
    )
    return thetas.tolist()


def get_restart_angles(thetas_array, rounds, restarts):
    assert type(restarts) == int and restarts > 0, (
        "max_circuits must be an integer greater than 0"
    )
    default_angles = [[1] * 2 * rounds]
    default_restarts = 1
    if thetas_array is None:
        if restarts == 1:
            return default_angles, default_restarts
        return get_random_angles(rounds, restarts), restarts

    if type(thetas_array) != list:
        print("thetas_array is not a list. Using random angles.")
        return get_random_angles(rounds, restarts), restarts

    if not all([type(item) == list for item in thetas_array]):
        print("thetas_array is not a list of lists. Using random angles.")
        return get_random_angles(rounds, restarts), restarts

    if not all([len(item) == 2 * rounds for item in thetas_array]):
        print("Each element of thetas_array must be a list of length 2 * rounds. Using random angles.")
        return get_random_angles(rounds, restarts), restarts

    return thetas_array, len(thetas_array)


def _cudaq_backend_id(backend_id):
    if backend_id in (None, "qasm_simulator", "statevector_simulator",
                      "cudaq_simulator"):
        return None
    return backend_id


def _theta_list(thetas_array):
    if hasattr(thetas_array, "tolist"):
        return thetas_array.tolist()
    return list(thetas_array)


def run(min_qubits=3, max_qubits=6, skip_qubits=2,
        max_circuits=1, num_shots=100,
        method=1, rounds=1, degree=3, alpha=0.1, thetas_array=None,
        parameterized=False, do_fidelities=True,
        max_iter=30, score_metric="fidelity", x_metric="cumulative_exec_time",
        y_metric="num_qubits", fixed_metrics={}, num_x_bins=15, y_size=None,
        x_size=None, use_fixed_angles=False, objective_func_type="approx_ratio",
        plot_results=True, save_res_to_file=False, save_final_counts=False,
        detailed_save_names=False, comfort=False, backend_id=None,
        provider_backend=None, eta=0.5, hub="ibm-q", group="open",
        project="main", exec_options=None, context=None, _instances=None,
        warmup=False, get_circuits=False, draw_circuits=True):
    if method not in (1, 2):
        raise NotImplementedError("CUDA-Q MaxCut currently supports methods 1 and 2 only.")

    dict_of_inputs = locals()
    thetas, max_circuits = get_restart_angles(thetas_array, rounds, max_circuits)
    dict_of_inputs = {**dict_of_inputs,
                      **{"thetas_array": thetas, "max_circuits": max_circuits}}
    for key in ["hub", "group", "project", "provider_backend", "exec_options"]:
        dict_of_inputs.pop(key)

    global maxcut_inputs
    maxcut_inputs = dict_of_inputs

    global QC_
    global minimizer_loop_index
    global opt_ts
    QC_ = None

    mpi.init()

    print(f"{benchmark_name} ({method}) Benchmark Program - CUDA-Q")

    if detailed_save_names:
        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        parent_folder_save = os.path.join(
            "__results", f"{backend_id}", objective_func_type,
            f"run_start_{start_time_str}",
        )
    else:
        parent_folder_save = os.path.join("__results", f"{backend_id}", objective_func_type)

    if save_res_to_file and mpi.leader() and not os.path.exists(parent_folder_save):
        os.makedirs(parent_folder_save)

    max_qubits = max(4, max_qubits)
    max_qubits = min(MAX_QUBITS, max_qubits)
    min_qubits = min(max(4, min_qubits), max_qubits)
    skip_qubits = max(2, skip_qubits)

    if context is None:
        context = f"{benchmark_name} ({method}) Benchmark"

    degree = max(3, degree)
    rounds = max(1, rounds)

    global do_compute_expectation
    do_compute_expectation = do_fidelities

    if y_size is None:
        y_size = 1.5

    possible_approx_ratios = {
        "cvar_ratio", "approx_ratio", "gibbs_ratio", "bestcut_ratio"
    }
    non_objFunc_ratios = possible_approx_ratios - {objective_func_type}
    function_mapper = {
        "cvar_ratio": compute_cvar,
        "approx_ratio": compute_sample_mean,
        "gibbs_ratio": compute_gibbs,
        "bestcut_ratio": compute_best_cut_from_measured,
    }

    if use_fixed_angles:
        fixed_angles = common.read_fixed_angles(
            os.path.join(os.path.dirname(__file__), "..", "_common",
                         "angles_regular_graphs.json"),
            _instances,
        )
        thetas_array = common.get_fixed_angles_for(fixed_angles, degree, rounds)
        if thetas_array is None:
            print(f"ERROR: no fixed angles for rounds = {rounds}")
            return

    metrics.init_metrics(warmup)

    def execution_handler(qc, result, num_qubits, circuit_id, num_shots):
        num_qubits = int(num_qubits)
        if not do_compute_expectation:
            return
        _, fidelity = analyze_and_print_result(
            qc, result, num_qubits, num_shots, secret_int=int(circuit_id)
        )
        metrics.store_metric(num_qubits, circuit_id, "fidelity", fidelity)

    def execution_handler2(qc, result, num_qubits, circuit_id, num_shots):
        global saved_result
        saved_result = result

    if method == 2:
        ex.init_execution(execution_handler2)
    else:
        ex.init_execution(execution_handler)

    ex.set_execution_target(
        _cudaq_backend_id(backend_id),
        provider_backend=provider_backend,
        hub=hub,
        group=group,
        project=project,
        exec_options=exec_options,
        context=context,
    )

    if get_circuits and method != 1:
        print(f"WARNING: get_circuits is not supported for method {method}")
        return None

    if get_circuits:
        all_qcs = {}

    for num_qubits in range(min_qubits, max_qubits + 1, 2):
        if method == 1:
            if not get_circuits:
                print(f"************\nExecuting [{max_circuits}] circuits for num_qubits = {num_qubits}")
            else:
                print(f"************\nCreating [{max_circuits}] circuits for num_qubits = {num_qubits}")
                all_qcs[str(num_qubits)] = {}
        else:
            print(f"************\nExecuting [{max_circuits}] restarts for num_qubits = {num_qubits}")

        if degree < 0:
            degree = max(3, (num_qubits + degree))

        instance_filename = os.path.join(
            os.path.dirname(__file__),
            "..",
            "_common",
            common.INSTANCE_DIR,
            f"mc_{num_qubits:03d}_{degree:03d}_000.txt",
        )
        nodes, edges = common.read_maxcut_instance(instance_filename, _instances)
        opt, _ = common.read_maxcut_solution(
            instance_filename[:-4] + ".sol", _instances
        )

        if nodes is None:
            print("  ... problem not found.")
            break

        for restart_ind in range(1, max_circuits + 1):
            if not use_fixed_angles:
                thetas_array = thetas[restart_ind - 1]

            if method == 1:
                ts = time.time()
                thetas_array_0 = thetas_array
                if use_fixed_angles:
                    thetas_array_0 = thetas_array[0]

                qc, params = MaxCut(
                    num_qubits, restart_ind, edges, rounds, thetas_array_0,
                    parameterized,
                )
                metrics.store_metric(num_qubits, restart_ind, "create_time",
                                     time.time() - ts)

                if get_circuits:
                    all_qcs[str(num_qubits)][str(restart_ind)] = qc
                    continue

                ex.submit_circuit(
                    qc, num_qubits, restart_ind, shots=num_shots,
                    params=params,
                )

            if method == 2:
                minimizer_loop_index = 0

                def expectation(thetas_array):
                    global minimizer_loop_index
                    global iter_size_dist
                    global iter_dist
                    global opt_ts
                    global saved_result

                    unique_id = restart_ind * 1000 + minimizer_loop_index
                    metrics.store_metric(
                        num_qubits, unique_id, "thetas_array",
                        _theta_list(thetas_array),
                    )

                    ts = time.time()
                    qc, params = MaxCut(
                        num_qubits, unique_id, edges, rounds, thetas_array,
                        parameterized,
                    )
                    metrics.store_metric(num_qubits, unique_id, "create_time",
                                         time.time() - ts)
                    metrics.store_metric(num_qubits, unique_id, "rounds", rounds)
                    metrics.store_metric(num_qubits, unique_id, "degree", degree)

                    ex.submit_circuit(
                        qc, num_qubits, unique_id, shots=num_shots,
                        params=params,
                    )
                    ex.finalize_execution(None, report_end=False)

                    objective_value = None
                    leader_error = None
                    if mpi.leader():
                        # Catch any exception so we still reach the bcast below;
                        # otherwise non-leader ranks would deadlock waiting.
                        try:
                            if do_compute_expectation:
                                _, fidelity = analyze_and_print_result(
                                    qc, saved_result, num_qubits, num_shots,
                                    secret_int=unique_id,
                                )
                                metrics.store_metric(num_qubits, unique_id, "fidelity", fidelity)

                            dict_of_vals = dict()
                            tc1 = time.time()
                            cuts, counts, sizes = compute_cutsizes(saved_result, nodes, edges)
                            dict_of_vals[objective_func_type] = function_mapper[
                                objective_func_type
                            ](counts, sizes, alpha=alpha)
                            metrics.store_metric(
                                num_qubits, unique_id, "opt_exec_time",
                                time.time() - tc1 + ts - opt_ts,
                            )

                            unique_counts, unique_sizes, cumul_counts = get_size_dist(
                                counts, sizes
                            )
                            iter_size_dist = {
                                "unique_sizes": unique_sizes,
                                "unique_counts": unique_counts,
                                "cumul_counts": cumul_counts,
                            }
                            metrics.store_metric(num_qubits, unique_id, None, iter_size_dist)

                            for score in non_objFunc_ratios:
                                dict_of_vals[score] = function_mapper[score](
                                    counts, sizes, alpha=alpha
                                )

                            dict_of_ratios = {
                                key: -1 * val / opt for key, val in dict_of_vals.items()
                            }
                            dict_of_ratios["gibbs_ratio"] = dict_of_ratios["gibbs_ratio"] / eta
                            metrics.store_metric(num_qubits, unique_id, None, dict_of_ratios)

                            best = -compute_best_cut_from_measured(counts, sizes)
                            metrics.store_metric(
                                num_qubits, unique_id, "bestcut_ratio", best / opt
                            )

                            quantile_sizes = compute_quartiles(counts, sizes)
                            metrics.store_metric(
                                num_qubits, unique_id, "quantile_optgaps",
                                (1 - quantile_sizes / opt).tolist(),
                            )

                            iter_dist = {"cuts": cuts, "counts": counts, "sizes": sizes}
                            objective_value = dict_of_vals[objective_func_type]
                        except Exception as exc:
                            leader_error = repr(exc)
                    minimizer_loop_index += 1

                    if comfort:
                        if minimizer_loop_index == 1:
                            print("")
                        print(".", end="")

                    opt_ts = time.time()
                    # MPI: return the rank-0 objective to every optimizer
                    # instance; surface leader exceptions on every rank.
                    objective_value, leader_error = mpi.bcast(
                        (objective_value, leader_error)
                    )
                    if leader_error is not None:
                        raise RuntimeError(
                            f"MaxCut COBYLA expectation failed on rank 0: {leader_error}"
                        )
                    return objective_value

                opt_ts = time.time()
                thetas_array_0 = thetas_array
                if use_fixed_angles:
                    thetas_array_0 = thetas_array[0]

                res = minimize(
                    expectation, thetas_array_0, method="COBYLA",
                    options={"maxiter": max_iter},
                )

                unique_id = restart_ind * 1000
                metrics.store_metric(
                    num_qubits, unique_id, "opt_exec_time", time.time() - opt_ts
                )

                if comfort:
                    print("")

                store_final_iter_to_metrics_json(
                    num_qubits=num_qubits,
                    degree=degree,
                    restart_ind=restart_ind,
                    num_shots=num_shots,
                    converged_thetas_list=res.x.tolist(),
                    opt=opt,
                    iter_size_dist=iter_size_dist,
                    iter_dist=iter_dist,
                    parent_folder_save=parent_folder_save,
                    dict_of_inputs=dict_of_inputs,
                    save_final_counts=save_final_counts,
                    save_res_to_file=save_res_to_file,
                    _instances=_instances,
                )

        if method == 1 and not get_circuits:
            ex.throttle_execution(metrics.finalize_group)

        if method == 2:
            metrics.process_circuit_metrics_2_level(num_qubits)
            metrics.finalize_group(str(num_qubits))

    if get_circuits:
        print("************\nReturning circuits and circuit information")
        return all_qcs, metrics.circuit_metrics

    if method == 1:
        ex.finalize_execution(None)
    else:
        ex.finalize_execution(metrics.finalize_group)

    if draw_circuits and print_sample_circuit:
        print("Sample Circuit:")
        if QC_ is None:
            print("  ... too large!")
        else:
            print(cudaq.draw(QC_[0], *QC_[1]))

    if method == 1:
        if plot_results:
            metrics.plot_metrics(
                f"Benchmark Results - {benchmark_name} ({method}) - CUDA-Q",
                options=dict(shots=num_shots, rounds=rounds),
            )
    elif plot_results:
        plot_results_from_data(**dict_of_inputs)


def plot_results_from_data(num_shots=100, rounds=1, degree=3, max_iter=30,
                           max_circuits=1, objective_func_type="approx_ratio",
                           method=2, use_fixed_angles=False,
                           score_metric="fidelity",
                           x_metric="cumulative_exec_time",
                           y_metric="num_qubits", fixed_metrics={},
                           num_x_bins=15, y_size=None, x_size=None,
                           x_min=None, x_max=None, offset_flag=False,
                           detailed_save_names=False, **kwargs):
    if detailed_save_names:
        cur_time = datetime.datetime.now()
        dt = cur_time.strftime("%Y-%m-%d_%H-%M-%S")
        short_obj_func_str = metrics.score_label_save_str[objective_func_type]
        suffix = f"-s{num_shots}_r{rounds}_d{degree}_mi{max_iter}_of-{short_obj_func_str}_{dt}"
    else:
        short_obj_func_str = metrics.score_label_save_str[objective_func_type]
        suffix = f"of-{short_obj_func_str}"

    obj_str = metrics.known_score_labels[objective_func_type]
    options = {
        "shots": num_shots,
        "rounds": rounds,
        "degree": degree,
        "restarts": max_circuits,
        "fixed_angles": use_fixed_angles,
        "\nObjective Function": obj_str,
    }
    suptitle = f"Benchmark Results - MaxCut ({method}) - CUDA-Q"

    metrics.plot_all_area_metrics(
        suptitle,
        score_metric=score_metric,
        x_metric=x_metric,
        y_metric=y_metric,
        fixed_metrics=fixed_metrics,
        num_x_bins=num_x_bins,
        x_size=x_size,
        y_size=y_size,
        x_min=x_min,
        x_max=x_max,
        offset_flag=offset_flag,
        options=options,
        suffix=suffix,
    )

    metrics.plot_metrics_optgaps(
        suptitle, options=options, suffix=suffix,
        objective_func_type=objective_func_type,
    )

    all_widths = [int(width) for width in metrics.circuit_metrics_final_iter.keys()]
    if all_widths:
        metrics.plot_cutsize_distribution(
            suptitle=suptitle,
            options=options,
            suffix=suffix,
            list_of_widths=[all_widths[-1]],
        )

    metrics.plot_angles_polar(suptitle=suptitle, options=options, suffix=suffix)


def load_data_and_plot(folder=None, backend_id=None, **kwargs):
    print("CUDA-Q MaxCut load_data_and_plot is not implemented.")


if __name__ == "__main__":
    print("Please run this benchmark from the parent directory:")
    print("  python maxcut/maxcut_benchmark.py")
