# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (03/2020)
# Contributor: Adrien Suau (<adrien.suau@cerfacs.fr>
#                          <adrien.suau@lirmm.fr>)
#              Siyuan Niu  (<siyuan.niu@lirmm.fr>)
# This software is governed by the CeCILL-B license under French law and
# abiding  by the  rules of  distribution of free software. You can use,
# modify  and/or  redistribute  the  software  under  the  terms  of the
# CeCILL-B license as circulated by CEA, CNRS and INRIA at the following
# URL "http://www.cecill.info".
#
# As a counterpart to the access to  the source code and rights to copy,
# modify and  redistribute granted  by the  license, users  are provided
# only with a limited warranty and  the software's author, the holder of
# the economic rights,  and the  successive licensors  have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using, modifying and/or  developing or reproducing  the
# software by the user in light of its specific status of free software,
# that  may mean  that it  is complicated  to manipulate,  and that also
# therefore  means that  it is reserved for  developers and  experienced
# professionals having in-depth  computer knowledge. Users are therefore
# encouraged  to load and  test  the software's  suitability as  regards
# their  requirements  in  conditions  enabling  the  security  of their
# systems  and/or  data to be  ensured and,  more generally,  to use and
# operate it in the same conditions as regards security.
#
# The fact that you  are presently reading this  means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.
# ======================================================================

import argparse
import itertools
import pickle
import sys
import typing as ty
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from multiprocessing import cpu_count

import numpy
from numpy.random import permutation

from hamap._circuit_manipulation import add_qubits_to_quantum_circuit
from hamap.hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
from hamap.initial_mapping import (
    get_best_mapping_from_annealing,
    get_best_mapping_from_iterative_forward_backward,
    get_best_mapping_sabre,
    get_best_mapping_random,
    _random_execution_policy,
    _hardware_aware_expand,
    _hardware_aware_reset,
    _random_shuffle,
)
from hamap.mapping import ha_mapping

sys.path.append(
    "/home/suau/Seafile/Qubit mapping problem bibliography/SABRE/sabre_dynamic_test/"
)

from sabre_distance_bridge import chips
import random
from sabre_distance_bridge import utils
from sabre_distance_bridge.mapping import (
    one_round_optimization,
    output_context,
)

from pathlib import Path
from qiskit import QuantumCircuit
import numpy as np

from sabre_distance_bridge.distance_matrix import (
    get_distance_matrix_swap_number,
    get_distance_matrix_error_cost,
    get_distance_matrix_execution_time_cost,
)

qx2 = chips.ibmqx2()
q20 = chips.ibmq20()
q_almaden = chips.ibmq20Almaden()
qx5 = chips.ibmq5()
q53 = chips.ibmq53()

chip = q20


def _seed_random():
    numpy.random.seed()
    random.seed()


def _argmin(l: ty.Iterable) -> int:
    return min(((v, i) for i, v in enumerate(l)), key=lambda tup: tup[0])[1]


def read_benchmark_circuit(category: str, name: str) -> QuantumCircuit:
    src_folder = Path(__file__).parent.parent.parent
    benchmark_folder = src_folder.parent / "benchmark"
    return QuantumCircuit.from_qasm_file(
        benchmark_folder / "circuits" / category / f"{name}.qasm"
    )


def separate_lists(iterable):
    ret1, ret2 = [], []
    for i, j in iterable:
        ret1.append(i)
        ret2.append(j)
    return ret1, ret2


def separate_lists_all_values_of_n(iterable):
    l = list(iterable)
    n_values_number = len(l[0][0])
    ret1 = [[] for _ in range(n_values_number)]
    ret2 = [[] for _ in range(n_values_number)]
    for elem1, elem2 in l:
        for k in range(n_values_number):
            ret1[k].append(elem1[k])
            ret2[k].append(elem2[k])
    return ret1, ret2


def print_statistics(result_type: str, results, timings):
    print(
        f"\t{result_type}:\n"
        f"\t\tAverage: {numpy.mean(results)}\n"
        f"\t\tMedian: {numpy.median(results)}\n"
        f"\t\tBest: {numpy.min(results)}\n"
        f"\t\tWorst: {numpy.max(results)}\n"
        f"\t\t25-50-75 percentiles: {numpy.percentile(results, [25,50,75])}\n"
        f"\t\t25-50-75 percentiles timing: {numpy.percentile(timings, [25,50,75])}"
    )


def mapping_algorithm(quantum_circuit: QuantumCircuit, hardware, initial_mapping):
    # 1. Compute distance matrices
    distance_matrix_error_cost = get_distance_matrix_error_cost(hardware)
    distance_matrix_execution_time_cost = get_distance_matrix_execution_time_cost(
        hardware
    )
    distance_matrix_swap_number = get_distance_matrix_swap_number(hardware)

    norm_error_cost = np.linalg.norm(distance_matrix_error_cost)
    norm_execution_time_cost = np.linalg.norm(distance_matrix_execution_time_cost)
    norm_swap_number_cost = np.linalg.norm(distance_matrix_swap_number)

    distance_matrix_error_cost_norm = distance_matrix_error_cost / norm_error_cost
    distance_matrix_execution_time_cost_norm = (
        distance_matrix_execution_time_cost / norm_execution_time_cost
    )
    distance_matrix_swap_number_norm = (
        distance_matrix_swap_number / norm_swap_number_cost
    )
    distance_mat = (
        0.5 * distance_matrix_error_cost_norm
        + 0 * distance_matrix_execution_time_cost_norm
        + 0.5 * distance_matrix_swap_number_norm
    )

    # 2. Write the quantum to a QASM file
    with open("filename.qasm", "w") as f:
        f.write(quantum_circuit.qasm())

    # 3. Re-open the QASM file for SABRE structures
    filename = "decod24-v2_43.qasm"
    (
        qubit_num,
        gate_type,
        gate_qubit,
        cx_gate_num,
        cx_gates,
        context,
    ) = utils.read_flatten_qasm(
        str(Path(__file__).parent.parent)
        + "/sabre_dynamic_test/sabre_distance_bridge/test/examples/"
        + filename
    )

    # 4. Apply the mapping algorithm
    # print('initial mapping ', initial_mapping)
    if isinstance(initial_mapping, dict):
        initial_mapping = [j for i, j in initial_mapping.items()]
    # print('initial mapping ', initial2_mapping)
    swap_num, bridge_num, final_mapping = one_round_optimization(
        initial_mapping,
        distance_mat,
        distance_matrix_swap_number,
        # get_distance_matrix_cnot_error_cost,
        gate_qubit,
        gate_type,
        context,
        chip.qubit_num,
        cx_gate_num,
        chip,
    )

    # Write the QASM from SABRE structure
    output = str(Path(__file__).parent) + "/output/1.qasm"
    with open(output, "w") as file_object:
        file_object.write("OPENQASM 2.0;\n")
        file_object.write('include "qelib1.inc";\n')
        file_object.write("qreg q[" + str(chip.qubit_num) + "];\n")
        file_object.write("creg c[" + str(chip.qubit_num) + "];\n")
        for line_idx in output_context:
            file_object.write(line_idx)

    # 5. QuantumCircuit from the QASM
    mapped_quantum_circuit = QuantumCircuit.from_qasm_file(output)
    mapped_quantum_circuit = QuantumCircuit.from_qasm_file(output)
    # print(mapped_quantum_circuit.count_ops())
    return mapped_quantum_circuit, final_mapping


def cost_function(mapping, circuit: QuantumCircuit, hardware: IBMQHardwareArchitecture):
    mapped_circuit, final_mapping = ha_mapping(circuit, mapping, hardware)
    count = mapped_circuit.count_ops()
    return 3 * count.get("swap", 0) + count.get("cnot", 0) + 4 * count.get("bridge", 0)


def cost(initial_mapping, quantum_circuit: QuantumCircuit, hardware):
    mapped_circuit, final_mapping = mapping_algorithm(
        quantum_circuit, hardware, initial_mapping
    )
    ops = mapped_circuit.count_ops()
    ops_num = ops.get("cx", 0) + ops.get("swap", 0) * 3 + ops.get("bridge", 0) * 4
    return ops_num


def random_tuple_strategy_results(tup):
    circuit, hardware, max_allowed_eval = tup
    mapping = get_best_mapping_random(
        circuit, cost_function, hardware, max_allowed_eval,
    )
    return cost_function(mapping, circuit, hardware)


def sabre_tuple_strategy_results(tup):
    circuit, hardware, max_allowed_eval = tup
    mapping = get_best_mapping_sabre(
        circuit, mapping_algorithm, cost_function, hardware, max_allowed_eval
    )
    return cost_function(mapping, circuit, hardware)


def iterated_tuple_strategy_results(tup):
    circuit, hardware, max_allowed_eval = tup
    mapping = get_best_mapping_from_iterative_forward_backward(
        circuit, hardware, mapping_algorithm, cost_function, max_allowed_eval
    )
    return cost_function(mapping, circuit, hardware)


def annealing_random_tuple_strategy_results(tup):
    circuit, hardware, max_allowed_eval = tup
    return get_best_mapping_from_annealing(
        circuit, hardware, cost_function, max_allowed_eval
    )


def annealing_hardware_aware_tuple_strategy_results(tup):
    circuit, hardware, max_allowed_eval = tup
    p1, p2 = 0.9, 0.08
    mapping = get_best_mapping_from_annealing(
        circuit,
        hardware,
        cost_function,
        max_allowed_eval,
        get_neighbour_func=_random_execution_policy(
            p1,
            p2,
            [_random_shuffle, _hardware_aware_expand, _hardware_aware_reset],
            hardware,
            circuit,
        ),
    )
    return cost_function(mapping, circuit, hardware)


def main():
    parser = argparse.ArgumentParser("Compare the annealing method to pure random.")

    parser.add_argument(
        "N",
        type=int,
        help="Number of allowed call to the mapping procedure. Should be strictly "
        "over 1 (i.e. 2 or more).",
    )
    parser.add_argument("M", type=int, help="Number of repetitions for statistics.")
    parser.add_argument(
        "Nstep", type=int, help="Steps used to increase N from step to N."
    )
    parser.add_argument(
        "circuit_name", type=str, help="Name of the quantum circuit to map."
    )
    parser.add_argument("hardware", type=str, help="Name of the hardware to consider.")

    args = parser.parse_args()

    N = args.N
    if N <= 1:
        raise RuntimeError("N should be 2 or more.")
    M = args.M
    Nstep = args.Nstep
    hardware = IBMQHardwareArchitecture.load(args.hardware)
    circuit = add_qubits_to_quantum_circuit(
        read_benchmark_circuit("sabre", args.circuit_name), hardware
    )

    results = dict()
    N_values = list(range(Nstep, N + 1, Nstep))

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for i, n in enumerate(N_values):
            print(f"Computing for {n}:")
            print("\tRandom...")
            best_random_results = list(
                executor.map(
                    random_tuple_strategy_results,
                    itertools.repeat([circuit, hardware, n], M),
                )
            )

            print("\tSABRE...")
            best_sabre_results = list(
                executor.map(
                    sabre_tuple_strategy_results,
                    itertools.repeat([circuit, hardware, n], M),
                )
            )

            print("\tAnnealing random...")
            best_annealing_random_results = list(
                executor.map(
                    annealing_random_tuple_strategy_results,
                    itertools.repeat([circuit, hardware, n], M),
                )
            )

            print("\tForward-backward...")
            best_forward_backward_results = list(
                executor.map(
                    iterated_tuple_strategy_results,
                    itertools.repeat([circuit, hardware, n], M),
                )
            )

            print("\tAnnealing hardware...")
            best_annealing_hardware_results = list(
                executor.map(
                    annealing_hardware_aware_tuple_strategy_results,
                    itertools.repeat([circuit, hardware, n], M),
                )
            )

            results[n] = {
                "random": {"results": deepcopy(best_random_results)},
                "annealing_random": {
                    "results": deepcopy(best_annealing_random_results)
                },
                "sabre": {"results": deepcopy(best_sabre_results)},
                "iterated": {"results": deepcopy(best_forward_backward_results)},
                "annealing_hardware": {
                    "results": deepcopy(best_annealing_hardware_results)
                },
            }

    print(f"Saving to results-{N}-{Nstep}-{M}-{args.circuit_name}-{args.hardware}.pkl")
    with open(
        f"results-{N}-{Nstep}-{M}-{args.circuit_name}-{args.hardware}.pkl", "wb"
    ) as f:
        pickle.dump(results, f)
