# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (03/2020)
# Contributor: Adrien Suau (<adrien.suau@cerfacs.fr>
#                           <adrien.suau@lirmm.fr>)
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
import random
import typing as ty
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from multiprocessing import cpu_count
from pathlib import Path
from time import time as now

import numpy
from numpy.random import permutation
from qiskit import QuantumCircuit

from hamap._circuit_manipulation import add_qubits_to_quantum_circuit
from hamap.hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
from hamap.initial_mapping import (
    get_initial_mapping_from_annealing,
    initial_mapping_from_sabre,
    initial_mapping_from_iterative_forward_backward,
    get_neighbour_random,
    get_random_mapping,
)
from hamap.mapping import ha_mapping


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


def random_strategy_results(
    allowed_calls_to_mapping: int,
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    allowed_calls_to_mapping_checkpoints: ty.List[int],
):
    _seed_random()
    best_random_cnot_number = float("inf")
    cnots_results = []
    times = []
    start = now()
    for i in range(allowed_calls_to_mapping):
        mapping = {
            qubit: i
            for qubit, i in zip(
                circuit.qubits, permutation(range(hardware.qubit_number))
            )
        }
        modified_circuit, _ = ha_mapping(circuit, mapping, hardware)
        op_count = modified_circuit.count_ops()
        best_random_cnot_number = min(
            best_random_cnot_number, 3 * op_count.get("swap", 0) + op_count.get("cx", 0)
        )
        if i + 1 in allowed_calls_to_mapping_checkpoints:
            cnots_results.append(best_random_cnot_number)
            times.append(now() - start)
    return cnots_results, times


def wrap_iterative_mapping_algorithm(
    quantum_circuit: QuantumCircuit, hardware: IBMQHardwareArchitecture, mapping,
):
    return ha_mapping(quantum_circuit, mapping, hardware)


def sabre_strategy_results(
    allowed_calls_to_mapping: int,
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    allowed_calls_to_mapping_checkpoints: ty.List[int],
):
    _seed_random()
    best_random_cnot_number = float("inf")
    cnots = []
    times = []
    start = now()
    for i in range(int(ceil(allowed_calls_to_mapping / 2))):
        initial_mapping = initial_mapping_from_sabre(
            circuit, hardware, wrap_iterative_mapping_algorithm
        )
        modified_circuit, _ = ha_mapping(circuit, initial_mapping, hardware)
        op_count = modified_circuit.count_ops()
        best_random_cnot_number = min(
            best_random_cnot_number, 3 * op_count.get("swap", 0) + op_count.get("cx", 0)
        )
        if (
            2 * (i + 1) in allowed_calls_to_mapping_checkpoints
            or 2 * i + 1 in allowed_calls_to_mapping_checkpoints
        ):
            cnots.append(best_random_cnot_number)
            times.append(now() - start)
    return cnots, times


def get_mapping_cost(mapping, quantum_circuit, hardware) -> float:
    mapped_quantum_circuit, _ = ha_mapping(quantum_circuit, mapping, hardware)
    return mapped_quantum_circuit.size() - quantum_circuit.size()


def annealing_strategy_results(
    allowed_calls_to_mapping: int,
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
):
    _seed_random()
    start = now()
    mapping, cost, iteration_number = get_initial_mapping_from_annealing(
        get_mapping_cost,
        circuit,
        hardware,
        max_steps=allowed_calls_to_mapping - 1,
        temp_begin=1.0,
        schedule_func=lambda t: 10 ** (-6 / (allowed_calls_to_mapping - 1)) * t,
    )
    modified_circuit, _ = ha_mapping(circuit, mapping, hardware)
    op_count = modified_circuit.count_ops()
    return 3 * op_count.get("swap", 0) + op_count.get("cx", 0), now() - start


def annealing_sabre_strategy_results(
    allowed_calls_to_mapping: int,
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
):
    _seed_random()
    start = now()
    mapping = initial_mapping_from_sabre(
        circuit, hardware, wrap_iterative_mapping_algorithm
    )
    # We want at least 2 steps in the annealing procedure.
    if allowed_calls_to_mapping > 4:
        n = allowed_calls_to_mapping - 3
        mapping, cost, iteration_number = get_initial_mapping_from_annealing(
            get_mapping_cost,
            circuit,
            hardware,
            max_steps=n,
            initial_mapping=mapping,
            temp_begin=1.0,
            schedule_func=lambda t: 10 ** (-6 / n) * t,
        )
    modified_circuit, _ = ha_mapping(circuit, mapping, hardware)
    op_count = modified_circuit.count_ops()
    return 3 * op_count.get("swap", 0) + op_count.get("cx", 0), now() - start


def forward_backward_strategy_results(
    allowed_calls_to_mapping: int,
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
):
    _seed_random()
    start = now()
    mapping_procedure_calls = 0
    best_cnot_number = float("inf")
    while (
        # We want to have at least 2 allowed calls to the mapping procedure.
        allowed_calls_to_mapping - mapping_procedure_calls > 1
        and best_cnot_number > 0
    ):
        mapping, nbcalls = initial_mapping_from_iterative_forward_backward(
            circuit,
            hardware,
            wrap_iterative_mapping_algorithm,
            maximum_mapping_procedure_calls=(
                allowed_calls_to_mapping - mapping_procedure_calls
            ),
        )
        mapping_procedure_calls += nbcalls
        modified_circuit, _ = ha_mapping(circuit, mapping, hardware)
        op_count = modified_circuit.count_ops()
        best_cnot_number = min(
            best_cnot_number, 3 * op_count.get("swap", 0) + op_count.get("cx", 0)
        )
    return best_cnot_number, now() - start


def forward_backward_annealing_strategy_results(
    allowed_calls_to_mapping: int,
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
):
    _seed_random()
    start = now()
    mapping, nbcals = initial_mapping_from_iterative_forward_backward(
        circuit,
        hardware,
        wrap_iterative_mapping_algorithm,
        maximum_mapping_procedure_calls=allowed_calls_to_mapping,
    )
    if allowed_calls_to_mapping - nbcals > 2:
        n = allowed_calls_to_mapping - nbcals
        mapping, cost, iteration_number = get_initial_mapping_from_annealing(
            get_mapping_cost,
            circuit,
            hardware,
            max_steps=n,
            initial_mapping=mapping,
            temp_begin=1.0,
            schedule_func=lambda t: 10 ** (-6 / n) * t,
        )
    modified_circuit, _ = ha_mapping(circuit, mapping, hardware)
    op_count = modified_circuit.count_ops()
    return 3 * op_count.get("swap", 0) + op_count.get("cx", 0), now() - start


def forward_backward_neighbour_strategy_results(
    allowed_calls_to_mapping: int,
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
):
    _seed_random()
    start = now()
    mapping = get_random_mapping(circuit)
    cnots = []
    while allowed_calls_to_mapping > 1:
        mapping, nbcalls = initial_mapping_from_iterative_forward_backward(
            circuit,
            hardware,
            wrap_iterative_mapping_algorithm,
            maximum_mapping_procedure_calls=allowed_calls_to_mapping,
            initial_mapping=mapping,
        )
        allowed_calls_to_mapping -= nbcalls
        modified_circuit, _ = ha_mapping(circuit, mapping, hardware)
        op_count = modified_circuit.count_ops()
        cnots.append(3 * op_count.get("swap", 0) + op_count.get("cx", 0))
        mapping = get_neighbour_random(mapping)

    return min(cnots), now() - start


def annealing_tuple_strategy_results(tup):
    return annealing_strategy_results(*tup)


def random_tuple_strategy_results(tup):
    return random_strategy_results(*tup)


def sabre_tuple_strategy_results(tup):
    return sabre_strategy_results(*tup)


def sabre_annealing_tuple_strategy_results(tup):
    return annealing_sabre_strategy_results(*tup)


def forward_backward_tuple_strategy_results(tup):
    return forward_backward_strategy_results(*tup)


def forward_backward_annealing_tuple_strategy_results(tup):
    return forward_backward_strategy_results(*tup)


def forward_backward_neighbour_tuple_strategy_results(tup):
    return forward_backward_neighbour_strategy_results(*tup)


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
        "Nstep", type=int, help="Steps used to increase N from step " "to N."
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

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        N_values = list(range(Nstep, N + 1, Nstep))
        print("Computing random...")
        best_random_results, random_timings = separate_lists_all_values_of_n(
            executor.map(
                random_tuple_strategy_results,
                itertools.repeat([N, circuit, hardware, N_values], M),
            )
        )
        print("Computing SABRE...")
        best_sabre_results, sabre_timings = separate_lists_all_values_of_n(
            executor.map(
                sabre_tuple_strategy_results,
                itertools.repeat([N, circuit, hardware, N_values], M),
            )
        )
        for i, n in enumerate(N_values):
            print(f"Computing for {n}:")
            print("\tannealing...")
            best_annealing_results, annealing_timings = separate_lists(
                executor.map(
                    annealing_tuple_strategy_results,
                    itertools.repeat([n, circuit, hardware], M),
                )
            )
            # print_statistics("Annealing", best_annealing_results, annealing_timings)
            print("\tSABRE-annealing...")
            best_sabre_annealing_results, sabre_annealing_timings = separate_lists(
                executor.map(
                    sabre_annealing_tuple_strategy_results,
                    itertools.repeat([n, circuit, hardware], M),
                )
            )
            # print_statistics(
            #     "SABRE + Annealing",
            #     best_sabre_annealing_results,
            #     sabre_annealing_timings,
            # )
            print("\tforward-backward...")
            (best_forward_backward_results, forward_backward_timings,) = separate_lists(
                executor.map(
                    forward_backward_tuple_strategy_results,
                    itertools.repeat([n, circuit, hardware], M),
                )
            )
            # print_statistics(
            #     "Forward-backward",
            #     best_forward_backward_results,
            #     forward_backward_timings,
            # )
            print("\tforward-backward annealing...")
            (
                best_forward_backward_annealing_results,
                forward_backward_annealing_timings,
            ) = separate_lists(
                executor.map(
                    forward_backward_tuple_strategy_results,
                    itertools.repeat([n, circuit, hardware], M),
                )
            )
            # print_statistics(
            #     "Forward-backward + Annealing",
            #     best_forward_backward_annealing_results,
            #     forward_backward_annealing_timings,
            # )
            print("\tforward-backward neighbour...")
            (
                best_forward_backward_neighbour_results,
                forward_backward_neighbour_timings,
            ) = separate_lists(
                executor.map(
                    forward_backward_neighbour_tuple_strategy_results,
                    itertools.repeat([n, circuit, hardware], M),
                )
            )
            results[n] = {
                "random": {
                    "results": best_random_results[i],
                    "times": random_timings[i],
                },
                "annealing": {
                    "results": best_annealing_results,
                    "times": annealing_timings,
                },
                "sabre": {"results": best_sabre_results[i], "times": sabre_timings[i]},
                "sabre_annealing": {
                    "results": best_sabre_annealing_results,
                    "times": sabre_annealing_timings,
                },
                "iterated": {
                    "results": best_forward_backward_results,
                    "times": forward_backward_timings,
                },
                "iterated_annealing": {
                    "results": best_forward_backward_annealing_results,
                    "times": forward_backward_annealing_timings,
                },
                "iterated_neighbour": {
                    "results": best_forward_backward_neighbour_results,
                    "times": forward_backward_neighbour_timings,
                },
            }

    with open(
        f"results-{N}-{Nstep}-{M}-{args.circuit_name}-{args.hardware}.pkl", "wb"
    ) as f:
        pickle.dump(results, f)
