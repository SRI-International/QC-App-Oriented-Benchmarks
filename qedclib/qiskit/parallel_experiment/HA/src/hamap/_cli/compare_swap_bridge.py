# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (04/2020)
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
import typing as ty
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from time import time as now

import numpy
from numpy.random import permutation
from qiskit import QuantumCircuit
from qiskit.circuit.quantumregister import Qubit

from hamap._circuit_manipulation import add_qubits_to_quantum_circuit
from hamap.distance_matrix import get_distance_matrix_swap_number
from hamap.hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
from hamap.heuristics import sabre_heuristic
from hamap.mapping import ha_mapping
from hamap.swap import (
    get_all_swap_candidates,
    get_all_swap_bridge_candidates,
)


def read_benchmark_circuit(category: str, name: str) -> QuantumCircuit:
    src_folder = Path(__file__).parent.parent.parent
    benchmark_folder = src_folder.parent / "benchmark"
    return QuantumCircuit.from_qasm_file(
        benchmark_folder / "circuits" / category / f"{name}.qasm"
    )


def using_only_swap_strategy(
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    mapping: ty.Dict[Qubit, int],
):
    start = now()
    modified_circuit, _ = ha_mapping(
        circuit,
        mapping,
        hardware,
        swap_cost_heuristic=sabre_heuristic,
        get_distance_matrix=get_distance_matrix_swap_number,
        get_candidates=get_all_swap_candidates,
    )
    duration = now() - start
    cnot_count = 3 * modified_circuit.count_ops().get(
        "swap", 0
    ) + modified_circuit.count_ops().get("cx", 0)
    # print(modified_circuit.draw("text"))
    return cnot_count, duration


def using_only_swap_strategy_tup(tup):
    return using_only_swap_strategy(*tup)


def using_swap_and_bridge_strategy(
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    mapping: ty.Dict[Qubit, int],
):
    start = now()
    modified_circuit, _ = ha_mapping(
        circuit,
        mapping,
        hardware,
        swap_cost_heuristic=sabre_heuristic,
        get_distance_matrix=get_distance_matrix_swap_number,
        get_candidates=get_all_swap_bridge_candidates,
    )
    duration = now() - start
    cnot_count = (
        3 * modified_circuit.count_ops().get("swap", 0)
        + 4 * modified_circuit.count_ops().get("bridge", 0)
        + modified_circuit.count_ops().get("cx", 0)
    )
    # print(modified_circuit.draw("text"))
    return cnot_count, duration


def using_swap_bridge_strategy_tup(tup):
    return using_swap_and_bridge_strategy(*tup)


def print_statistics(result_type: str, results, timings):
    print(
        f"{result_type}:\n"
        f"\tAverage: {numpy.mean(results)}\n"
        f"\tMedian: {numpy.median(results)}\n"
        f"\tBest: {numpy.min(results)}\n"
        f"\tWorst: {numpy.max(results)}\n"
        f"\t25-50-75 percentiles: {numpy.percentile(results, [25,50,75])}\n"
        f"\t25-50-75 percentiles timing: {numpy.percentile(timings, [25,50,75])}"
    )


def separate_lists(iterable):
    ret1, ret2 = [], []
    for i, j in iterable:
        ret1.append(i)
        ret2.append(j)
    return ret1, ret2


def main():
    parser = argparse.ArgumentParser(
        "Compare the Bridge+SWAP approach to the simple SWAP one."
    )

    parser.add_argument(
        "N",
        type=int,
        help="Number of initial mapping that will be explored. Should be strictly "
        "over 1 (i.e. 2 or more).",
    )
    parser.add_argument(
        "circuit_name", type=str, help="Name of the quantum circuit to map."
    )
    parser.add_argument("hardware", type=str, help="Name of the hardware to consider.")

    args = parser.parse_args()

    N = args.N
    if N < 1:
        raise RuntimeError("N should be 2 or more.")
    hardware = IBMQHardwareArchitecture.load(args.hardware)
    circuit = add_qubits_to_quantum_circuit(
        read_benchmark_circuit("sabre", args.circuit_name), hardware
    )

    initial_mappings = [
        {
            qubit: i
            for qubit, i in zip(
                circuit.qubits, permutation(range(hardware.qubit_number))
            )
        }
        for _ in range(N)
    ]

    # using_swap_bridge_strategy_tup([circuit, hardware, initial_mappings[0]])
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        best_swap_results, swap_timings = separate_lists(
            executor.map(
                using_only_swap_strategy_tup,
                zip(
                    itertools.repeat(circuit, N),
                    itertools.repeat(hardware, N),
                    initial_mappings,
                ),
            )
        )
        print_statistics("SWAP", best_swap_results, swap_timings)
        best_swap_bridge_results, swap_bridge_timings = separate_lists(
            executor.map(
                using_swap_bridge_strategy_tup,
                zip(
                    itertools.repeat(circuit, N),
                    itertools.repeat(hardware, N),
                    initial_mappings,
                ),
            )
        )
        print_statistics("SWAP+Bridge", best_swap_bridge_results, swap_bridge_timings)


if __name__ == "__main__":
    main()
