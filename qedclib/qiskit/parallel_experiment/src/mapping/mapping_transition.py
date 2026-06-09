# ======================================================================
# MIT License
#
# Copyright (c) [2020] [LIRMM]
# Contributor: Siyuan Niu (<siyuan.niu@lirmm.fr>)
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ======================================================================
from hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit import Qubit, Clbit, QuantumRegister, ClassicalRegister
from qiskit.dagcircuit import DAGCircuit
from mapping.iterative_mapping import iterative_mapping_algorithm
from mapping.initial_mapping_wrapper import initial_mapping
from mapping.initial_mapping_construct import cost
from HA.src.hamap import ha_mapping_paper_compliant
from partition_process.qubit_partition import (
    partition_hardware_heuristic,
    partition_circuits,
    largest_circuit_logical_degree,
    hardware_qubit_physical_degree,
)
import networkx as nx
from tools.submit2 import result_fidelity, energy_result
import logging
import typing as ty
import numpy as np
import time

logger = logging.getLogger("mapping_transisiton")

def _modify_dag_circuit(circuit: QuantumCircuit, previous_qubits_used: int,
                        shared_qreg: QuantumRegister) -> QuantumCircuit:
    """
    Re-index a circuit's active (non-idle) qubits into shared_qreg starting at
    previous_qubits_used. All circuits in a multiprogramming context share the same
    shared_qreg so their Qubit objects are compatible when building the merged DAG.
    """
    dag = circuit_to_dag(circuit)
    idle_set = set(dag.idle_wires())
    active_qubits = [q for q in circuit.qubits if q not in idle_set]
    n_active = len(active_qubits)
    n_clbits = len(circuit.clbits)

    new_circuit = QuantumCircuit(shared_qreg)
    if n_clbits > 0:
        new_creg = ClassicalRegister(n_clbits, 'c')
        new_circuit.add_register(new_creg)

    qubit_map = {active_qubits[i]: shared_qreg[i + previous_qubits_used]
                 for i in range(n_active)}
    clbit_map = {circuit.clbits[i]: new_circuit.clbits[i] for i in range(n_clbits)}

    for inst in circuit.data:
        new_circuit.append(
            inst.operation,
            [qubit_map[q] for q in inst.qubits],
            [clbit_map[c] for c in inst.clbits],
        )
    return new_circuit

def multiprogram_initial_mapping(
        circuits: ty.List[QuantumCircuit],
        mappings: ty.List[ty.Dict[Qubit, int]]
) -> ty.Dict[Qubit, int]:
    """
    Contruct a complete initial mapping for multiprogramming process
    """

    multi_initial_mapping = dict()
    partition_qubit_number_diff = len(circuits[0].qubits)
    for index, circuit in enumerate(circuits):
        dag = circuit_to_dag(circuit)
        qubits_non_idle = [qubit for qubit in circuit.qubits if qubit not in dag.idle_wires()]
        partition_qubit_number_diff -= len(qubits_non_idle)
        for qubit in qubits_non_idle:
            multi_initial_mapping[qubit] = mappings[index][qubit]

    left_physical_qubit_list = []

    for i in range(len(circuits[0].qubits)):
        if i not in multi_initial_mapping.values():
            left_physical_qubit_list.append(i)

    j = 0
    for i, qubit in enumerate(circuits[0].qubits):
        if qubit in multi_initial_mapping.keys():
            continue
        else:
            multi_initial_mapping[qubit] = left_physical_qubit_list[j]
            j += 1

    return multi_initial_mapping

def cost_gate_num(quantum_circuit: QuantumCircuit):
    cx_num = quantum_circuit.count_ops().get("cx", 0)
    swap_num = quantum_circuit.count_ops().get("swap", 0) * 3
    ops_num = (
        cx_num + swap_num
    )
    return ops_num

def multiprogram_mapping(circuits: ty.List[QuantumCircuit],
                         hardware: IBMQHardwareArchitecture,
                         circuit_partitions: ty.List,
                         ):
    """
    Perform the qubit mapping algorithm for the multiprogramming mechanism.
    Include initial mapping generation and mapping transition.
    """
    circuit_partitions = [list(part.value) for part in circuit_partitions]
    print(circuit_partitions)

    # Build a single shared register so every update circuit uses the same Qubit objects.
    # Each circuit contributes exactly len(circuit_partitions[i]) logical qubits.
    total_logical_qubits = sum(len(p) for p in circuit_partitions)
    shared_qreg = QuantumRegister(total_logical_qubits, 'q')

    # obtain the complete initial mapping of the merged circuit
    circuit_initial_mapping = dict()
    computed_initial_mappings = []
    update_circuits = []
    previous_qubit_used = 0

    num_cnots_circuits = sum([cost_gate_num(circuit) for circuit in circuits])

    for index, circuit in enumerate(circuits):
        circuit = _modify_dag_circuit(circuit, previous_qubit_used, shared_qreg)
        update_circuits.append(circuit)
        computed_initial_mapping = initial_mapping(
            circuit, hardware, circuit_partitions[index], iterative_mapping_algorithm, cost, "sabre", 10,
            circuit_initial_mapping,
        )
        computed_initial_mappings.append(computed_initial_mapping)
        previous_qubit_used += len(circuit_partitions[index])
    merge_final_mapping = multiprogram_initial_mapping(update_circuits, computed_initial_mappings)

    # the result circuit of the merged circuit
    merge_circuit, merged_final_mapping = iterative_mapping_algorithm(
        update_circuits,
        hardware,
        merge_final_mapping,
        circuit_partitions,
        )
    num_cnots_merge_circuit = cost_gate_num(merge_circuit)
    num_additional_cnots = num_cnots_merge_circuit - num_cnots_circuits

    print(f"additional cnots is {num_additional_cnots}")

    initial_layout = merge_final_mapping.values()

    return initial_layout, merge_circuit


def circuits_schedule(circuits: ty.List[QuantumCircuit],
                      hardware_graph,
                      hardware: IBMQHardwareArchitecture,
                      cnot_error_matrix: np.ndarray,
                      readout_error: ty.List,
                      partition_method: ty.Callable[[
                           IBMQHardwareArchitecture,
                           nx.DiGraph,
                           QuantumCircuit,
                           np.ndarray,
                           ty.List,
                           ty.Set,
                           ty.List,
                           int,
                           ty.Dict,
                       ], ty.List],
                      epslon,
                      weight_lambda,
                      ansatz_parameter:ty.List=None,
                      crosstalk_properties: ty.Dict=None):

    initial_layouts = []
    final_circuits = []
    partitions = []

    # Sort circuit according to ascending order of CNOT density
    if not ansatz_parameter:
        circuits = sorted(circuits, key=lambda x: x.count_ops().get("cx", 0) / x.cregs[0].size, reverse=True)

    # Pick up K circuits that are able to be executed on hardware at the same time
    # sum(n_i) <= N (qubit number of hardware), 1 <= i <= K
    circuit_list = []
    qubit_circuit_sum = 0
    for circuit in circuits:
        qubit_circuit_sum += circuit.cregs[0].size
        if qubit_circuit_sum <= hardware.qubit_number:
            circuit_list.append(circuit)
        else:
            break

    qubit_physical_degree, largest_physical_degree = hardware_qubit_physical_degree(hardware)

    largest_logical_degrees = []
    for circuit in circuit_list:
        largest_logical_degrees.append(largest_circuit_logical_degree(circuit))

    # Partition independently (PHA algorithm)
    partition_fidelity_independent_list = []
    independent_partitions = []
    for circuit in circuit_list:
        independent_partition = partition_circuits([circuit],
                                                   hardware_graph,
                                                   hardware,
                                                   cnot_error_matrix,
                                                   readout_error,
                                                   qubit_physical_degree,
                                                   largest_physical_degree,
                                                   largest_logical_degrees,
                                                   weight_lambda,
                                                   partition_method,
                                                   )
        partitions.append(independent_partition[0].value)
        partition_fidelity_independent_list.append(independent_partition[0].fidelity)
        independent_partitions.append(independent_partition)

    # If K > 1, circuits are executed on the hardware simultaneously (Parallelism metric), K = length of circuit list
    while len(circuit_list) > 1:
        # Partition simultaneously (multiprogramming)
        start = time.time()
        multiple_partition = partition_circuits(circuit_list,
                                                hardware_graph,
                                                hardware,
                                                cnot_error_matrix,
                                                readout_error,
                                                qubit_physical_degree,
                                                largest_physical_degree,
                                                largest_logical_degrees,
                                                weight_lambda,
                                                partition_method,
                                                crosstalk_properties,
                                                )
        #print(f"time is {time.time() - start}")

        if not multiple_partition:
            circ = circuit_list.pop()
            largest_logical_degrees.pop()
            partition_fidelity_independent_list.pop()
            partitions.pop()
            qubit_circuit_sum -= circ.cregs[0].size
            continue

        partition_fidelity_multiple = 0.0
        partition_fidelity_independent = sum(partition_fidelity_independent_list)
        for partition in multiple_partition:
            partition_fidelity_multiple += partition.fidelity


        # Post qubit partition process
        partition_fidelity_difference = abs(partition_fidelity_independent - partition_fidelity_multiple)
        #print("paritition fidelity difference is", partition_fidelity_difference)

        if partition_fidelity_difference < epslon:
            print(f"circuits that are executed simultaneously with threshold {epslon}, and the fidelity difference is {partition_fidelity_difference}.")
            for idx, circuit in enumerate(circuit_list):
                print(f"circuit name : {circuit.name}")
                print("Independent partition (PHA)")
                initial_layout, final_circuit = multiprogram_mapping([circuit], hardware, independent_partitions[idx])
                initial_layouts.append(initial_layout)
                final_circuits.append(final_circuit)

            print("Simultaneous partition")
            initial_layout, final_circuit = multiprogram_mapping(circuit_list, hardware, multiple_partition)
            initial_layouts.append(initial_layout)
            final_circuits.append(final_circuit)
            #print(final_circuit.qasm())
            partitions.append([partition.value for partition in multiple_partition])
            print(f"The number of circuits that are executed on hardware simultaneously is {len(circuit_list)}")
            break

        else:
            circ = circuit_list.pop()
            largest_logical_degrees.pop()
            partition_fidelity_independent_list.pop()
            partitions.pop()
            qubit_circuit_sum -= circ.cregs[0].size
            continue

    # If only one circuit can be executed on the hardware, all the circuits should be executed independently (using HA)
    if len(circuit_list) == 1:
        # HA mapping
        for circuit in circuits:
            circuit_initial_mapping_ha = dict()
            computed_initial_mapping = initial_mapping(
                circuit, hardware, None, ha_mapping_paper_compliant, cost, "sabre", 10, circuit_initial_mapping_ha
            )

            mapped_circuit, final_mapping = ha_mapping_paper_compliant(
                circuit, hardware, computed_initial_mapping,
            )
            num_cnots_circuits = cost_gate_num(circuit)
            num_cnots_merge_circuit = cost_gate_num(mapped_circuit)
            num_additional_cnots = num_cnots_merge_circuit - num_cnots_circuits
            print(f"additional cnots is {num_additional_cnots}")
            initial_layouts.append(computed_initial_mapping.values())
            final_circuits.append(mapped_circuit)
            partitions.append(None)

    #calculate the hardware throughput
    print("hardware throughput is:", qubit_circuit_sum / hardware.qubit_number)

    #quit()
    if ansatz_parameter is None:
        result_fidelity(hardware, initial_layouts, final_circuits, partitions)
    else:
        energy_result(hardware, initial_layouts, final_circuits, partitions, ansatz_parameter)

