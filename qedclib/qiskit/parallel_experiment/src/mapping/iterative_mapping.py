# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (12/2020)
# Contributor: Adrien Suau (<adrien.suau@cerfacs.fr>
#                           <adrien.suau@lirmm.fr>)
#               Siyuan Niu (<siyuan.niu@lirmm.fr>)
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

import typing as ty
import logging
import numpy
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit, Clbit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.dag_to_circuit import dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGNode
from hardware.partition_distance_matrix import partition_distance_matrix, adj_matrix_construct
from mapping.swap_heuristics import swap_heuristic

logger = logging.getLogger("iterative_mapping")

import copy
from hardware.distance_matrx import (
    get_distance_matrix_swap_number_and_error,
)
from mapping.gates import (
    TwoQubitGate,
    SwapTwoQubitGate,
)
from hardware.IBMQHardwareArchitecture import (
    IBMQHardwareArchitecture,
)
from mapping.layer import QuantumLayer, update_layer, second_layer_construct, is_operation_critical
from mapping.swap import (
    get_all_swap_bridge_candidates,
)
from mapping._mapping_to_str import mapping_to_str_multiple

def _create_empty_dagcircuit_from_existing(dagcircuits: ty.List[DAGCircuit]) -> DAGCircuit:
    result = DAGCircuit()
    for qreg in dagcircuits[0].qregs.values():
        result.add_qreg(qreg)

    return result

def _modify_dag_circuit(circuit: QuantumCircuit, previous_qubits_used: int):
    from qiskit.circuit import QuantumRegister, ClassicalRegister
    n_qubits = len(circuit.qubits)
    n_clbits = len(circuit.clbits)
    new_qreg = QuantumRegister(previous_qubits_used + n_qubits, 'q')
    new_dag = DAGCircuit()
    new_dag.add_qreg(new_qreg)
    if n_clbits > 0:
        new_creg = ClassicalRegister(n_clbits, 'c')
        new_dag.add_creg(new_creg)
    orig_dag = circuit_to_dag(circuit)
    for node in orig_dag.topological_op_nodes():
        new_dag.apply_operation_back(
            node.op,
            qargs=[new_qreg[orig_dag.find_bit(q).index + previous_qubits_used] for q in node.qargs],
            cargs=[new_dag.cregs['c'][orig_dag.find_bit(c).index] for c in node.cargs] if node.cargs else [],
        )
    return new_dag

def iterative_mapping_algorithm(
    quantum_circuits: ty.List[QuantumCircuit],
    hardware: IBMQHardwareArchitecture,
    initial_mapping: ty.Dict[Qubit, int],
    allocated_partition: ty.List[ty.List[int]],
    swap_cost_heuristic: ty.Callable[
    [
        IBMQHardwareArchitecture,  # Hardware information
        QuantumLayer,  # Current front layer
        ty.List[DAGNode],  # Topologically sorted list of nodes
        int,  # Index of the first non-processed gate.
        ty.Dict[Qubit, int],  # The mapping before applying the tested SWAP/Bridge
        ty.Dict[Qubit, int], # The initial mapping
        numpy.ndarray,  # The distance matrix between each qubits
        TwoQubitGate,  # The SWAP/Bridge we want to rank
    ],
    float,
    ] = swap_heuristic,
    get_candidates: ty.Callable[
        [QuantumLayer, IBMQHardwareArchitecture, ty.List, ty.Dict[Qubit, int], ty.Set[str],],
        ty.List[TwoQubitGate],
    ] = get_all_swap_bridge_candidates,
    get_distance_matrix: ty.Callable[
        [int, numpy.ndarray], numpy.ndarray
    ] = partition_distance_matrix,
) -> ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]]:
    """Map the given quantum circuit to the hardware topology provided.

    :param quantum_circuit: the quantum circuit to map.
    :param initial_mapping: the initial mapping used to start the iterative mapping
        algorithm.
    :param hardware: hardware data such as connectivity, gate time, gate errors, ...
    :param swap_cost_heuristic: the heuristic cost function that will estimate the cost
        of a given SWAP/Bridge according to the current state of the circuit.
    :param get_candidates: a function that takes as input the current front
        layer and the hardware description and that returns a list of tuples
        representing the SWAP/Bridge that should be considered by the heuristic.
    :param get_distance_matrix: a function that takes as first (and only) parameter the
        hardware representation and that outputs a numpy array containing the cost of
        performing a SWAP between each pair of qubits.
    :return: The final circuit along with the mapping obtained at the end of the
        iterative procedure.
    """
    # Creating the internal data structures that will be used in this function.
    dag_circuits = [circuit_to_dag(quantum_circuit) for quantum_circuit in quantum_circuits]
    # Sum of gate number of all quantum circuits
    all_gate_number = sum([len(quantum_circuit) for quantum_circuit in quantum_circuits])
    distance_matrix = []

    merged_allocated_partition = []
    if len(dag_circuits) == 1 and not isinstance(allocated_partition[0], list):
        allocated_partition = [allocated_partition]
    for i in range(len(dag_circuits)):
        distance_matrix.append(get_distance_matrix(hardware.qubit_number,
                                               adj_matrix_construct(
                                                   hardware,
                                                   get_distance_matrix_swap_number_and_error,
                                                   allocated_partition[i],
                                               )))

        merged_allocated_partition.extend(allocated_partition[i])

    resulting_dag_quantum_circuit = _create_empty_dagcircuit_from_existing(dag_circuits)
    current_mapping = initial_mapping
    explored_mappings: ty.List[ty.Set[str]] = [set() for _ in range(len(dag_circuits))]
    # Sorting all the quantum operations in topological order once for all.
    # May require significant memory on large circuits...
    topological_nodes: ty.List[ty.List[DAGNode]] = [list(dag_circuit.topological_op_nodes()) for dag_circuit in dag_circuits]
    current_node_indexs = [0 for _ in range(len(dag_circuits))]
    # Creating the initial front layer.
    front_layers = []
    for index, dag_circuit in enumerate(dag_circuits):
        front_layer = QuantumLayer()
        current_node_indexs[index] = update_layer(
            front_layer, topological_nodes[index], current_node_indexs[index]
        )
        front_layers.append(front_layer)

    trans_mapping = initial_mapping.copy()

    # Start of the iterative algorithm
    while all_gate_number:
        swap_candidate_dict = dict()
        original_front_layers = copy.deepcopy(front_layers)
        for index, front_layer in enumerate(front_layers):
            execute_gate_list = QuantumLayer()
            for op in front_layer.ops:
                if hardware.can_natively_execute_operation(op, current_mapping):
                    execute_gate_list.add_operation(op)
                # Delaying the remove operation because we do not want to remove from
                # a container we are iterating on.
            if not execute_gate_list.is_empty():
                front_layer.remove_operations_from_layer(execute_gate_list)
                execute_gate_list.apply_back_to_dag_circuit(
                    resulting_dag_quantum_circuit, initial_mapping, trans_mapping
                )
                # Empty the explored mappings because at least one gate has been executed.
                [explored_mapping.clear() for explored_mapping in explored_mappings]
                all_gate_number -= len(execute_gate_list)
                #print(all_gate_number)
                original_front_layers = copy.deepcopy(front_layers)
            elif not front_layer.ops:
                continue
            else:
                second_layer = second_layer_construct(topological_nodes[index], current_node_indexs[index])
                gate_to_resolve = []
                for op in front_layer.ops:
                    if is_operation_critical(op, second_layer) == True:
                        gate_to_resolve.append(op)
                if not gate_to_resolve:
                    for op in front_layer.ops:
                        gate_to_resolve.append(op)

                swap_candidate_list = get_candidates(
                    gate_to_resolve, hardware, allocated_partition[index], current_mapping, initial_mapping, trans_mapping, explored_mappings[index], merged_allocated_partition
                )
                swap_candidate_dict[index] = swap_candidate_list

        if swap_candidate_dict:

            inverse_mapping = {val: key for key, val in initial_mapping.items()}
            for index, swap_list in swap_candidate_dict.items():
                best_swap_qubits = None
                best_cost = float("inf")
                for potential_swap in swap_list:
                    cost = swap_cost_heuristic(
                        hardware,
                        original_front_layers[index],
                        topological_nodes[index],
                        current_node_indexs[index],
                        current_mapping,
                        initial_mapping,
                        trans_mapping,
                        distance_matrix[index],
                        potential_swap,
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_swap_qubits = potential_swap

                if not best_swap_qubits:
                    logger.error(f"No swap gate selected'.")
                    exit(1)

                current_mapping = best_swap_qubits.update_mapping(current_mapping)
                if isinstance(best_swap_qubits, SwapTwoQubitGate):
                    control, target = current_mapping[best_swap_qubits.left], current_mapping[best_swap_qubits.right]
                    swap_control, swap_target = inverse_mapping[control], inverse_mapping[target]
                    best_swap_qubits = SwapTwoQubitGate(
                        swap_control, swap_target
                    )

                    trans_mapping[best_swap_qubits.left], trans_mapping[best_swap_qubits.right] = (
                        trans_mapping[best_swap_qubits.right],
                        trans_mapping[best_swap_qubits.left],
                    )
                    #print(f"best swap qubits is ({best_swap_qubits.left},{best_swap_qubits.right})")
                else:
                    # print(
                    #     f"best bridge qubits is ({best_swap_qubits.left},{best_swap_qubits.middle},{best_swap_qubits.right})")
                    all_gate_number -= 1
                explored_mappings[index].add(mapping_to_str_multiple(current_mapping))
                best_swap_qubits.apply(resulting_dag_quantum_circuit, front_layers[index], initial_mapping,
                                       trans_mapping)

        for index, front_layer in enumerate(front_layers):
            current_node_indexs[index] = update_layer(
                front_layer, topological_nodes[index], current_node_indexs[index]
            )

    # We are done here, we just need to return the results
    # resulting_dag_quantum_circuit.draw(scale=1, filename="qcirc.dot")
    resulting_circuit = dag_to_circuit(resulting_dag_quantum_circuit)
    # print("result mapping:", current_mapping)
    return resulting_circuit, current_mapping




