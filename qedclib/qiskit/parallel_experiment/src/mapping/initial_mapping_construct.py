# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (02/2020)
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

import numpy
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag
import logging
from hardware.IBMQHardwareArchitecture import (
    IBMQHardwareArchitecture,
)

logger = logging.getLogger("initial_mapping_contruct")


def final_complete_mapping_construct(
                           circuit: QuantumCircuit,
                           mapping: ty.Dict[Qubit, int]
                           ) -> ty.Dict[Qubit, int]:
    final_complete_mapping = dict()
    left_physical_qubit_list = []
    for i in range(len(circuit.qubits)):
        if i not in mapping.values():
            left_physical_qubit_list.append(i)

    j = 0
    for i, qubit in enumerate(circuit.qubits):
        if qubit in mapping.keys():
            continue
        else:
            #final_complete_mapping[Qubit(QuantumRegister(size, f"q_Q{index}"), qubit.index)] = left_physical_qubit_list[j]
            final_complete_mapping[qubit] = left_physical_qubit_list[j]
            j += 1
    mapping.update(final_complete_mapping)

    return mapping


def get_random_mapping(quantum_circuit: QuantumCircuit) -> ty.Dict[Qubit, int]:
    size = len(quantum_circuit.qubits)
    random_sampling = numpy.random.permutation(size)
    #return  {Qubit(QuantumRegister(size, f"q_Q{index}"), qubit.index): random_sampling[i] for i, qubit in enumerate(quantum_circuit.qubits)}
    return {qubit: random_sampling[i] for i, qubit in enumerate(quantum_circuit.qubits)}


def multiple_programming_get_random_mapping(quantum_circuit: QuantumCircuit,
                                            partition: ty.List,
                                            circuit_initial_mapping: ty.Dict[Qubit, int],
                                            ) -> ty.Dict[Qubit, int]:
    size = len(quantum_circuit.qubits)
    dag = circuit_to_dag(quantum_circuit)
    qubits_non_idle = [qubit for qubit in quantum_circuit.qubits if qubit not in dag.idle_wires()]
    partition_qubit_number_diff = len(partition) - len(qubits_non_idle)
    while partition_qubit_number_diff > 0:
        for qubit_selected in quantum_circuit.qubits:
            if qubit_selected not in qubits_non_idle and qubit_selected not in circuit_initial_mapping.keys():
                qubits_non_idle.append(qubit_selected)
                partition_qubit_number_diff -= 1
                break

    random_sampling = numpy.random.permutation(len(qubits_non_idle))
    initial_mapping = {qubit: partition[random_sampling[i]] for i, qubit in enumerate(qubits_non_idle)}
    final_complete_mapping_construct(quantum_circuit, initial_mapping)
    return initial_mapping

def initial_mapping_from_sabre(
    quantum_circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    partition : ty.List,
    mapping_algorithm: ty.Callable[
        [QuantumCircuit, IBMQHardwareArchitecture, ty.Dict[Qubit, int], ty.List, int],
        ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]],
    ],
    circuit_initial_mapping: ty.Dict[Qubit, int],
    initial_mapping: ty.Optional[ty.Dict[Qubit, int]] = None,

) -> ty.Dict[Qubit, int]:
    # First make sure that the quantum circuit has the same number of quantum bits as
    # the hardware. (remove!)
    #quantum_circuit = add_qubits_to_quantum_circuit(quantum_circuit, hardware)
    reversed_quantum_circuit = quantum_circuit.inverse()
    # Generate a random initial mapping
    if initial_mapping is None:
        # initial mapping for independent circuit
        if partition == None or len(partition) == hardware.qubit_number:
            initial_mapping = get_random_mapping(quantum_circuit)
        #initial mapping for merged circuit
        else:
            initial_mapping = multiple_programming_get_random_mapping(quantum_circuit, partition, circuit_initial_mapping)
    # qubit_initial_mapping = [5,4,10,1,8,11,0,14,12,7,6,13,3,2,9]
    # initial_mapping = dict()
    # for index, qubit in enumerate(quantum_circuit.qubits):
    #     initial_mapping[qubit] = qubit_initial_mapping[index]

    #print("initial mapping is :", initial_mapping)
    # Performing the forward step
    if partition == None:
        forward_circuit, reversed_mapping = mapping_algorithm(
            quantum_circuit, hardware, initial_mapping,
        )
        # print("reversed mapping is:", reversed_mapping)
        # And the backward step
        _, final_mapping = mapping_algorithm(
            reversed_quantum_circuit, hardware, reversed_mapping,
        )
    else:
        forward_circuit, reversed_mapping = mapping_algorithm(
            [quantum_circuit], hardware, initial_mapping, partition
        )
        #print("reversed mapping is:", reversed_mapping)
        # And the backward step
        _, final_mapping = mapping_algorithm(
            [reversed_quantum_circuit], hardware, reversed_mapping, partition,
        )
    #print("final mapping is:", final_mapping)
    return final_mapping

def get_best_mapping_sabre(
    circuit: QuantumCircuit,
    partition : ty.List,
    mapping_algorithm: ty.Callable[
        [QuantumCircuit, IBMQHardwareArchitecture, ty.Dict[Qubit, int], ty.List],
        ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]],
    ],
    cost_function: ty.Callable[
        [ty.Dict[Qubit, int], QuantumCircuit, IBMQHardwareArchitecture, ty.List, ], float
    ],
    hardware: IBMQHardwareArchitecture,
    maximum_allowed_evaluations: int,
    circuit_initial_mapping: ty.Dict[Qubit, int],
) -> ty.Dict[Qubit, int]:
    if maximum_allowed_evaluations < 2:
        print("Not enough allowed evaluations!")
        exit(1)
    best_mapping = initial_mapping_from_sabre(circuit, hardware, partition, mapping_algorithm, circuit_initial_mapping)
    #print("best mapping is:", best_mapping)
    best_cost = cost_function(best_mapping, circuit, hardware, partition, mapping_algorithm)
    for i in range(maximum_allowed_evaluations // 2 - 1):
        mapping = initial_mapping_from_sabre(circuit, hardware, partition, mapping_algorithm, circuit_initial_mapping)
        #print("mapping is:", mapping)
        cost = cost_function(mapping, circuit, hardware, partition, mapping_algorithm)
        if cost < best_cost:
            best_mapping, best_cost = mapping, cost
    return best_mapping

def cost(initial_mapping,
         quantum_circuit: QuantumCircuit,
         hardware: IBMQHardwareArchitecture,
         partition: ty.List,
         mapping_algorithm: ty.Callable[
             [QuantumCircuit, IBMQHardwareArchitecture, ty.Dict[Qubit, int], ty.List],
             ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]],
         ],
):
    if partition == None:
        mapped_circuit, final_mapping = mapping_algorithm(
            quantum_circuit, hardware, initial_mapping,
        )
    else:
        mapped_circuit, final_mapping = mapping_algorithm(
            [quantum_circuit], hardware, initial_mapping, partition
        )
    cx_num = mapped_circuit.count_ops().get("cx", 0)
    swap_num = mapped_circuit.count_ops().get("swap", 0) * 3
    ops_num = (
            cx_num + swap_num
    )
    #print(f"cx is {cx_num} + swap_num {swap_num} is {ops_num}")
    return ops_num