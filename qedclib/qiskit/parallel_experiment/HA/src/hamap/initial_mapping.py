# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (02/2020)
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
import math
import random
import typing as ty
from copy import copy

import numpy
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag

from hamap._circuit_manipulation import add_qubits_to_quantum_circuit
from hamap.hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
from hamap.optimisation.simulated_annealing import simulated_annealing


def get_random_mapping(quantum_circuit: QuantumCircuit) -> ty.Dict[Qubit, int]:
    random_sampling = numpy.random.permutation(len(quantum_circuit.qubits))
    return {qubit: random_sampling[i] for i, qubit in enumerate(quantum_circuit.qubits)}


def _is_fixed_point(swap_numbers: ty.List[int]) -> bool:
    if len(swap_numbers) < 2:
        return False
    return swap_numbers[-1] == swap_numbers[-2]


def _argmin(l: ty.Iterable) -> int:
    return min(((v, i) for i, v in enumerate(l)), key=lambda tup: tup[0])[1]


def _count_swaps(circuit: QuantumCircuit) -> int:
    return circuit.count_ops().get("swap", 0)


def _count_cnots(circuit: QuantumCircuit) -> int:
    ops = circuit.count_ops()
    return 3 * ops.get("swap", 0) + ops.get("cx", 0) + 4 * ops.get("bridge", 0)


def initial_mapping_from_iterative_forward_backward(
    quantum_circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    mapping_algorithm: ty.Callable[
        [QuantumCircuit, IBMQHardwareArchitecture, ty.Dict[Qubit, int]],
        ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]],
    ],
    circuit_cost: ty.Callable[
        [ty.Dict[Qubit, int], QuantumCircuit, IBMQHardwareArchitecture], float
    ],
    initial_mapping: ty.Optional[ty.Dict[Qubit, int]] = None,
    maximum_mapping_procedure_calls: int = 20,
) -> ty.Tuple[ty.Dict[Qubit, int], float, int]:
    """Implementation of the initial_mapping method used by SABRE.

    :param quantum_circuit: the quantum circuit we want to find an initial mapping for
    :param hardware: the target hardware specifications
    :param mapping_algorithm: the algorithm used to map a quantum circuit to the
        given hardware with the given initial mapping.
    :param circuit_cost: a function computing the cost of a circuit. By default,
        the cost is the number of SWAP gates.
    :param initial_mapping: starting point of the algorithm. Default to a random guess.
    :param maximum_mapping_procedure_calls: the maximum number of calls to the
        mapping procedure. If the algorithm converged before, a lower number
        of evaluations will be performed.
    :return: the initial mapping, the cost of this mapping and the number of
        calls to the provided mapping procedure performed.
    """
    if maximum_mapping_procedure_calls < 2:
        raise RuntimeError(
            "You should do at least 1 iteration (2 calls to the mapping procedure)!"
        )
    # First make sure that the quantum circuit has the same number of quantum bits as
    # the hardware.
    quantum_circuit = add_qubits_to_quantum_circuit(quantum_circuit, hardware)
    reversed_quantum_circuit = quantum_circuit.inverse()
    # Generate a random initial mapping
    if initial_mapping is None:
        initial_mapping = get_random_mapping(quantum_circuit)

    # And improve this initial mapping according to an iterated method inspired from
    # SABRE.
    costs = list()
    mappings: ty.List[ty.Dict[Qubit, int]] = [initial_mapping]
    # We apply the forward-backward approach
    forward_mapping = initial_mapping
    for i in range(maximum_mapping_procedure_calls // 2):
        # Performing the forward step
        forward_circuit, reversed_mapping = mapping_algorithm(
            quantum_circuit, hardware, forward_mapping
        )
        # And the backward step
        _, forward_mapping = mapping_algorithm(
            reversed_quantum_circuit, hardware, reversed_mapping
        )
        # Adding the cost of the mapping to the list
        costs.append(circuit_cost(forward_mapping, quantum_circuit, hardware))
        # Adding the current mapping to the list in case we will do more iterations.
        mappings.append(forward_mapping)
        # If there is a repetition or we have a cost of 0, we can stop here.
        if costs[-1] == 0 or _is_fixed_point(costs):
            break

    current_calls_to_mapping_procedure = 2 * (i + 1)
    # If we finished the allowed number of iterations, return the best result.
    best_mapping_index = _argmin(costs)
    return (
        mappings[best_mapping_index],
        costs[best_mapping_index],
        current_calls_to_mapping_procedure + 1,
    )


def initial_mapping_from_sabre(
    quantum_circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    mapping_algorithm: ty.Callable[
        [QuantumCircuit, IBMQHardwareArchitecture, ty.Dict[Qubit, int]],
        ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]],
    ],
    initial_mapping: ty.Optional[ty.Dict[Qubit, int]] = None,
) -> ty.Dict[Qubit, int]:
    # First make sure that the quantum circuit has the same number of quantum bits as
    # the hardware.
    quantum_circuit = add_qubits_to_quantum_circuit(quantum_circuit, hardware)
    reversed_quantum_circuit = quantum_circuit.inverse()
    # Generate a random initial mapping
    if initial_mapping is None:
        initial_mapping = get_random_mapping(quantum_circuit)

    # Performing the forward step
    forward_circuit, reversed_mapping = mapping_algorithm(
        quantum_circuit, hardware, initial_mapping
    )
    # And the backward step
    _, final_mapping = mapping_algorithm(
        reversed_quantum_circuit, hardware, reversed_mapping
    )
    return final_mapping


def get_neighbour_random(mapping: ty.Dict[Qubit, int]) -> ty.Dict[Qubit, int]:
    inverse_mapping = {v: k for k, v in mapping.items()}
    a, b = random.choices(list(inverse_mapping.keys()), k=2)
    inverse_mapping[a], inverse_mapping[b] = inverse_mapping[b], inverse_mapping[a]
    return {k: v for v, k in inverse_mapping.items()}


NeighbourMappingAlgorithmType = ty.Callable[
    [ty.Dict[Qubit, int], QuantumCircuit, IBMQHardwareArchitecture], ty.Dict[Qubit, int]
]


def _random_execution_policy(
    p1: float,
    p2: float,
    algorithms: ty.List[NeighbourMappingAlgorithmType],
    hardware: IBMQHardwareArchitecture,
    circuit: QuantumCircuit,
) -> NeighbourMappingAlgorithmType:
    def ret(mapping: ty.Dict[Qubit, int]) -> ty.Dict[Qubit, int]:
        p = random.random()
        if p < p1:
            ret_value = algorithms[0](mapping, circuit, hardware)
        elif p < p1 + p2:
            ret_value = algorithms[1](mapping, circuit, hardware)
        else:
            ret_value = algorithms[2](mapping, circuit, hardware)
        return ret_value

    return ret


def _random_shuffle(
    mapping: ty.Dict[Qubit, int], _: QuantumCircuit, _2: IBMQHardwareArchitecture
) -> ty.Dict[Qubit, int]:
    values = list(mapping.values())
    random.shuffle(values)
    new_mapping = dict()
    for i, qubit in enumerate(mapping):
        new_mapping[qubit] = values[i]
    return new_mapping


def _random_expand(
    mapping: ty.Dict[Qubit, int],
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
) -> ty.Dict[Qubit, int]:
    qubit_number = hardware.qubit_number
    if len(mapping) == qubit_number:
        return _random_shuffle(mapping, circuit, hardware)
    not_used_qubits = list(set(range(qubit_number)) - set(mapping.values()))
    new_qubit = random.choice(not_used_qubits)
    new_mapping = copy(mapping)
    new_mapping[random.choice(list(new_mapping.keys()))] = new_qubit
    return new_mapping


def _get_idle_qubits(circuit: QuantumCircuit) -> ty.List[Qubit]:
    dag = circuit_to_dag(circuit)
    return [bit for bit in dag.idle_wires() if isinstance(bit, Qubit)]


def _hardware_aware_expand(
    mapping: ty.Dict[Qubit, int],
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
) -> ty.Dict[Qubit, int]:
    qubits = list(mapping.values())
    idle_qubits = {mapping[qubit] for qubit in _get_idle_qubits(circuit)}
    used_qubits = list(set(qubits) - idle_qubits)
    if not idle_qubits:
        return _random_shuffle(mapping, circuit, hardware)
    # Compute a weight for each qubit.
    # A qubit with a lot of links to other qubits in the mapping is good.
    # A qubit with bad links is not that good.
    weights: ty.List[float] = list()
    outside_qubits_weights = dict()
    for qubit in used_qubits:
        weights.append(0.5 * (1 - hardware.get_qubit_readout_error(qubit)))
        for neighbour in hardware.neighbors(qubit):
            # Only count the neighbour if it is also in the mapping.
            if neighbour in used_qubits:
                weights[-1] += 1 - hardware.get_link_error_rate(qubit, neighbour)
            # Else, we keep an eye on the qubits that are not in the mapping because
            # we will need the best of them to add it to the mapping.
            else:
                if neighbour not in outside_qubits_weights.keys():
                    outside_qubits_weights[neighbour] = 0.5 * (
                        1 - hardware.get_qubit_readout_error(qubit)
                    )
                else:
                    outside_qubits_weights[neighbour] += (
                        outside_qubits_weights.get(neighbour, 0)
                        + 1
                        - hardware.get_link_error_rate(qubit, neighbour)
                    )
    worst_qubit_index = _argmin(weights)
    best_outside_qubit_index = None
    best_outside_weight = 0
    for neighbour, weight in outside_qubits_weights.items():
        if weight > best_outside_weight:
            best_outside_qubit_index = neighbour
            best_outside_weight = weight
    # Now exchange the 2 qubits
    inverse_mapping = {v: k for k, v in mapping.items()}
    inverse_mapping[worst_qubit_index], inverse_mapping[best_outside_qubit_index] = (
        inverse_mapping[best_outside_qubit_index],
        inverse_mapping[worst_qubit_index],
    )
    return {v: k for k, v in inverse_mapping.items()}


def _random_reset(
    mapping: ty.Dict[Qubit, int],
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
) -> ty.Dict[Qubit, int]:
    qubits = list(mapping.keys())
    values = random.sample(list(range(hardware.qubit_number)), len(qubits))
    new_mapping = dict()
    for q, v in zip(qubits, values):
        new_mapping[q] = v
    return new_mapping


def _hardware_aware_reset(
    mapping: ty.Dict[Qubit, int],
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
) -> ty.Dict[Qubit, int]:
    starting_qubit = random.randint(0, hardware.qubit_number - 1)
    qubits: ty.List[int] = [starting_qubit]
    weights: ty.Dict[int, float] = dict()
    while len(qubits) < len(mapping):
        # 1. Update the weights
        for neighbour in hardware.neighbors(qubits[-1]):
            if neighbour not in qubits:
                if neighbour not in weights.keys():
                    weights[neighbour] = 0.5 * (
                        1 - hardware.get_qubit_readout_error(neighbour)
                    )
                else:
                    weights[neighbour] += (
                        weights.get(neighbour, 0)
                        + 1
                        - hardware.get_link_error_rate(qubits[-1], neighbour)
                    )
        # Find the best weighted qubit
        best_weight, best_qubit = 0, None
        for qubit, weight in weights.items():
            if weight > best_weight:
                best_qubit = qubit
                best_weight = weight
        # Insert it in the qubit list
        qubits.append(best_qubit)
        del weights[best_qubit]
    # Finally, return a mapping with the chosen qubits
    return {qubit: idx for qubit, idx in zip(mapping.keys(), qubits)}


def get_neighbour_improved(
    mapping: ty.Dict[Qubit, int],
    hardware: IBMQHardwareArchitecture,
    policy: ty.Callable[
        [
            ty.Dict[Qubit, int],
            IBMQHardwareArchitecture,
            ty.List[NeighbourMappingAlgorithmType],
        ],
        NeighbourMappingAlgorithmType,
    ],
    algorithms: ty.List[NeighbourMappingAlgorithmType],
) -> ty.Dict[Qubit, int]:
    algorithm = policy(mapping, hardware, algorithms)
    return algorithm(mapping, hardware)


def get_initial_mapping_from_annealing(
    cost_function: ty.Callable[
        [ty.Dict[Qubit, int], QuantumCircuit, IBMQHardwareArchitecture], float
    ],
    quantum_circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    initial_mapping: ty.Optional[ty.Dict[Qubit, int]] = None,
    get_neighbour_func: ty.Callable[
        [ty.Dict[Qubit, int]], ty.Dict[Qubit, int]
    ] = get_neighbour_random,
    max_steps: int = 1000,
    temp_begin: float = 10.0,
    cost_threshold: float = 1e-6,
    schedule_func: ty.Callable[[float], float] = lambda x: x * 0.99,
) -> ty.Tuple[ty.Dict[Qubit, int], float, int]:
    # Generate a random initial mapping
    if initial_mapping is None:
        initial_mapping = get_random_mapping(quantum_circuit)

    mapping, cost, iteration_number = simulated_annealing(
        initial_mapping,
        lambda mapping: cost_function(mapping, quantum_circuit, hardware),
        get_neighbour_func,
        temp_begin,
        max_steps,
        schedule_func,
        cost_threshold,
    )
    return mapping, cost, iteration_number


def get_best_mapping_random(
    circuit: QuantumCircuit,
    cost_function: ty.Callable[
        [ty.Dict[Qubit, int], QuantumCircuit, IBMQHardwareArchitecture], float
    ],
    hardware: IBMQHardwareArchitecture,
    maximum_allowed_evaluations: int,
) -> ty.Dict[Qubit, int]:
    best_mapping = get_random_mapping(circuit)
    best_cost = cost_function(best_mapping, circuit, hardware)
    for _ in range(maximum_allowed_evaluations - 1):
        mapping = get_random_mapping(circuit)
        cost = cost_function(mapping, circuit, hardware)
        if cost < best_cost:
            best_mapping = mapping
            best_cost = cost
    return best_mapping


def get_best_mapping_sabre(
    circuit: QuantumCircuit,
    mapping_algorithm: ty.Callable[
        [QuantumCircuit, IBMQHardwareArchitecture, ty.Dict[Qubit, int]],
        ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]],
    ],
    cost_function: ty.Callable[
        [ty.Dict[Qubit, int], QuantumCircuit, IBMQHardwareArchitecture], float
    ],
    hardware: IBMQHardwareArchitecture,
    maximum_allowed_evaluations: int,
) -> ty.Dict[Qubit, int]:
    if maximum_allowed_evaluations < 2:
        print("Not enough allowed evaluations!")
        exit(1)
    best_mapping = initial_mapping_from_sabre(circuit, hardware, mapping_algorithm)
    best_cost = cost_function(best_mapping, circuit, hardware)
    for i in range(maximum_allowed_evaluations // 2 - 1):
        mapping = initial_mapping_from_sabre(circuit, hardware, mapping_algorithm)
        cost = cost_function(mapping, circuit, hardware)
        if cost < best_cost:
            best_mapping, best_cost = mapping, cost
    return best_mapping


def get_best_mapping_from_annealing(
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    cost_function: ty.Callable[
        [ty.Dict[Qubit, int], QuantumCircuit, IBMQHardwareArchitecture], float
    ],
    maximum_allowed_evaluations: int,
    get_neighbour_func: ty.Callable[
        [ty.Dict[Qubit, int]], ty.Dict[Qubit, int]
    ] = get_neighbour_random,
):
    temp_begin = 1000.0
    alpha = math.exp(
        (-6 * math.log(10) - math.log(temp_begin)) / maximum_allowed_evaluations
    )
    mapping, *_ = get_initial_mapping_from_annealing(
        cost_function,
        circuit,
        hardware,
        max_steps=maximum_allowed_evaluations,
        temp_begin=temp_begin,
        schedule_func=lambda x: x * alpha,
        get_neighbour_func=get_neighbour_func,
    )
    return mapping


def get_best_mapping_from_iterative_forward_backward(
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    mapping_algorithm: ty.Callable[
        [QuantumCircuit, IBMQHardwareArchitecture, ty.Dict[Qubit, int]],
        ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]],
    ],
    circuit_cost: ty.Callable[
        [ty.Dict[Qubit, int], QuantumCircuit, IBMQHardwareArchitecture], float
    ],
    maximum_allowed_evaluations: int,
):
    (
        best_mapping,
        best_cost,
        call_number,
    ) = initial_mapping_from_iterative_forward_backward(
        circuit,
        hardware,
        mapping_algorithm,
        circuit_cost=circuit_cost,
        maximum_mapping_procedure_calls=maximum_allowed_evaluations,
    )
    while maximum_allowed_evaluations - call_number >= 2:
        mapping, cost, i = initial_mapping_from_iterative_forward_backward(
            circuit,
            hardware,
            mapping_algorithm,
            circuit_cost=circuit_cost,
            maximum_mapping_procedure_calls=maximum_allowed_evaluations - call_number,
        )
        call_number += i
        if cost < best_cost:
            best_mapping, best_cost = mapping, cost
    return best_mapping
