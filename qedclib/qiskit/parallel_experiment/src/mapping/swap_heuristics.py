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

import logging
import typing as ty

import numpy
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGNode

from hardware.IBMQHardwareArchitecture import (
    IBMQHardwareArchitecture,
)
from mapping.layer import QuantumLayer, update_layer
from mapping.gates import (
    TwoQubitGate,
    SwapTwoQubitGate,
)

logger = logging.getLogger("heuristics")


def _gate_op_cost(
    op: DAGNode,
    distance_matrix: numpy.ndarray,
    mapping: ty.Dict[Qubit, int],
    hardware: IBMQHardwareArchitecture,
) -> float:
    if hardware.is_ignored_operation(op):
        return 0
    if len(op.qargs) == 1:
        # SABRE ignores 1-qubit gates
        return 0
    elif len(op.qargs) == 2:
        # This is a CNOT
        source, sink = op.qargs
        qubit_control, qubit_target = mapping[source], mapping[sink]
        return distance_matrix[qubit_control, qubit_target]
    else:
        logger.warning(
            f"Found a quantum operation applied on '{len(op.qargs)}' qubits. This "
            f"operation will be excluded from the cost computation."
        )
        return 0


def swap_heuristic(
    hardware: IBMQHardwareArchitecture,
    front_layer: QuantumLayer,
    topological_nodes: ty.List[DAGNode],
    current_node_index: int,
    current_mapping: ty.Dict[Qubit, int],
    initial_mapping: ty.Dict[Qubit, int],
    trans_mapping: ty.Dict[Qubit, int],
    distance_matrix: numpy.ndarray,
    tentative_gate: TwoQubitGate,
    look_ahead_depth: int = 20,
    look_ahead_weight: float = 0.5,
) -> float:
    """The heuristic cost function used in the SABRE optimiser.

    :param hardware: the SABRE optimiser does not take into account the hardware data to
        compute the heuristic cost, only to generate the possible SWAPs to evaluate
        with this heuristic. The SABRE heuristic only uses the distance matrix.
        Nevertheless, this implementation uses the hardware data to check if some
        gates are ignored (such as barriers for example).
    :param front_layer: the current front layer. Used to compute an "immediate" cost,
        i.e. a quantity that will tell us if the SWAP/Bridge is useful to execute
        gates in the front layer.
    :param topological_nodes: the list of all the DAGNodes of the quantum circuit,
        sorted in topological order.
    :param current_node_index: index of the first non-processed node.
    :param current_mapping: the mapping *before* applying the given SWAP.
    :param distance_matrix: the pre-computed distance matrix between each qubits.
    :param tentative_gate: the SWAP we want to estimate the usefulness of.
    :param look_ahead_depth: the depth of the look-ahead. The procedure will consider
        gates that will be executed in the future (i.e. not in the front layer) up to
        the given depth. Note that 1-qubit gates are not ignored, which means that a
        depth of 3 will not guarantee that there is at least 3 CNOTs in the
        look-ahead set.
    :param look_ahead_weight: weight of the look-ahead. The actual gates (i.e. the
        gates in the front layer) have a weight of 1.
    :return: the heuristic cost of the given SWAP/Bridge according to the current
        state of the algorithm.
    """
    # First, compute the proposed new mapping
    new_mapping = tentative_gate.update_mapping(current_mapping)
    # Compute H_basic, the cost associated with the distance.
    H_basic = 0.0
    H_basic_gate_number = 0
    H_tentative = 0.0
    H_tentative_gate_number = 0

    for op in front_layer.ops:
        # Only add the gate to the cost if the gate is not already implemented by the
        # SWAP/Bridge
        if not tentative_gate.implements_operation(op, initial_mapping, trans_mapping):
            H_basic += _gate_op_cost(op, distance_matrix, new_mapping, hardware)
            H_basic_gate_number += 1
    # Compute H, the cost cost that encourage parallelism and adds some look-ahead
    # ability.
    if isinstance(tentative_gate, SwapTwoQubitGate):
        H_tentative += tentative_gate.cost(hardware, current_mapping, distance_matrix)
        H_tentative_gate_number += 3
    else:
        H_tentative += tentative_gate.cost(hardware, initial_mapping, distance_matrix)
        H_tentative_gate_number += 4

    future_nodes_layer = QuantumLayer(max_depth=look_ahead_depth)
    # We do not use the return of update_layer because we do not care about the
    # number of gates that were added. Still, we add the firsts look_ahead_depth layers
    # of our future gates in this set to have this look-ahead ability.
    _ = update_layer(future_nodes_layer, topological_nodes, current_node_index)
    # The decay is not implemented in the code the authors gave us and not
    # sufficiently explained in the paper to implement it without guessing. Not
    # implementing it for the moment...
    # H_basic = (H_basic / H_basic_gate_number) if H_basic_gate_number != 0 else 0
    # gain = gain / H_basic_gate_number if H_basic_gate_number != 0 else 0

    H = (H_basic + H_tentative) / (H_basic_gate_number + H_tentative_gate_number)

    H_extended = 0.0
    if future_nodes_layer:
        # Only add this cost if there are nodes in the future_node_layer
        H_extended += (
            look_ahead_weight
            * sum(
                _gate_op_cost(op, distance_matrix, new_mapping, hardware)
                for op in future_nodes_layer.ops
            )
            / len(future_nodes_layer)
        )
    H += H_extended
    return H


