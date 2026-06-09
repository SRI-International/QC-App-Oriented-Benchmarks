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

import logging
import typing as ty

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from HA.src.hamap.gates import (
    TwoQubitGate,
    SwapTwoQubitGate,
    BridgeTwoQubitGate,
)
from HA.src.hamap.hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
from HA.src.hamap.layer import QuantumLayer
from HA.src.hamap.mapping_to_str import mapping_to_str

logger = logging.getLogger("hamap.swap")


def get_all_swap_candidates(
    layer: QuantumLayer,
    hardware: IBMQHardwareArchitecture,
    current_mapping: ty.Dict[Qubit, int],
    explored_mappings: ty.Set[str],
) -> ty.List[SwapTwoQubitGate]:
    # First compute all the qubits involved in the given layer
    qubits_involved_in_front_layer = set()
    for op in layer.ops:
        qubits_involved_in_front_layer.update(op.qargs)
    inverse_mapping = {val: key for key, val in current_mapping.items()}
    # Then for all the possible links that involve at least one of the qubits used by
    # the gates in the given layer, add this link as a possible SWAP.
    all_swaps = list()
    for involved_qubit in qubits_involved_in_front_layer:
        qubit_index = current_mapping[involved_qubit]
        # For all the links that involve the current qubit.
        for source, sink in hardware.out_edges(qubit_index):
            two_qubit_gate = SwapTwoQubitGate(
                inverse_mapping[source], inverse_mapping[sink]
            )
            # Check that the mapping has not already been explored in this
            # SWAP-insertion pass.
            if (
                mapping_to_str(two_qubit_gate.update_mapping(current_mapping))
                not in explored_mappings
            ):
                all_swaps.append(two_qubit_gate)
    return all_swaps


def get_all_bridge_candidates(
    layer: QuantumLayer,
    hardware: IBMQHardwareArchitecture,
    initial_mapping: ty.Dict[Qubit, int],
    trans_mapping: ty.Dict[Qubit, int],
    current_mapping: ty.Dict[Qubit, int],
    explored_mappings: ty.Set[str],
) -> ty.List[BridgeTwoQubitGate]:
    all_bridges = []

    inverse_trans_mapping = {val: key for key, val in trans_mapping.items()}
    inverse_mapping = {val: key for key, val in initial_mapping.items()}
    for op in layer.ops:
        if len(op.qargs) < 2:
            # We just pass 1 qubit gates because they do not participate in the
            # Bridge operation
            continue
        if len(op.qargs) != 2:
            logger.warning("A 3-qubit or more gate has been found in the circuit.")
            continue

        control, target = op.qargs
        control_index = initial_mapping[inverse_trans_mapping[initial_mapping[control]]]
        target_index = initial_mapping[inverse_trans_mapping[initial_mapping[target]]]
        # For each qubit q linked with control, check if target is linked with q.
        for _, potential_middle_index in hardware.out_edges(control_index):
            for _, potential_target_index in hardware.out_edges(potential_middle_index):
                if potential_target_index == target_index:
                    two_qubit_gate = BridgeTwoQubitGate(
                        inverse_trans_mapping[initial_mapping[control]],
                        inverse_mapping[potential_middle_index],
                        inverse_trans_mapping[initial_mapping[target]],
                    )
                    # Check that the mapping has not already been explored in this
                    # SWAP-insertion pass.
                    if (
                        mapping_to_str(two_qubit_gate.update_mapping(current_mapping))
                        not in explored_mappings
                    ):
                        all_bridges.append(two_qubit_gate)
    return all_bridges


def get_all_swap_bridge_candidates(
    layer: QuantumLayer,
    hardware: IBMQHardwareArchitecture,
    initial_mapping: ty.Dict[Qubit, int],
    current_mapping: ty.Dict[Qubit, int],
    trans_mapping: ty.Dict[Qubit, int],
    explored_mappings: ty.Set[str],
) -> ty.List[TwoQubitGate]:
    swap_candidates = get_all_swap_candidates(
        layer, hardware, current_mapping, explored_mappings
    )
    bridge_candidates = get_all_bridge_candidates(
        layer, hardware, initial_mapping, trans_mapping, current_mapping, explored_mappings
    )
    return swap_candidates + bridge_candidates


def change_mapping(
    start_mapping: ty.Dict[Qubit, int],
    final_mapping: ty.Dict[Qubit, int],
    circuit: QuantumCircuit,
) -> None:
    reverse_initial_mapping = {val: key for key, val in start_mapping.items()}
    reverse_final_mapping = {val: key for key, val in final_mapping.items()}
    # For each qubit index, exchange the qubit currently occupying the position (
    # given by start_mapping) with the qubit that should be there at the end (given
    # by final_mapping) and update the current mapping. The last swap is not needed
    # because the qubit should already be in the right place.
    for i in range(len(reverse_initial_mapping) - 1):
        s1, s2 = reverse_initial_mapping[i], reverse_final_mapping[i]
        if s1 != s2:
            circuit.swap(s1, s2)
            # Reflect the SWAP on the current mapping
            start_mapping[s1], start_mapping[s2] = start_mapping[s2], start_mapping[s1]
            reverse_initial_mapping = {val: key for key, val in start_mapping.items()}
