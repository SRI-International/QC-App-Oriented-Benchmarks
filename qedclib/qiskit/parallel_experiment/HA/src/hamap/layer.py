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

from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit, DAGNode

logger = logging.getLogger("hamap.layer")


class QuantumLayer:
    def __init__(self, max_depth: int = 1):
        self._operations_dict: ty.Dict[Qubit, ty.List[DAGNode]] = dict()
        self._operations: ty.List[DAGNode] = list()
        self._max_depth = max_depth

    def is_qubit_busy(self, qubit: Qubit) -> bool:
        return (
            qubit in self._operations_dict
            and len(self._operations_dict[qubit]) >= self._max_depth
        )

    def is_operation_addable(self, op: DAGNode) -> bool:
        for qubit in op.qargs:
            if self.is_qubit_busy(qubit):
                return False
        return True

    def add_operation(self, op: DAGNode):
        # First check that the operation can be added
        if not self.is_operation_addable(op):
            logger.warning(
                "Trying to add to the layer an operation that already overlaps "
                "with another operation in the layer. Ignoring the add."
            )
            return
        # Then if the operation can be added, add it.
        for qubit in op.qargs:
            # If the qubit key does not exist, create the entry with an empty list.
            # Then, append op to the list.
            self._operations_dict.setdefault(qubit, []).append(op)
        self._operations.append(op)

    def remove_operation(self, op: DAGNode):
        for qubit in op.qargs:
            self._operations_dict[qubit].remove(op)
            # If the list of operations on a given qubit becomes empty, we delete the
            # whole entry.
            if not self._operations_dict[qubit]:
                del self._operations_dict[qubit]
        self._operations.remove(op)

    def remove_operations_from_layer(self, layer: "QuantumLayer"):
        for op in layer._operations:
            self.remove_operation(op)

    def is_empty(self) -> bool:
        return len(self._operations) == 0

    def apply_back_to_dag_circuit(
        self,
        dag_circuit: DAGCircuit,
        initial_mapping: ty.Dict[Qubit, int],
        trans_mapping: ty.Dict[Qubit, int],
    ):
        reversed_trans_mapping = {val: key for key, val in trans_mapping.items()}
        for op in self._operations:
            logical_qubits = [initial_mapping[qubit] for qubit in op.qargs]
            new_physical_qubits = [
                reversed_trans_mapping[qubit_index] for qubit_index in logical_qubits
            ]
            #print(new_physical_qubits)
            dag_circuit.apply_operation_back(
                op.op, new_physical_qubits, list(op.cargs)
            )

    def __len__(self) -> int:
        return len(self._operations)

    @property
    def ops(self) -> ty.List[DAGNode]:
        """Returns the list of **all** the operations."""
        return self._operations


#
# class IgnoringQuantumLayer(QuantumLayer):
#     """A class that ignores some quantum operations.
#
#     Ignoring in this context means that the ignored gates are not counted in the gate
#     depth or in the gate count.
#     WARNING: ignored gates are still added to the layer internally and will be
#     present in the ops attribute, or will be added to the DAGCircuit in
#     apply_back_to_dag_circuit.
#     """
#
#     def __init__(
#         self, is_operation_ignored: ty.Callable[[DAGNode], bool], max_depth: int = 1
#     ):
#         super().__init__(max_depth)
#         # self._is_operation_ignored = is_operation_ignored
#         self._is_operation_ignored = lambda x: False
#         self._ignored_operation_count = 0
#
#     def is_operation_addable(self, op: DAGNode) -> bool:
#         if self._is_operation_ignored(op):
#             return True
#         return super().is_operation_addable(op)
#
#     def add_operation(self, op: DAGNode):
#         if self._is_operation_ignored(op):
#             self._ignored_operation_count += 1
#             self._operations.append(op)
#             return
#         super().add_operation(op)
#
#     def remove_operation(self, op: DAGNode):
#         if self._is_operation_ignored(op):
#             self._ignored_operation_count -= 1
#             self._operations.remove(op)
#             return
#         super().remove_operation(op)
#
#     def __len__(self) -> int:
#         return super().__len__() - self._ignored_operation_count
#
#     @staticmethod
#     def _compose_ignore_functions(*functions_to_compose):
#         def _internal_function(x):
#             result = False
#             for func in functions_to_compose:
#                 result = result or func(x)
#             return result
#
#         return _internal_function
#
#
# class IBMQQuantumLayer(IgnoringQuantumLayer):
#     @staticmethod
#     def _is_ibmq_ignored_operation(op: DAGNode) -> bool:
#         return op.name == "barrier"
#
#     def __init__(
#         self,
#         is_operation_ignored: ty.Callable[[DAGNode], bool] = None,
#         max_depth: int = 1,
#     ):
#         if is_operation_ignored is None:
#             is_operation_ignored = IBMQQuantumLayer._is_ibmq_ignored_operation
#         else:
#             # We need to compose the two functions in one.
#             is_operation_ignored = IgnoringQuantumLayer._compose_ignore_functions(
#                 is_operation_ignored, IBMQQuantumLayer._is_ibmq_ignored_operation
#             )
#         super().__init__(is_operation_ignored, max_depth)
#
#
# class IBMQOnlyCNOTQuantumLayer(IBMQQuantumLayer):
#     """A class where only CNOT are accounted in the depth.
#
#     1-qubit gates are still added, but not counted in the maximum depth or in the
#     gate count.
#     """
#
#     @staticmethod
#     def _is_1_qubit_gate(op: DAGNode) -> bool:
#         return len(op.qargs) == 1
#
#     def __init__(
#         self,
#         is_operation_ignored: ty.Callable[[DAGNode], bool] = None,
#         max_depth: int = 1,
#     ):
#         if is_operation_ignored is None:
#             is_operation_ignored = IBMQOnlyCNOTQuantumLayer._is_1_qubit_gate
#         else:
#             # We need to compose the two functions in one.
#             is_operation_ignored = IgnoringQuantumLayer._compose_ignore_functions(
#                 is_operation_ignored, IBMQOnlyCNOTQuantumLayer._is_1_qubit_gate
#             )
#         super().__init__(is_operation_ignored, max_depth)


def update_layer(
    layer: QuantumLayer,
    topological_order_nodes: ty.List[DAGNode],
    current_node_index: int,
) -> int:
    """Updates the given layer with new operations if possible and returns it.

    WARNING: the given layer is modified in place.

    :param layer: a layer of gates. Might be empty or already populated with some gates.
    :param topological_order_nodes: a list of DAGNodes sorted in topological order.
    :param current_node_index: index of the first non-processed node.
    :return: The index of the first node non added to the layer. If the return value
        is greater or equal to len(topological_order_nodes), this means that all the
        nodes have been processed.
    """
    if current_node_index >= len(topological_order_nodes):
        # If the index is after the last node, then just return it.
        return current_node_index

    # Add nodes while we can.
    for node_index in range(current_node_index, len(topological_order_nodes)):
        if not layer.is_operation_addable(topological_order_nodes[node_index]):
            return node_index
        layer.add_operation(topological_order_nodes[node_index])
    # If we are here this means that we processed all the nodes left, so just return
    # a value saying that there is nothing left to process.
    return len(topological_order_nodes)
