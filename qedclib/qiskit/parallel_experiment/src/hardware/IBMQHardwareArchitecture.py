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

import logging
import typing as ty
import pickle
from pathlib import Path
import networkx as nx
import numpy as np
from hardware.HardwareArchitecture import HardwareArchitecture
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGNode

logger = logging.getLogger("hardware.IBMQHardwareArchitecture")

def cnot_error_rate_function(vertex, hardware):
    source, sink = vertex
    return hardware.get_link_error_rate(source, sink)

def swap_number(*_) -> float:
    return 1.0

def readout_error_rate_function(qubit_index, hardware):
    return hardware.get_qubit_readout_error(qubit_index)

class IBMQHardwareArchitecture(HardwareArchitecture):
    _hardware_directory: Path = Path(
        __file__
    ).parent/ "architectures_saved_data"

    @staticmethod
    def _get_value(value: float, unit: str):
        # We want time in nanoseconds
        if unit == "us":
            return value * 10 ** 3
        elif unit == "ns":
            return value

        # We want everything in Hertz
        elif unit == "GHz":
            return value * 10 ** 9
        elif unit == "":
            return value
        else:
            logger.error(f"Unsupported unit: '{unit}'")
            exit(1)

    @staticmethod
    def _get_backend(backend_or_name):
        from qiskit.providers.fake_provider import GenericBackendV2
        from qiskit.transpiler import CouplingMap

        # If QED-C passed an actual backend object, use it.
        if not isinstance(backend_or_name, str):
            return backend_or_name

        # Fallback for old code paths.
        logger.info(
            f"Creating fake backend approximating '{backend_or_name}' "
            "(65-qubit heavy-square topology)."
        )

        coupling_map = CouplingMap.from_heavy_square(5)

        backend = GenericBackendV2(
            num_qubits=coupling_map.size(),
            basis_gates=["id", "rz", "sx", "x", "cx"],
            coupling_map=coupling_map,
            seed=42,
        )

        return backend

    def __init__(
            self,
            backend_name: str,
            weight_func: ty.Callable[
                [ty.Tuple[int, int], nx.classes.reportviews.OutEdgeView], float
            ] = None,
            incoming_graph_data=None,
            **kwargs,
    ):
        """
        The architecute of any IBMQ hardware
        :param backend_name: The name of IBMQ hardware
        :param weight_func: a function taking an edge identifier and all the edges of
            the architecture in parameters and that return the cost associated to this
            edge. If None, the function returns the execution time of the CNOT gate on
            the given link.
        :param incoming_graph_data: forwarded to :py:method:`networkx.Digraph.__init__`.
        :param kwargs: forwarded to :py:method:`networkx.Digraph.__init__`.
        """
        super().__init__(incoming_graph_data, **kwargs)
        self._ignored_gates = {"barrier"}
        if weight_func is None:
            weight_func = swap_number

        self._weight_func = weight_func
        self.name = backend_name
        backend = IBMQHardwareArchitecture._get_backend(backend_name)

        qubit_number = backend.num_qubits
        coupling_edges = list(backend.coupling_map.get_edges())
        self._coupling_graph = coupling_edges

        for qubit_index in range(qubit_number):
            readout_error = 0.01
            if 'measure' in backend.target and (qubit_index,) in backend.target['measure']:
                props = backend.target['measure'][(qubit_index,)]
                if props and props.error is not None:
                    readout_error = props.error
            self.add_qubit(readout_error=readout_error, T1=0.0, T2=0.0, frequency=0.0)

        for source, sink in coupling_edges:
            gate_error = 0.01
            gate_length = 300.0
            if 'cx' in backend.target and (source, sink) in backend.target['cx']:
                cx_props = backend.target['cx'][(source, sink)]
                if cx_props:
                    if cx_props.error is not None:
                        gate_error = cx_props.error
                    if cx_props.duration is not None:
                        gate_length = cx_props.duration * 1e9
            self.add_link(source, sink, gate_error=gate_error, gate_length=gate_length)

        self.update_link_weights()


    @property
    def hardware_coupling_graph(self):
        return self._coupling_graph


    def get_link_error_rate(self, source: int, sink: int) -> float:
        return self.edges[source, sink]['gate_error']

    def get_qubit_readout_error(self, qubit_index: int) -> float:
        return self.nodes[qubit_index]['readout_error']

    def get_single_qubit_error(self, qubit_index: int) -> float:
        return self.nodes[qubit_index].get('readout_error', 0.01)

    def update_link_weights(self, distance_matrix: np.ndarray=None):
        """
        Updates the weight on each qubit link
        """
        weight_attributes = dict()
        if distance_matrix is None:
            for edge in self.edges:
                weight_attributes[edge] = self._weight_func(edge,self)
        else:
            for edge in self.edges:
                i = edge[0]
                j = edge[1]
                weight_attributes[edge] = distance_matrix.item(i,j)

        nx.set_edge_attributes(self, weight_attributes, name="weight")

    @property
    def weight_function(self):
        return self._weight_func

    @weight_function.setter
    def weight_function(self, value):
        self._weight_func = value
        self.update_link_weights()

    def get_link_execution_time(self, source: int, sink: int) -> float:
        """Returns the execution time of the CNOT gate between source and sink in \
        nano-seconds."""
        return self.edges[source, sink]["gate_length"]

    def is_ignored_operation(self, op: DAGNode) -> bool:
        return op.name in self._ignored_gates

    def can_natively_execute_operation(
        self, op: DAGNode, current_mapping: ty.Dict[Qubit, int]
    ) -> bool:
        if self.is_ignored_operation(op):
            return True

        if len(op.qargs) == 1:
            # If this is a 1-qubit operation, then the hardware can always execute it
            # natively.
            return True
        elif len(op.qargs) == 2:
            # q1, q2 = initial_mapping[op.qargs[0]], initial_mapping[op.qargs[1]]
            # inverse_mapping = {val: key for key, val in current_mapping.items()}
            # control, target = inverse_mapping[q1], inverse_mapping[q2]
            source, sink = current_mapping[op.qargs[0]], current_mapping[op.qargs[1]]
            # source = mapping[op.qargs[0]]
            # sink = mapping[op.qargs[1]]
            return (source, sink) in self.edges
        else:
            logger.error(
                f"Found invalid operation acting on {len(op.qargs)} qubits. "
                f"Ignoring the operation {op.name} and exiting."
            )
            exit(1)

    def save(self, hardware_name: str):
        filepath = (
            IBMQHardwareArchitecture._hardware_directory / f"{hardware_name}.archdata"
        )
        with open(str(filepath), "wb") as f:
            logger.info(f"Saving IBMQHardwareArchitecture instance in '{filepath}'.")
            pickle.dump(self, f)

    @staticmethod
    def load(hardware_name: str) -> "IBMQHardwareArchitecture":
        filepath = (
            IBMQHardwareArchitecture._hardware_directory / f"{hardware_name}.archdata"
        )
        with open(str(filepath), "rb") as f:
            logger.info(f"Loading IBMQHardwareArchitecture instance from '{filepath}'.")
            return pickle.load(f)











