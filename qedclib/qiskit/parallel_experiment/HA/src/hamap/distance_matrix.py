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

from functools import partial

import networkx as nx
import numpy

from HA.src.hamap.hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture


def _get_swap_number(*_) -> float:
    return 1.0


def get_distance_matrix_swap_number(
    hardware: IBMQHardwareArchitecture,
) -> numpy.ndarray:
    hardware.weight_function = _get_swap_number
    return nx.floyd_warshall_numpy(hardware)


def _get_swap_execution_time_cost(node, hardware: IBMQHardwareArchitecture) -> float:
    source, sink = node
    cnot_cost = hardware.get_link_execution_time(source, sink)
    reversed_cnot_cost = hardware.get_link_execution_time(sink, source)
    return cnot_cost + reversed_cnot_cost + min(cnot_cost, reversed_cnot_cost)


def get_distance_matrix_execution_time_cost(
    hardware: IBMQHardwareArchitecture,
) -> numpy.ndarray:
    hardware.weight_function = _get_swap_execution_time_cost
    return nx.floyd_warshall_numpy(hardware)


def _get_swap_error_cost(node, hardware: IBMQHardwareArchitecture) -> float:
    source, sink = node
    cnot_fidelity = 1 - hardware.get_link_error_rate(source, sink)
    reversed_cnot_fidelity = 1 - hardware.get_link_error_rate(sink, source)
    return 1 - cnot_fidelity * reversed_cnot_fidelity * max(
        cnot_fidelity, reversed_cnot_fidelity
    )


def get_distance_matrix_error_cost(hardware: IBMQHardwareArchitecture) -> numpy.ndarray:
    hardware.weight_function = _get_swap_error_cost
    return nx.floyd_warshall_numpy(hardware)


def _get_mixed_cost(
    node,
    hardware: IBMQHardwareArchitecture,
    swap_weight: float,
    execution_time_weight: float,
    error_weight: float,
) -> float:
    swap_cost = swap_weight * _get_swap_number(node, hardware)
    execution_time_cost = execution_time_weight * _get_swap_execution_time_cost(
        node, hardware
    )
    error_cost = error_weight * _get_swap_error_cost(node, hardware)
    return (swap_cost + execution_time_cost + error_cost) / (
        swap_weight + execution_time_weight + error_weight
    )


def get_distance_matrix_mixed(
    hardware: IBMQHardwareArchitecture,
    swap_weight: float,
    execution_time_weight: float,
    error_weight: float,
) -> numpy.ndarray:
    # if swap_weight < 0 or execution_time_weight < 0 or error_weight < 0:
    #     raise RuntimeError("All the weight should be positive.")
    # coefficient_sum = swap_weight + execution_time_weight + error_weight
    # if coefficient_sum < 1e-10:
    #     raise RuntimeError("The coefficients you provided are too small.")
    # hardware.weight_function = partial(
    #     _get_mixed_cost,
    #     swap_weight=swap_weight,
    #     execution_time_weight=execution_time_weight,
    #     error_weight=error_weight,
    # )
    # return nx.floyd_warshall_numpy(hardware)
    distance_matrix_swap_number = get_distance_matrix_error_cost(hardware)
    norm_swap_number = numpy.linalg.norm(distance_matrix_swap_number)
    swap_cost = swap_weight * distance_matrix_swap_number / norm_swap_number

    distance_matrix_execution_time = get_distance_matrix_execution_time_cost(hardware)
    norm_execution_time = numpy.linalg.norm(distance_matrix_execution_time)
    execution_time_cost = execution_time_weight * distance_matrix_execution_time / norm_execution_time

    distance_matrix_error_cost = get_distance_matrix_swap_number(hardware)
    norm_error_cost = numpy.linalg.norm(distance_matrix_error_cost)
    error_cost = error_weight * distance_matrix_error_cost / norm_error_cost

    return (swap_cost + execution_time_cost + error_cost) / (
            swap_weight + execution_time_weight + error_weight
    )

def get_distance_matrix_swap_number_and_error(
    hardware: IBMQHardwareArchitecture,
) -> numpy.ndarray:
    return get_distance_matrix_mixed(hardware, 0.5, 0, 0.5)
