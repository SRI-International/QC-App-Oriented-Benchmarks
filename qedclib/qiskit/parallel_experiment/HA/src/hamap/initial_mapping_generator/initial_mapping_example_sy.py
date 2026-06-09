# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (05/2020)
# Contributor: Adrien Suau (<adrien.suau@cerfacs.fr>
#                           <adrien.suau@lirmm.fr>)
#
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

# Hack for import of IBMQSubmitter
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent)
from initial_mapping_wrapper import initial_mapping

from qiskit import QuantumCircuit

from sabre_distance_bridge import chips
import random
from sabre_distance_bridge import utils
import os
import time
from sabre_distance_bridge.mapping import (
    one_round_optimization,
    output_context,
    change_mapping,
)
from sabre_output_test.mapping import (
    one_round_optimization_sabre,
    output_context_sabre,
)
from HA.src.hamap.mapping import ha_mapping, ha_mapping_paper_compliant
from pathlib import Path
from qiskit import QuantumCircuit, Aer, execute
import numpy as np

from sabre_output_test.graph import floyd

from sabre_distance_bridge.distance_matrix import (
    get_distance_matrix_swap_number,
    get_distance_matrix_error_cost,
    get_distance_matrix_execution_time_cost,
    get_distance_matrix_cnot_error_cost,
    get_qubit_readout_error,
    get_qubit_matrix_execution_time_cost,
)

from HA.src.hamap import IBMQHardwareArchitecture

qx2 = chips.ibmqx2()
q20 = chips.ibmq20()
q16 = chips.ibmq16()
q_almaden = chips.ibmq20Almaden()
qx5 = chips.ibmq5()


chip = qx5


# initial mapping


def mapping_algorithm(quantum_circuit: QuantumCircuit, hardware, initial_mapping):
    # 1. Compute distance matrices
    cnot_execution_matrix = get_qubit_matrix_execution_time_cost(hardware)

    distance_matrix_error_cost = get_distance_matrix_error_cost(hardware)
    distance_matrix_execution_time_cost = get_distance_matrix_execution_time_cost(
        hardware
    )
    distance_matrix_swap_number = get_distance_matrix_swap_number(hardware)

    norm_error_cost = np.linalg.norm(distance_matrix_error_cost)
    norm_execution_time_cost = np.linalg.norm(distance_matrix_execution_time_cost)
    norm_swap_number_cost = np.linalg.norm(distance_matrix_swap_number)

    distance_matrix_error_cost_norm = distance_matrix_error_cost / norm_error_cost
    distance_matrix_execution_time_cost_norm = (
        distance_matrix_execution_time_cost / norm_execution_time_cost
    )
    distance_matrix_swap_number_norm = (
        distance_matrix_swap_number / norm_swap_number_cost
    )
    distance_mat = (
        0.5 * distance_matrix_error_cost_norm
        + 0 * distance_matrix_execution_time_cost_norm
        + 0.5 * distance_matrix_swap_number_norm
    )

    # 2. Write the quantum to a QASM file
    with open("filename.qasm", "w") as f:
        f.write(quantum_circuit.qasm())

    # 3. Re-open the QASM file for SABRE structures

    (
        qubit_num,
        gate_type,
        gate_qubit,
        cx_gate_num,
        cx_gates,
        context,
    ) = utils.read_flatten_qasm("filename.qasm")

    # 4. Apply the mapping algorithm
    # print('initial mapping ', initial_mapping)
    if isinstance(initial_mapping, dict):
        initial_mapping = [j for i, j in initial_mapping.items()]
    # print('initial mapping ', initial2_mapping)
    swap_num, bridge_num, final_mapping, execution_time = one_round_optimization(
        initial_mapping,
        distance_mat,
        distance_matrix_swap_number,
        cnot_execution_matrix,
        gate_qubit,
        gate_type,
        context,
        chip.qubit_num,
        cx_gate_num,
        chip,
    )

    # Write the QASM from SABRE structure
    output = str(Path(__file__).parent) + "/output/1.qasm"
    with open(output, "w") as file_object:
        file_object.write("OPENQASM 2.0;\n")
        file_object.write('include "qelib1.inc";\n')
        file_object.write("qreg q[" + str(chip.qubit_num) + "];\n")
        file_object.write("creg c[" + str(chip.qubit_num) + "];\n")
        for line_idx in output_context:
            file_object.write(line_idx)

    # 5. QuantumCircuit from the QASM
    mapped_quantum_circuit = QuantumCircuit.from_qasm_file(output)
    # print(mapped_quantum_circuit.count_ops())
    return mapped_quantum_circuit, final_mapping


def mapping_algorithm_sabre(quantum_circuit: QuantumCircuit, hardware, initial_mapping):
    distance_mat = floyd(chip.qubit_num, chip.adj_mat)
    cnot_execution_matrix = get_qubit_matrix_execution_time_cost(hardware)

    # 2. Write the quantum to a QASM file
    with open("filename.qasm", "w") as f:
        f.write(quantum_circuit.qasm())

    # 3. Re-open the QASM file for SABRE structures

    (
        qubit_num,
        gate_type,
        gate_qubit,
        cx_gate_num,
        cx_gates,
        context,
    ) = utils.read_flatten_qasm("filename.qasm")

    if isinstance(initial_mapping, dict):
        initial_mapping = [j for i, j in initial_mapping.items()]

    swap_num, final_mapping, execution_time = one_round_optimization_sabre(
        initial_mapping,
        distance_mat,
        gate_qubit,
        gate_type,
        context,
        chip.qubit_num,
        cx_gate_num,
        chip,
        cnot_execution_matrix,
    )
    output = str(Path(__file__).parent) + "/output/1.qasm"
    # print(output_context_sabre)
    with open(output, "w") as file_object:
        file_object.write("OPENQASM 2.0;\n")
        file_object.write('include "qelib1.inc";\n')
        file_object.write("qreg q[" + str(chip.qubit_num) + "];\n")
        file_object.write("creg c[" + str(chip.qubit_num) + "];\n")
        for line_idx in output_context_sabre:
            file_object.write(line_idx)

    # 5. QuantumCircuit from the QASM
    mapped_quantum_circuit = QuantumCircuit.from_qasm_file(output)
    # print(mapped_quantum_circuit.count_ops())
    return mapped_quantum_circuit, final_mapping


def cost(initial_mapping, quantum_circuit: QuantumCircuit, hardware):
    mapped_circuit, final_mapping = ha_mapping(
        quantum_circuit, initial_mapping, hardware
    )
    cx_num = mapped_circuit.count_ops().get("cx", 0)
    swap_num = mapped_circuit.count_ops().get("swap", 0) * 3
    ops_num = (
        cx_num + swap_num
    )
    print(f"cx is {cx_num} + swap_num {swap_num} is {ops_num}")
    return ops_num


# qc = QuantumCircuit(5, 5)
# qc.x(0)
# qc.h(4)
# filename = "4gt5_75.qasm"
# file_path = (
#     str((Path(__file__).parent).parent)
#     + "/sabre_dynamic_test/sabre_distance_bridge/test/examples/"
#     + filename
# )
file_path = "/home/siyuan/Seafile/Qubit mapping problem bibliography/SABRE/sabre_dynamic_test/sabre_distance_bridge/test/examples/3_17_13.qasm"
qc = QuantumCircuit.from_qasm_file(file_path)

hardware = IBMQHardwareArchitecture("ibmq_montreal")
# print(get_distance_matrix_error_cost(hardware))
for i in range(1):
    # computed_initial_mapping_random = initial_mapping(
    #     qc, hardware, mapping_algorithm, cost, "random", 100
    # )
    # print('random initial mapping', computed_initial_mapping_random)

    # computed_initial_mapping_iterated = initial_mapping(
    #     qc, hardware, mapping_algorithm, cost, "sabre", 100
    # )
    # if isinstance(computed_initial_mapping_iterated, dict):
    #     computed_initial_mapping_iterated = [i for i in computed_initial_mapping_iterated.values()]
    # print('iterated initial mapping', computed_initial_mapping_iterated)
    computed_initial_mapping = initial_mapping(
        qc, hardware, ha_mapping, cost, "sabre", 100
    )
    if isinstance(computed_initial_mapping, dict):
        computed_initial_mapping = [i for i in computed_initial_mapping.values()]
    print(computed_initial_mapping)
"""
best_cost, best_mapping = ....
for i in range(300 // 3):
    random_initial_mapping = random()
    initial_mapping = sabre(random_initial_mapping, circuit)
    cost = cost_function(initial_mapping, circuit)
    if cost < best_cost:
        best_cost = cost
        best_mapping = initial_mapping

"""
