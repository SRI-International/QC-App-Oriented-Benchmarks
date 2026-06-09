# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (03/2020)
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
import typing as ty
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit.quantumregister import Qubit

from hamap._circuit_manipulation import add_qubits_to_quantum_circuit
from hamap.hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
from hamap.initial_mapping import (
    get_random_mapping,
    initial_mapping_from_iterative_forward_backward,
)
from hamap.mapping import ha_mapping


def read_benchmark_circuit(category: str, name: str) -> QuantumCircuit:
    src_folder = Path(__file__).parent.parent.parent
    benchmark_folder = src_folder.parent / "benchmark"
    return QuantumCircuit.from_qasm_file(
        benchmark_folder / "circuits" / category / f"{name}.qasm"
    )


def mapping_algorithm(
    quantum_circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    mapping: ty.Dict[Qubit, int],
):
    return ha_mapping(quantum_circuit, mapping, hardware)


def test_iterated_sabre(
    quantum_circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    iteration_number: int,
    initial_mapping: ty.Optional[ty.Dict[Qubit, int]] = None,
) -> int:
    quantum_circuit = add_qubits_to_quantum_circuit(quantum_circuit, hardware)
    circuit, mapping = initial_mapping_from_iterative_forward_backward(
        quantum_circuit,
        hardware,
        mapping_algorithm=mapping_algorithm,
        maximum_mapping_procedure_calls=iteration_number,
        initial_mapping=initial_mapping,
    )
    return circuit.count_ops().get("swap", 0)


def main():
    N = 10
    hardware = IBMQHardwareArchitecture.load("ibmq_16_melbourne")
    circuit = add_qubits_to_quantum_circuit(
        read_benchmark_circuit("sabre", "ising_model_10"), hardware
    )
    initial_mapping = get_random_mapping(circuit)

    for i in range(1, N):
        print(
            f"Swap number {i}: {test_iterated_sabre(circuit, hardware, i, initial_mapping)}"
        )


main()
