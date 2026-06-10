import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
import networkx as nx
from hardware.distance_matrx import (
    get_distance_matrix_cnot_error_cost,
    get_qubit_readout_error,
)
from tools.read_circuit import read_benchmark_circuit
from mapping.mapping_transition import _modify_dag_circuit, multiprogram_initial_mapping
from mapping.initial_mapping_wrapper import initial_mapping
from mapping.initial_mapping_construct import cost
from mapping.iterative_mapping import iterative_mapping_algorithm
from partition_process.qubit_partition import (
    partition_hardware_heuristic,
    partition_circuits,
    largest_circuit_logical_degree,
    hardware_qubit_physical_degree,
)
from qiskit.circuit import QuantumRegister


def get_simultaneous_partition(circuit_list, hardware_graph, hardware,
                                cnot_error_matrix, readout_error,
                                qubit_physical_degree, largest_physical_degree,
                                largest_logical_degrees, weight_lambda,
                                partition_method, epslon):
    """
    Replicates the fidelity-threshold selection from circuits_schedule to find
    the simultaneous partition accepted within epslon.
    Returns (accepted_circuit_list, multiple_partition) or (None, None).
    """
    circuit_list = list(circuit_list)
    largest_logical_degrees = list(largest_logical_degrees)

    independent_fidelities = []
    independent_partitions = []
    for circuit in circuit_list:
        ind = partition_circuits(
            [circuit], hardware_graph, hardware, cnot_error_matrix, readout_error,
            qubit_physical_degree, largest_physical_degree, largest_logical_degrees,
            weight_lambda, partition_method,
        )
        independent_fidelities.append(ind[0].fidelity)
        independent_partitions.append(ind)

    while len(circuit_list) > 1:
        multi = partition_circuits(
            circuit_list, hardware_graph, hardware, cnot_error_matrix, readout_error,
            qubit_physical_degree, largest_physical_degree, largest_logical_degrees,
            weight_lambda, partition_method,
        )
        if not multi:
            circuit_list.pop()
            largest_logical_degrees.pop()
            independent_fidelities.pop()
            continue

        fidelity_multi = sum(p.fidelity for p in multi)
        fidelity_indep = sum(independent_fidelities)
        if abs(fidelity_indep - fidelity_multi) < epslon:
            return circuit_list, multi

        circuit_list.pop()
        largest_logical_degrees.pop()
        independent_fidelities.pop()

    return None, None


def compute_initial_mapping(circuit_list, hardware, multiple_partition):
    """
    Compute the combined initial mapping for the simultaneous circuits,
    stopping before SWAP routing.
    Returns (circuit_partitions, per_circuit_mappings, merged_mapping).
    """
    circuit_partitions = [list(part.value) for part in multiple_partition]
    total_logical_qubits = sum(len(p) for p in circuit_partitions)
    shared_qreg = QuantumRegister(total_logical_qubits, 'q')

    circuit_initial_mapping = {}
    per_circuit_mappings = []
    update_circuits = []
    previous_qubit_used = 0

    for index, circuit in enumerate(circuit_list):
        updated = _modify_dag_circuit(circuit, previous_qubit_used, shared_qreg)
        update_circuits.append(updated)
        mapping = initial_mapping(
            updated, hardware, circuit_partitions[index],
            iterative_mapping_algorithm, cost, "sabre", 10,
            circuit_initial_mapping,
        )
        per_circuit_mappings.append(mapping)
        previous_qubit_used += len(circuit_partitions[index])

    merged_mapping = multiprogram_initial_mapping(update_circuits, per_circuit_mappings)
    return circuit_partitions, per_circuit_mappings, merged_mapping


def print_initial_mapping(circuit_list, circuit_partitions, per_circuit_mappings, merged_mapping):
    """
    Per-circuit: filter to only the active qubits that belong to this circuit's
    slice of shared_qreg (indices [offset, offset+n)), sort by qubit index, and
    print as a compact list so position i = physical qubit for logical qubit i.

    Combined: same compact list over the full shared_qreg.
    """
    print("\n=== Per-circuit initial mappings ===")
    offset = 0
    for idx, circuit in enumerate(circuit_list):
        n = len(circuit_partitions[idx])
        mapping = per_circuit_mappings[idx]
        # Qubits for this circuit occupy shared_qreg[offset : offset+n]
        active_physical = [
            phys for q, phys in sorted(mapping.items(), key=lambda kv: kv[0]._index)
            if offset <= q._index < offset + n
        ]
        print(f"  {circuit.name}:")
        print(f"    partition : {circuit_partitions[idx]}")
        print(f"    mapping   : {active_physical}  (index i -> physical qubit for logical qubit i)")
        offset += n

    combined = [
        phys for _, phys in sorted(merged_mapping.items(), key=lambda kv: kv[0]._index)
    ]
    print(f"\n=== Combined initial mapping ===")
    print(f"  {combined}  (index i -> physical qubit for shared logical qubit i)")


if __name__ == '__main__':
    hardware = IBMQHardwareArchitecture("ibmq_manhattan")
    hardware_graph = nx.DiGraph(hardware._coupling_graph)
    cnot_error_matrix = get_distance_matrix_cnot_error_cost(hardware)
    readout_error = get_qubit_readout_error(hardware)

    circuits = []
    for name in ["3_17_13", "4mod5-v1_22", "mod5mils_65", "alu-v0_27", "decod24-v2_43"]:
        c = read_benchmark_circuit(name)
        c.name = name
        circuits.append(c)

    epslon = 0.1
    weight_lambda = 2
    partition_method = partition_hardware_heuristic

    # Sort by descending CNOT density, same as circuits_schedule
    circuits = sorted(
        circuits,
        key=lambda x: x.count_ops().get("cx", 0) / x.num_qubits,
        reverse=True,
    )

    # Build the list of circuits that fit on the hardware
    circuit_list = []
    qubit_sum = 0
    for circuit in circuits:
        qubit_sum += circuit.cregs[0].size
        if qubit_sum <= hardware.qubit_number:
            circuit_list.append(circuit)
        else:
            break

    qubit_physical_degree, largest_physical_degree = hardware_qubit_physical_degree(hardware)
    largest_logical_degrees = [largest_circuit_logical_degree(c) for c in circuit_list]

    print(f"Circuits considered for simultaneous execution: {[c.name for c in circuit_list]}")

    accepted_circuits, multiple_partition = get_simultaneous_partition(
        circuit_list, hardware_graph, hardware,
        cnot_error_matrix, readout_error,
        qubit_physical_degree, largest_physical_degree,
        largest_logical_degrees, weight_lambda,
        partition_method, epslon,
    )

    if multiple_partition is None:
        print("No simultaneous partition found within the fidelity threshold.")
        sys.exit(0)

    print(f"\nAccepted simultaneous circuits: {[c.name for c in accepted_circuits]}")

    circuit_partitions, per_circuit_mappings, merged_mapping = compute_initial_mapping(
        accepted_circuits, hardware, multiple_partition,
    )

    print_initial_mapping(accepted_circuits, circuit_partitions, per_circuit_mappings, merged_mapping)
