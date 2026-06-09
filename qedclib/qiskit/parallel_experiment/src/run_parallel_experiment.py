import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit_experiments.framework import (
    BaseExperiment,
    BaseAnalysis,
    ParallelExperiment,
)

from hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
from hardware.distance_matrx import (
    get_distance_matrix_cnot_error_cost,
    get_qubit_readout_error,
)
from tools.read_circuit import read_benchmark_circuit
from partition_process.qubit_partition import (
    partition_hardware_heuristic,
    largest_circuit_logical_degree,
    hardware_qubit_physical_degree,
)
from find_initial_mapping import get_simultaneous_partition, compute_initial_mapping


def strip_idle_qubits(circuit):
    """Remove idle qubits from a circuit (QASM files define qreg q[65] but
    only use a few qubits). Returns a compact circuit with only active qubits."""
    active = set()
    for inst in circuit.data:
        for qubit in inst.qubits:
            active.add(circuit.find_bit(qubit).index)

    if len(active) == circuit.num_qubits:
        return circuit

    active_sorted = sorted(active)
    remap = {old: new for new, old in enumerate(active_sorted)}

    new_qc = QuantumCircuit(len(active_sorted), name=circuit.name)
    for inst in circuit.data:
        if inst.clbits:
            continue
        new_qubits = [
            new_qc.qubits[remap[circuit.find_bit(q).index]] for q in inst.qubits
        ]
        new_qc.append(inst.operation, new_qubits)

    return new_qc


class NoAnalysis(BaseAnalysis):
    def _run_analysis(self, experiment_data):
        return [], []


class CircuitExperiment(BaseExperiment):
    def __init__(self, circuit, physical_qubits, label):
        super().__init__(
            physical_qubits=physical_qubits,
            analysis=NoAnalysis(),
            backend=None,
        )
        self._circuit = circuit.remove_final_measurements(inplace=False)
        self._label = label

    def circuits(self):
        qc = self._circuit.copy()
        qc.name = self._label
        qc.measure_all()
        qc.metadata = {
            "component": self._label,
            "physical_qubits": self.physical_qubits,
        }
        return [qc]


def extract_physical_qubits(circuit_list, circuit_partitions, per_circuit_mappings):
    """
    Extract ordered physical qubit assignments for each circuit.
    Returns a list of tuples: physical_qubits[i][j] = physical qubit for
    logical qubit j of circuit i.
    """
    result = []
    offset = 0
    for idx in range(len(circuit_list)):
        n = len(circuit_partitions[idx])
        mapping = per_circuit_mappings[idx]
        active_physical = [
            phys
            for q, phys in sorted(mapping.items(), key=lambda kv: kv[0]._index)
            if offset <= q._index < offset + n
        ]
        result.append(tuple(active_physical))
        offset += n
    return result


if __name__ == "__main__":
    # ---- hardware setup ----
    hardware = IBMQHardwareArchitecture("ibmq_manhattan")
    hardware_graph = nx.DiGraph(hardware._coupling_graph)
    cnot_error_matrix = get_distance_matrix_cnot_error_cost(hardware)
    readout_error = get_qubit_readout_error(hardware)

    # ---- load benchmark circuits ----
    circuit_names = [
        "3_17_13", "4mod5-v1_22", "mod5mils_65", 
        # "alu-v0_27", "decod24-v2_43",
    ]
    circuits = []
    for name in circuit_names:
        c = read_benchmark_circuit(name)
        c.name = name
        circuits.append(c)

    epslon = 0.1
    weight_lambda = 2
    partition_method = partition_hardware_heuristic

    # sort by descending CNOT density (same ordering as circuits_schedule)
    circuits = sorted(
        circuits,
        key=lambda x: x.count_ops().get("cx", 0) / x.cregs[0].size,
        reverse=True,
    )

    # keep circuits that fit on hardware
    circuit_list = []
    qubit_sum = 0
    for circuit in circuits:
        qubit_sum += circuit.cregs[0].size
        if qubit_sum <= hardware.qubit_number:
            circuit_list.append(circuit)
        else:
            break

    qubit_physical_degree, largest_physical_degree = hardware_qubit_physical_degree(
        hardware
    )
    largest_logical_degrees = [
        largest_circuit_logical_degree(c) for c in circuit_list
    ]

    print(f"Circuits considered: {[c.name for c in circuit_list]}")

    # ---- partition and mapping ----
    accepted_circuits, multiple_partition = get_simultaneous_partition(
        circuit_list,
        hardware_graph,
        hardware,
        cnot_error_matrix,
        readout_error,
        qubit_physical_degree,
        largest_physical_degree,
        largest_logical_degrees,
        weight_lambda,
        partition_method,
        epslon,
    )

    if multiple_partition is None:
        print("No simultaneous partition found within the fidelity threshold.")
        sys.exit(0)

    print(f"Accepted circuits: {[c.name for c in accepted_circuits]}")

    circuit_partitions, per_circuit_mappings, merged_mapping = compute_initial_mapping(
        accepted_circuits, hardware, multiple_partition,
    )

    physical_qubits_per_circuit = extract_physical_qubits(
        accepted_circuits, circuit_partitions, per_circuit_mappings,
    )

    # ---- print mapping info ----
    print("\n" + "=" * 80)
    print("Mapping from find_initial_mapping:")
    for idx, circuit in enumerate(accepted_circuits):
        pq = physical_qubits_per_circuit[idx]
        print(f"  {circuit.name}:")
        print(f"    partition        : {circuit_partitions[idx]}")
        print(f"    physical qubits  : {list(pq)}")
        for j, p in enumerate(pq):
            print(f"      logical q[{j}] -> physical q[{p}]")

    # ---- create a simulated backend matching the hardware topology ----
    cmap = CouplingMap(hardware._coupling_graph)
    backend = GenericBackendV2(
        num_qubits=cmap.size(),
        coupling_map=cmap,
        basis_gates=["id", "rz", "sx", "x", "cx"],
        seed=42,
    )

    # ---- build and run ParallelExperiment ----
    # Strip idle qubits: QASM files define qreg q[65] but only use a few
    compact_circuits = [strip_idle_qubits(c) for c in accepted_circuits]

    experiments = [
        CircuitExperiment(
            circuit=compact_circuits[idx],
            physical_qubits=physical_qubits_per_circuit[idx],
            label=circuit.name,
        )
        for idx, circuit in enumerate(accepted_circuits)
    ]

    parallel = ParallelExperiment(
        experiments=experiments,
        backend=backend,
        flatten_results=False,
    )

    composite = parallel.circuits()[0]
    print("\n" + "=" * 80)
    print("Combined ParallelExperiment circuit:")
    print(composite.draw())

    expdata = parallel.run(backend=backend, shots=1024)
    expdata.block_for_results()

    # ---- results ----
    print("\n" + "=" * 80)
    print("Split results per circuit:")

    for i, child in enumerate(expdata.child_data()):
        name = accepted_circuits[i].name
        pq = physical_qubits_per_circuit[i]

        print(f"\nCircuit {name}")
        print(f"  Partition: {circuit_partitions[i]}")
        print(f"  Mapping:")
        for j, p in enumerate(pq):
            print(f"    logical q[{j}] -> physical q[{p}]")

        datum = child.data(0)
        print(f"  Counts: {datum.get('counts', datum)}")
