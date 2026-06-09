from qiskit import QuantumCircuit, IBMQ
from HA.src.hamap import (
    ha_mapping,
    ha_mapping_paper_compliant,
    IBMQHardwareArchitecture,
)
from IBMQSubmitter import IBMQSubmitter
from initial_mapping_wrapper import initial_mapping

circuit = QuantumCircuit.from_qasm_file('/home/siyuan/Seafile/Qubit mapping problem bibliography/SABRE/sabre_dynamic_test/sabre_distance_bridge/test/examples/4mod5-v1_22.qasm')
hardware = IBMQHardwareArchitecture("ibmq_16_melbourne")
#initial_mapping = {qubit: i for i,qubit in enumerate(circuit.qubits)}
# initial_mapping = {}
#
# mapping = [3, 1, 2, 25, 10, 26, 17, 7, 18, 8, 0, 5, 19, 13, 23, 11, 9, 20, 14, 24, 22, 16, 21, 4, 6, 12, 15]

def cost(initial_mapping, quantum_circuit: QuantumCircuit, hardware):
    mapped_circuit, final_mapping = ha_mapping(
        quantum_circuit, initial_mapping, hardware
    )

    ops_num = (
        mapped_circuit.count_ops().get("cx", 0)
        + mapped_circuit.count_ops().get("swap", 0) * 3
    )
    # print("ops number is", ops_num)
    return ops_num

def cost_gate_num(quantum_circuit: QuantumCircuit):
    cx_num = quantum_circuit.count_ops().get("cx", 0)
    swap_num = quantum_circuit.count_ops().get("swap", 0) * 3
    ops_num = (
        cx_num + swap_num
    )
    return ops_num

computed_initial_mapping = initial_mapping(
    circuit, hardware, ha_mapping, cost, "sabre", 100
)
if isinstance(computed_initial_mapping, dict):
    computed_initial_mapping = [i for i in computed_initial_mapping.values()]
print(computed_initial_mapping)

final_initial_mapping = {}

for i,qubit in enumerate(circuit.qubits):
    final_initial_mapping[qubit] = computed_initial_mapping[i]


mapped_circuit, final_mapping = ha_mapping(
    circuit, final_initial_mapping, hardware
)

additional_gate = cost_gate_num(mapped_circuit) - cost_gate_num(circuit)

print(f"additional circuit is {additional_gate}")

#quit()
print("Loading account...")
IBMQ.load_account()
provider = IBMQ.get_provider(
    hub="ibm-q-france", group="univ-montpellier", project="default"
)

backend = provider.get_backend("ibmq_16_melbourne")
print(f"Running on {backend.name()}.")

submitter = IBMQSubmitter(backend, tags=["4mod5-v1_22","ha_mapping"])

mapped_circuit.measure_active()
submitter.add_circuit(
    mapped_circuit, computed_initial_mapping, backend
)


# qc = result_circuits[0]
# qc.measure_active()
# submitter.add_circuit(
#     qc, [4, 9, 3, 11, 2, 1, 8, 12, 10, 13, 7, 14, 0, 6, 5], backend
# )

print(f"Submitting {len(submitter)} circuits...")
submitter.submit()
print("Done! Saving...")
