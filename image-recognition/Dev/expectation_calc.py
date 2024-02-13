from typing import Dict, List, Optional

from qiskit import Aer, execute
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.opflow import ComposedOp, PauliExpectation, StateFn, SummedOp
from qiskit.result import sampled_expectation_value
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.quantum_info import SparsePauliOp


qasm_backend = Aer.get_backend("qasm_simulator")
def get_measured_qubits(circuit: QuantumCircuit) -> List[int]:
    """
    Get a list of indices of the qubits being measured in a given quantum circuit.
    """
    measured_qubits = []

    for gate, qubits, clbits in circuit.data:
        if gate.name == "measure":
            measured_qubits.extend([qubit.index for qubit in qubits])

    measured_qubits = sorted(list(set(measured_qubits)))

    return measured_qubits

def expectation_run(circuit: QuantumCircuit, shots: Optional[int] = None) -> Dict[str, float]:
        """Run a quantum circuit on the noise-free simulator and return the probabilities."""

        # Refactored error check
        # if circuit.num_parameters != 0:
            # raise QiskitError(ErrorMessages.UNDEFINED_PARAMS.value)
        if len(get_measured_qubits(circuit)) == 0:
            circuit.measure_all()

        if isinstance(shots, int):
            counts = (execute(circuit, backend= qasm_backend, shots=shots).result().get_counts())
            probs = normalize_counts(counts, num_qubits=circuit.num_qubits)
        return probs

def normalize_counts(counts, num_qubits=None):
    """
    Normalize the counts to get probabilities and convert to bitstrings.
    """
    normalizer = sum(counts.values())

    try:
        dict({str(int(key, 2)): value for key, value in counts.items()})
        if num_qubits is None:
            num_qubits = max(len(key) for key in counts)
        bitstrings = {key.zfill(num_qubits): value for key, value in counts.items()}
    except ValueError:
        bitstrings = counts

    probabilities = dict({key: value / normalizer for key, value in bitstrings.items()})
    assert abs(sum(probabilities.values()) - 1) < 1e-9
    return probabilities


    
def prepare_circuits(base_circuit, observables):
    """
    Prepare the qubit-wise commuting circuits for a given operator.
    """
    if isinstance(observables, ComposedOp):
        observables = SummedOp([observables])    
    circuits = list()
    
    for obs in observables:
        circuit = base_circuit.copy()
        circuit.append(obs[1], qargs=list(range(base_circuit.num_qubits)))
        circuits.append(circuit) 
    return circuits, observables

def compute_probabilities(circuits, shots=None):
    probabilities = list()
    for circuit in circuits:
        probability = expectation_run(circuit, shots)
        probabilities.append(probability)
    return probabilities

def calculate_expectation_values(probabilities, observables):
    """
    Return the expectation values for an operator given the probabilities.
    """
    expectation_values = list()
    for idx, op in enumerate(observables):
        expectation_value = sampled_expectation_value(probabilities[idx], op[0].primitive)
        expectation_values.append(expectation_value)
    return expectation_values


# main function
def calculate_expectation(base_circuit, shots=None , num_qubits=None):
     
    if  num_qubits == 4:
        operator = PauliSumOp(SparsePauliOp("Z" * num_qubits))

    elif num_qubits == 8:
        pauli_x = PauliSumOp(SparsePauliOp("X" * num_qubits))
        pauli_y = PauliSumOp(SparsePauliOp("Y" * num_qubits))
        pauli_z = PauliSumOp(SparsePauliOp("Z" * num_qubits))
        operator = 0.5 * pauli_x + 0.3 * pauli_y - 0.7 * pauli_z
    
    measurable_expression = StateFn(operator, is_measurement=True)
    # print("measurable_expression",measurable_expression)
    observables = PauliExpectation().convert(measurable_expression)
    circuits, formatted_observables = prepare_circuits(base_circuit, observables)
    probabilities = compute_probabilities(circuits, shots)
    expectation_values = calculate_expectation_values(probabilities, formatted_observables)
    
    return sum(expectation_values)
