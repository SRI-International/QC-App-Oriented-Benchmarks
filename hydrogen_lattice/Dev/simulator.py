"""Class for local, noise-free simulators."""
from typing import Dict, List, Optional

from qiskit import Aer, execute
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.opflow import ComposedOp, PauliExpectation, StateFn, SummedOp
from qiskit.quantum_info import Statevector
from qiskit.result import sampled_expectation_value

from enum import Enum


class Backend(Enum):
    STATEVECTOR_SIMULATOR = "statevector_simulator"
    QASM_SIMULATOR = "qasm_simulator"


class ErrorMessages(Enum):
    UNDEFINED_PARAMS = "Parameters undefined. You need to bind/assign parameters before executing."
    UNRECOGNIZED_SHOTS = "shots: {shots} not understood"


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


class Simulator:
    """Class for the noise-free simulator."""

    def __init__(self) -> None:
        """Initialize the noise-free simulator."""
        super().__init__()
        self.statevector_backend = Aer.get_backend(Backend.STATEVECTOR_SIMULATOR.value)
        self.qasm_backend = Aer.get_backend(Backend.QASM_SIMULATOR.value)

    def run(self, circuit: QuantumCircuit, shots: Optional[int] = None) -> Dict[str, float]:
        """Run a quantum circuit on the noise-free simulator and return the probabilities."""

        # Refactored error check
        if circuit.num_parameters != 0:
            raise QiskitError(ErrorMessages.UNDEFINED_PARAMS.value)

        if len(get_measured_qubits(circuit)) == 0:
            circuit.measure_all()

        if shots is None:
            measured_qubits = get_measured_qubits(circuit)
            statevector = (
                execute(
                    circuit.remove_final_measurements(inplace=False),
                    backend=self.statevector_backend,
                )
                .result()
                .get_statevector()
            )
            probs = Statevector(statevector).probabilities_dict(qargs=measured_qubits)
        elif isinstance(shots, int):
            counts = execute(circuit, backend=self.qasm_backend, shots=shots).result().get_counts()
            probs = self.normalize_counts(counts, num_qubits=circuit.num_qubits)
        else:
            raise TypeError(ErrorMessages.UNRECOGNIZED_SHOTS.value.format(shots=shots))

        return probs

    @staticmethod
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

    def compute_expectation(self, base_circuit, operator, parameters=None, shots=None):
        """
        Compute the expected value of the operator given a base-circuit and pauli operator.

        No checks are made to ensure consistency between operator and base circuit.

        Parameters
        ----------
        base_circuit : :obj:`QuantumCircuit`
            Base circuit in computational basis. Basis rotation gates will be appended as needed given the operator.
        operator : :obj:`PauliSumOp`
            Operator expressed as a sum of Pauli operators. This is assumed to be consistent with the base circuit.
        parameters : :obj:`Optional[Union[List, ndarray]]`
            Optional parameters to pass in if the circuit is parameterized
        """
        if parameters is not None and base_circuit.num_parameters != len(parameters):
            raise ValueError(f"Circuit has {base_circuit.num_parameters} but parameter length is {len(parameters)}.")

        measurable_expression = StateFn(operator, is_measurement=True)
        observables = PauliExpectation().convert(measurable_expression)
        circuits, formatted_observables = self._prepare_circuits(base_circuit, observables)
        probabilities = self._compute_probabilities(circuits, parameters, shots)
        expectation_values = self._calculate_expectation_values(probabilities, formatted_observables)
        return sum(expectation_values)

    @staticmethod
    def _prepare_circuits(base_circuit, observables):
        """
        Prepare the qubit-wise commuting circuits for a given operator.
        """
        circuits = list()

        if isinstance(observables, ComposedOp):
            observables = SummedOp([observables])
        for obs in observables:
            circuit = base_circuit.copy()
            circuit.append(obs[1], qargs=list(range(base_circuit.num_qubits)))
            circuits.append(circuit)
        return circuits, observables

    def _compute_probabilities(self, circuits, parameters=None, shots=None):
        """
        Compute the probabilities for a list of circuits with given parameters.
        """
        probabilities = list()
        for my_circuit in circuits:
            if parameters is not None:
                circuit = my_circuit.assign_parameters(parameters, inplace=False)
            else:
                circuit = my_circuit.copy()

            result = self.run(circuit, shots)
            probabilities.append(result)

        return probabilities

    @staticmethod
    def _calculate_expectation_values(probabilities, observables):
        """
        Return the expectation values for an operator given the probabilities.
        """
        expectation_values = list()
        for idx, op in enumerate(observables):
            expectation_value = sampled_expectation_value(probabilities[idx], op[0].primitive)
            expectation_values.append(expectation_value)

        return expectation_values
