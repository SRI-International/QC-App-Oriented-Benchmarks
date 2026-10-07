# Fire Opal executor interface
#
# This module implements the "executor" intercept for execution on IBM devices using Q-CTRL's Fire
# Opal error suppression software.
import time
import fireopal
from qiskit import QuantumCircuit

verbose = False


class FireOpalResult:
    def __init__(self, counts: dict[str, float], exec_time: float) -> None:
        self.counts = counts
        self.exec_time = exec_time

    def get_counts(self, qc=None) -> dict[str, float]:
        return self.counts


class FireOpalBackend:
    def __init__(
        self,
        ibm_backend_id: str,
        hub: str,
        group: str,
        project: str,
        token: str,
    ) -> None:
        self.ibm_backend_id = ibm_backend_id
        self.credentials = fireopal.credentials.make_credentials_for_ibmq(
            token=token, hub=hub, group=group, project=project,
        )

    def name(self) -> str:
        return f"Fire Opal ({self.ibm_backend_id})"

    def run(self, circuit: QuantumCircuit, shots: int) -> dict[str, float]:
        qasm_circuit = circuit.qasm()
        # Validate circuit
        validation_errors = fireopal.validate(
            circuits=[qasm_circuit], credentials=self.credentials, backend_name=self.ibm_backend_id,
        )["results"]
        if len(validation_errors) > 0:
            raise ValueError("The circuit failed validation for the given backend.")
        # Run circuit
        execution_results = fireopal.execute(
            circuits=[qasm_circuit],
            shot_count=shots,
            credentials=self.credentials,
            backend_name=self.ibm_backend_id,
        )
        return execution_results["results"][0]


def run(circuit, backend_name, backend, shots=100, **backend_exec_options) -> FireOpalResult:
    if verbose:
        print(f"   ... executing circuit on backend={backend_name}")

    start_time = time.time()
    fire_opal_counts = backend.run(circuit=circuit, shots=shots)
    exec_time = time.time() - start_time

    if verbose:
        print(f"... result = {fire_opal_counts}")

    return FireOpalResult(counts=fire_opal_counts, exec_time=exec_time)
