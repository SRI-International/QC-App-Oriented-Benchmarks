"""
03_backend_switching.py — Run the same circuits on different backends.

Usage:
    python 03_backend_switching.py                     # local qasm_simulator (default)
    python 03_backend_switching.py -b qasm_simulator
    python 03_backend_switching.py -b statevector_simulator
    python 03_backend_switching.py -b ibm              # least-busy IBM backend (ibm_cloud)
    python 03_backend_switching.py -b ibm_sherbrooke   # specific IBM backend
    python 03_backend_switching.py -b ionq              # IonQ simulator (default)
    python 03_backend_switching.py -b ionq_qpu         # IonQ QPU (requires access)
    python 03_backend_switching.py -b iqm              # IQM Garnet via Resonance

Environment variables for hardware backends:
    IBM:  credentials saved via QiskitRuntimeService.save_account()
    IonQ: QISKIT_IONQ_API_TOKEN
    IQM:  IQM_API_TOKEN
"""

import qedclib
import argparse
import os
import time

qedclib.initialize("qiskit")
import execute as ex

from qiskit import QuantumCircuit

NUM_SHOTS = 1000


def make_test_circuits():
    """Create a small set of circuits for cross-backend comparison."""
    circuits = []

    # GHZ state — entanglement test
    for n in [3, 5, 8]:
        qc = QuantumCircuit(n, n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n), range(n))
        qc.name = f"ghz_{n}q"
        circuits.append(qc)

    # Random circuit — stress test
    import numpy as np
    np.random.seed(42)
    n = 6
    qc = QuantumCircuit(n, n)
    for _ in range(10):
        q = np.random.randint(n)
        gate = np.random.choice(['h', 'x', 't', 's'])
        getattr(qc, gate)(q)
        if np.random.random() > 0.5:
            q2 = (q + 1) % n
            qc.cx(q, q2)
    qc.measure(range(n), range(n))
    qc.name = "random_6q"
    circuits.append(qc)

    return circuits


def configure_backend(backend_id):
    """Configure execution target based on backend_id string."""

    # IBM backends: "ibm" for least-busy, or specific like "ibm_sherbrooke"
    # Uses IBM Cloud channel; set IBM_INSTANCE env var to your CRN or service name
    if backend_id.startswith("ibm"):
        ibm_instance = os.environ.get("IBM_INSTANCE", "")
        if not ibm_instance:
            print("  WARNING: IBM_INSTANCE not set — will use saved default credentials.")
            print("  Set IBM_INSTANCE to your CRN or service name to target a specific plan.")

        if backend_id == "ibm":
            # Find least-busy backend
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(channel="ibm_cloud", instance=ibm_instance)
            backend = service.least_busy(simulator=False, operational=True, min_num_qubits=100)
            backend_id = backend.name
            print(f"  Least-busy IBM backend: {backend_id}")

        ex.set_execution_target(
            backend_id=backend_id,
            project=ibm_instance,
            exec_options={"use_ibm_quantum_platform": False, "use_sessions": False},
        )
        return backend_id

    # IonQ backends: "ionq" uses simulator, "ionq_qpu" for real hardware
    if backend_id.startswith("ionq"):
        from qiskit_ionq import IonQProvider
        provider = IonQProvider()
        # Default to simulator — IonQ QPU requires explicit request
        ionq_backend = "ionq_simulator" if backend_id == "ionq" else backend_id
        provider_backend = provider.get_backend(ionq_backend)

        ex.set_execution_target(
            backend_id=backend_id,
            provider_backend=provider_backend,
        )
        return backend_id

    # IQM backends: "iqm" uses Garnet via Resonance
    if backend_id.startswith("iqm"):
        from iqm.qiskit_iqm import IQMProvider

        iqm_server_url = "https://resonance.meetiqm.com"
        quantum_computer = "garnet"
        iqm_token = os.environ.get("IQM_API_TOKEN")

        provider = IQMProvider(iqm_server_url, quantum_computer=quantum_computer, token=iqm_token)
        provider_backend = provider.get_backend()

        ex.set_execution_target(
            backend_id=quantum_computer,
            provider_backend=provider_backend,
        )
        return quantum_computer

    # Local simulators: qasm_simulator, statevector_simulator, etc.
    ex.set_execution_target(backend_id=backend_id)
    return backend_id


def print_results(circuits, counts_list):
    """Print top outcomes for each circuit."""
    # Normalize: get_counts() returns dict for single circuit, list for multiple
    if isinstance(counts_list, dict):
        counts_list = [counts_list]

    for i, counts in enumerate(counts_list):
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k}:{v}" for k, v in top)
        print(f"  {circuits[i].name}: {top_str}")


# === Main ===

parser = argparse.ArgumentParser(description="Test qedclib on different backends")
parser.add_argument("-b", "--backend", default="qasm_simulator",
                    help="Backend to use (default: qasm_simulator)")
args = parser.parse_args()

circuits = make_test_circuits()
print(f"Test circuits: {[qc.name for qc in circuits]}")

print(f"\n{'='*50}")
print(f"Backend: {args.backend}")
print(f"{'='*50}")

try:
    actual_id = configure_backend(args.backend)
    ex.verbose = True

    t0 = time.time()
    job_id, result = ex.execute_circuits(circuits, num_shots=NUM_SHOTS)
    elapsed = time.time() - t0

    counts_list = result.get_counts()
    print(f"\nExecuted {len(circuits)} circuits in {elapsed:.3f}s "
          f"({elapsed/len(circuits)*1000:.1f}ms/circuit)")

    print_results(circuits, counts_list)

except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Backend test complete ---")
