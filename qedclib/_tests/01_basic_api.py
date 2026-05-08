"""
01_basic_api.py — Verify qedclib works as a standalone library.

Tests: import, set_api, set_execution_target, execute_circuits, get results.
This is the simplest possible use case — just run some circuits and get counts.
"""

import qedclib
import time

# Set the API we want to use
qedclib.set_api("qiskit")

# Now we can get the execute module
qedclib.initialize("qiskit")
import execute as ex

print(f"qedclib version: {qedclib.__version__}")
print(f"API: {qedclib.get_api()}")

# Configure backend — start with local simulator
ex.set_execution_target(backend_id="qasm_simulator")
ex.verbose = True

# Build a few simple circuits
from qiskit import QuantumCircuit

circuits = []
for n_qubits in [2, 4, 6]:
    qc = QuantumCircuit(n_qubits, n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits):
        qc.measure(i, i)
    qc.name = f"hadamard_{n_qubits}q"
    circuits.append(qc)

print(f"\nCreated {len(circuits)} circuits")
for qc in circuits:
    print(f"  {qc.name}: {qc.num_qubits} qubits, {qc.size()} gates")

# Execute them all as a batch
print("\n--- Executing batch ---")
t0 = time.time()
job_id, result = ex.execute_circuits(circuits, num_shots=1000)
elapsed = time.time() - t0

print(f"\nJob ID: {job_id}")
print(f"Elapsed: {elapsed:.3f}s")

# Get counts from each circuit
counts_list = result.get_counts()
print(f"Got {len(counts_list)} result sets")

for i, counts in enumerate(counts_list):
    n_outcomes = len(counts)
    total_shots = sum(counts.values())
    print(f"  Circuit {i} ({circuits[i].name}): {n_outcomes} outcomes, {total_shots} shots")
    # Show top 3 outcomes
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for bitstring, count in sorted_counts[:3]:
        print(f"    {bitstring}: {count} ({100*count/total_shots:.1f}%)")

print("\n--- Basic API test complete ---")
