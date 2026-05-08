"""
04_batch_scaling.py — Test batch execution at scale.

Scenario: ML-style workload where you have hundreds of parameterized circuits
(e.g., a variational classifier evaluating many data points).

Tests:
  - How does execution time scale with batch size?
  - Memory behavior with large batches
  - Does batched_run (if available) help vs. one big execute_circuits call?
"""

import qedclib
import numpy as np
import time

qedclib.set_api("qiskit")
qedclib.initialize("qiskit")
import execute as ex

from qiskit import QuantumCircuit

ex.set_execution_target(backend_id="qasm_simulator")
ex.verbose = False

NUM_SHOTS = 500


def make_variational_circuit(n_qubits, params):
    """
    Simple variational circuit — the kind used in VQE, QAOA, or
    quantum ML classifiers. Each circuit has different rotation angles.
    """
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Feature map layer
    for i in range(n_qubits):
        qc.ry(params[i % len(params)], i)

    # Entangling layer
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # Variational layer
    for i in range(n_qubits):
        qc.rz(params[(i + 1) % len(params)], i)
        qc.ry(params[(i + 2) % len(params)], i)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run_batch_test(n_circuits, n_qubits):
    """Create and execute a batch, return timing info."""
    np.random.seed(42)

    # Generate random parameter sets (like different training data points)
    circuits = []
    for j in range(n_circuits):
        params = np.random.uniform(0, 2 * np.pi, size=n_qubits + 2)
        qc = make_variational_circuit(n_qubits, params)
        qc.name = f"var_{j}"
        circuits.append(qc)

    t_create = time.time()
    create_time = t_create - t_create  # already done above

    t0 = time.time()
    job_id, result = ex.execute_circuits(circuits, num_shots=NUM_SHOTS)
    exec_time = time.time() - t0

    counts_list = result.get_counts()
    assert len(counts_list) == n_circuits, f"Expected {n_circuits} results, got {len(counts_list)}"

    return {
        "n_circuits": n_circuits,
        "exec_time": exec_time,
        "per_circuit_ms": exec_time / n_circuits * 1000,
    }


# === Scaling test: vary batch size ===
print("Batch size scaling test")
print(f"{'batch':>8}  {'qubits':>6}  {'total(s)':>9}  {'per-circuit(ms)':>15}")
print("-" * 45)

n_qubits = 6
for batch_size in [10, 50, 100, 200, 500]:
    r = run_batch_test(batch_size, n_qubits)
    print(f"{r['n_circuits']:8d}  {n_qubits:6d}  {r['exec_time']:9.3f}  {r['per_circuit_ms']:15.1f}")


# === Scaling test: vary qubit count at fixed batch size ===
print(f"\n\nQubit count scaling (batch=100)")
print(f"{'qubits':>6}  {'total(s)':>9}  {'per-circuit(ms)':>15}")
print("-" * 35)

batch_size = 100
for n_q in [4, 8, 12]:
    r = run_batch_test(batch_size, n_q)
    print(f"{n_q:6d}  {r['exec_time']:9.3f}  {r['per_circuit_ms']:15.1f}")


# === Test: what happens if we submit individual circuits vs. batch? ===
print(f"\n\nSingle vs batch submission (100 circuits, {n_qubits} qubits)")

np.random.seed(42)
circuits = []
for j in range(100):
    params = np.random.uniform(0, 2 * np.pi, size=n_qubits + 2)
    qc = make_variational_circuit(n_qubits, params)
    circuits.append(qc)

# Batch
t0 = time.time()
job_id, result = ex.execute_circuits(circuits, num_shots=NUM_SHOTS)
batch_time = time.time() - t0

# One-at-a-time
t0 = time.time()
for qc in circuits:
    job_id, result = ex.execute_circuits([qc], num_shots=NUM_SHOTS)
serial_time = time.time() - t0

print(f"  Batch (1 call):     {batch_time:.3f}s")
print(f"  Serial (100 calls): {serial_time:.3f}s")
print(f"  Speedup:            {serial_time/batch_time:.1f}x")

print("\n--- Batch scaling test complete ---")
