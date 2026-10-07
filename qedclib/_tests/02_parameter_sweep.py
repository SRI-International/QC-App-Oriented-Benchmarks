"""
02_parameter_sweep.py — Phase diagram style parameter sweep.

Scenario: Condensed matter simulation where we sweep a coupling parameter
and measure the ground state energy proxy (expectation values from measurement).

This is representative of VQE-style or phase transition studies where you
generate many circuits with different parameter values and run them all.
"""

import qedclib
import numpy as np
import time

qedclib.initialize("qiskit")
import execute as ex

from qiskit import QuantumCircuit

# Configure backend
ex.set_execution_target(backend_id="qasm_simulator")
ex.verbose = False

NUM_SHOTS = 2000


def make_ising_circuit(n_qubits, coupling_J, field_h, trotter_steps=1):
    """
    Simple 1D transverse-field Ising model circuit.
    H = -J * sum(Z_i Z_{i+1}) - h * sum(X_i)

    Trotterized time evolution — a common circuit pattern for
    phase diagram exploration (varying J/h ratio).
    """
    qc = QuantumCircuit(n_qubits, n_qubits)
    dt = 0.5 / max(trotter_steps, 1)

    for step in range(trotter_steps):
        # ZZ interaction: CNOT - Rz - CNOT
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * coupling_J * dt, i + 1)
            qc.cx(i, i + 1)

        # Transverse field: Rx on each qubit
        for i in range(n_qubits):
            qc.rx(2 * field_h * dt, i)

    # Measure all
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def magnetization_from_counts(counts, n_qubits, num_shots):
    """Compute <Z> magnetization from measurement counts."""
    mag = 0.0
    for bitstring, count in counts.items():
        # Each bit: 0 -> +1, 1 -> -1
        z_val = sum(1 if b == '0' else -1 for b in bitstring)
        mag += z_val * count
    return mag / (num_shots * n_qubits)


# === Parameter sweep ===

n_qubits = 6
trotter_steps = 3
J_values = np.linspace(0.1, 2.0, 20)  # coupling strength
h_fixed = 1.0  # fixed transverse field

print(f"Parameter sweep: {len(J_values)} values of J, {n_qubits} qubits, {trotter_steps} Trotter steps")
print(f"Total circuits: {len(J_values)}")

# Create all circuits upfront
circuits = []
for J in J_values:
    qc = make_ising_circuit(n_qubits, J, h_fixed, trotter_steps)
    qc.name = f"ising_J{J:.2f}"
    circuits.append(qc)

# Execute all at once as a batch
print(f"\nSubmitting batch of {len(circuits)} circuits ({NUM_SHOTS} shots each)...")
t0 = time.time()
job_id, result = ex.execute_circuits(circuits, num_shots=NUM_SHOTS)
elapsed = time.time() - t0
print(f"Batch complete in {elapsed:.3f}s ({elapsed/len(circuits)*1000:.1f}ms per circuit)")

# Extract magnetization for each parameter value
counts_list = result.get_counts()
magnetizations = []
for i, counts in enumerate(counts_list):
    mag = magnetization_from_counts(counts, n_qubits, NUM_SHOTS)
    magnetizations.append(mag)

# Print results (phase diagram data)
print(f"\n{'J/h':>8}  {'<Z>':>8}  {'phase':>12}")
print("-" * 32)
for J, mag in zip(J_values, magnetizations):
    phase = "ordered" if abs(mag) > 0.3 else "disordered"
    print(f"{J/h_fixed:8.3f}  {mag:8.4f}  {phase:>12}")

# === Now try with different qubit counts (scaling study) ===
print(f"\n\n--- Scaling study: fixed J/h=1.0, varying qubit count ---")
J_fixed = 1.0

for n_q in [4, 6, 8, 10]:
    qc = make_ising_circuit(n_q, J_fixed, h_fixed, trotter_steps)
    qc.name = f"ising_{n_q}q"

    t0 = time.time()
    job_id, result = ex.execute_circuits([qc], num_shots=NUM_SHOTS)
    elapsed = time.time() - t0

    counts = result.get_counts()
    if isinstance(counts, list):
        counts = counts[0]
    mag = magnetization_from_counts(counts, n_q, NUM_SHOTS)
    depth = qc.depth()
    print(f"  {n_q:2d} qubits: <Z>={mag:+.4f}, depth={depth}, time={elapsed:.3f}s")

print("\n--- Parameter sweep complete ---")
