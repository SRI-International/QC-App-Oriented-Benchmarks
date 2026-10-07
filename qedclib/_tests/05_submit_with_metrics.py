"""
05_submit_with_metrics.py — Demonstrate submit_circuits with automatic metrics.

This shows the higher-level API flow for standalone qedclib usage:
    init → submit_circuits (auto-stores timing) → finalize → get metrics

Unlike execute_circuits (raw execution, no metrics), submit_circuits
handles batching, timing extraction, and metrics storage automatically.
Circuits are organized as a nested dict {group: {circuit_id: qc}},
and metrics are keyed to match that structure.

Usage:
    python 05_submit_with_metrics.py
"""

import qedclib
from qedclib import metrics

qedclib.initialize("qiskit")
import execute as ex

from qiskit import QuantumCircuit

# Configure backend
ex.set_execution_target(backend_id="qasm_simulator")
ex.verbose = True

NUM_SHOTS = 1000


def make_circuits_dict():
    """
    Build circuits as a nested dict {group: {circuit_id: qc}}.
    Groups typically correspond to qubit widths.
    """
    circuits = {}
    for n_qubits in [4, 6, 8, 10]:
        group = str(n_qubits)
        circuits[group] = {}
        for circuit_id in range(3):   # 3 circuits per group
            qc = QuantumCircuit(n_qubits, n_qubits)
            qc.h(0)
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            qc.measure(range(n_qubits), range(n_qubits))
            qc.name = f"ghz_{n_qubits}q_{circuit_id}"
            circuits[group][str(circuit_id)] = qc
    return circuits


# === Submit and collect metrics ===

circuits = make_circuits_dict()
total = sum(len(g) for g in circuits.values() if isinstance(g, dict))
print(f"Created {total} circuits in {len(circuits)} groups")

# submit_circuits auto-calls metrics.init_metrics() if needed
ex.submit_circuits(circuits, num_shots=NUM_SHOTS)

# Finalize: aggregate per-group averages
metrics.end_metrics()
metrics.finalize_all_groups()

# === Retrieve metrics via accessors ===

# Per-circuit metrics: {group: {circuit: {metric: value}}}
cm = metrics.get_circuit_metrics()
print("\n--- Per-circuit metrics ---")
for group in sorted(cm, key=lambda g: int(g) if g.isdigit() else 0):
    if not isinstance(cm[group], dict):
        continue
    for circuit_id in cm[group]:
        m = cm[group][circuit_id]
        print(f"  group={group} circuit={circuit_id}: "
              f"elapsed={m.get('elapsed_time', '?')}s, "
              f"exec={m.get('exec_time', '?')}s")

# Group-level metrics: averages and std deviations across circuits in each group
gm = metrics.get_group_metrics()
print("\n--- Group metrics (averaged +/- std dev) ---")
print(f"  {'group':>6}  {'avg_elapsed':>12}  {'std_elapsed':>12}  {'avg_exec':>12}  {'std_exec':>12}")
print(f"  {'-----':>6}  {'----------':>12}  {'----------':>12}  {'--------':>12}  {'--------':>12}")
for i, group in enumerate(gm["groups"]):
    avg_elapsed = gm["avg_elapsed_times"][i] if i < len(gm["avg_elapsed_times"]) else 0
    std_elapsed = gm["std_elapsed_times"][i] if i < len(gm["std_elapsed_times"]) else 0
    avg_exec = gm["avg_exec_times"][i] if i < len(gm["avg_exec_times"]) else 0
    std_exec = gm["std_exec_times"][i] if i < len(gm["std_exec_times"]) else 0
    print(f"  {group:>6}  {avg_elapsed:12.4f}  {std_elapsed:12.4f}  {avg_exec:12.4f}  {std_exec:12.4f}")

print("\n--- Submit with metrics test complete ---")
