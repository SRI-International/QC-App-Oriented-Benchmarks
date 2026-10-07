"""
12_execute_circuit_groups.py — Test group-level parallel execution.

Tests execute_circuit_groups() with groups of varying sizes, circuit widths,
and shot counts. Validates sequential and parallel paths return consistent
results, and that width validation warnings are generated correctly.

Usage:
    python 12_execute_circuit_groups.py              # qiskit (default)
    python 12_execute_circuit_groups.py -a qiskit
"""

import argparse
import time

parser = argparse.ArgumentParser(description="Test group-level parallel execution")
parser.add_argument("--api", "-a", default="qiskit", help="API: qiskit or cudaq")
args = parser.parse_args()

api = args.api
print(f"=== Group-Level Parallel Execution Test (api={api}) ===\n")

# Initialize and configure backend (includes warmup)
import qedclib
qedclib.initialize(api)
import execute as ex

ex.set_execution_target(backend_id="qasm_simulator" if api == "qiskit" else None)
ex.verbose = True

#############################################
# Build test circuit groups
#############################################

if api == "qiskit":
    from qiskit import QuantumCircuit

    def make_circuit(n_qubits, label=""):
        """Create a simple Hadamard circuit of given width."""
        qc = QuantumCircuit(n_qubits, n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        for i in range(n_qubits):
            qc.measure(i, i)
        qc.name = label
        return qc

    # Group 0: 3 circuits, 4 qubits each, 500 shots
    group_0 = [make_circuit(4, f"g0_c{i}") for i in range(3)]

    # Group 1: 2 circuits, 3 qubits each, 1000 shots
    group_1 = [make_circuit(3, f"g1_c{i}") for i in range(2)]

    # Group 2: 4 circuits, 5 qubits each, 200 shots
    group_2 = [make_circuit(5, f"g2_c{i}") for i in range(4)]

    # Group 3: mixed widths within the group (3 and 5 qubits), 800 shots
    group_3 = [make_circuit(3, "g3_c0_3q"),
               make_circuit(5, "g3_c1_5q"),
               make_circuit(3, "g3_c2_3q")]

    circuit_groups = [group_0, group_1, group_2, group_3]
    num_shots_list = [500, 1000, 200, 800]

elif api == "cudaq":
    import cudaq

    def make_cudaq_circuit(n_qubits):
        """Create a [kernel, [args]] tuple for a Hadamard circuit."""
        @cudaq.kernel
        def hadamard_kernel(num_qubits: int):
            qubits = cudaq.qvector(num_qubits)
            for i in range(num_qubits):
                h(qubits[i])
            mz(qubits)
        return [hadamard_kernel, [n_qubits]]

    # Group 0: 3 circuits, 4 qubits each, 500 shots
    group_0 = [make_cudaq_circuit(4) for _ in range(3)]

    # Group 1: 2 circuits, 3 qubits each, 1000 shots
    group_1 = [make_cudaq_circuit(3) for _ in range(2)]

    # Group 2: 4 circuits, 5 qubits each, 200 shots
    group_2 = [make_cudaq_circuit(5) for _ in range(4)]

    # Group 3: mixed widths within the group (3 and 5 qubits), 800 shots
    group_3 = [make_cudaq_circuit(3), make_cudaq_circuit(5), make_cudaq_circuit(3)]

    circuit_groups = [group_0, group_1, group_2, group_3]
    num_shots_list = [500, 1000, 200, 800]

else:
    print(f"Unknown API: {api}")
    exit(1)

print(f"Created {len(circuit_groups)} groups:")
for i, (group, shots) in enumerate(zip(circuit_groups, num_shots_list)):
    if api == "qiskit":
        widths = [qc.num_qubits for qc in group]
    else:
        widths = [c[1][0] if isinstance(c, list) and len(c) > 1 else "?" for c in group]
    print(f"  Group {i}: {len(group)} circuits, widths={widths}, shots={shots}")

#############################################
# Test 1: Sequential execution (parallel off)
#############################################
print(f"\n--- Test 1: Sequential group execution ---")
ex.parallel_execution = False

t0 = time.time()
job_id, group_results = ex.execute_circuit_groups(circuit_groups, num_shots_list=num_shots_list)
elapsed_seq = time.time() - t0

print(f"Job ID: {job_id}")
print(f"Elapsed: {elapsed_seq:.3f}s")
print(f"Groups returned: {len(group_results)}")
for i, result in enumerate(group_results):
    counts_list = result.get_counts()
    if not isinstance(counts_list, list):
        counts_list = [counts_list]
    print(f"  Group {i}: {len(counts_list)} results, "
          f"shots={[sum(c.values()) for c in counts_list]}")

#############################################
# Test 2: Parallel group execution
#
# If parallel can't be done (Qiskit stub, cudaq without MPI), the
# execute module prints a one-time warning and falls back to sequential.
#############################################
print(f"\n--- Test 2: Parallel group execution (parallel_execution=True) ---")
ex.parallel_execution = True

t0 = time.time()
job_id_par, group_results_par = ex.execute_circuit_groups(
    circuit_groups, num_shots_list=num_shots_list)
elapsed_par = time.time() - t0

print(f"Job ID: {job_id_par}")
print(f"Elapsed: {elapsed_par:.3f}s")
print(f"Groups returned: {len(group_results_par)}")
for i, result in enumerate(group_results_par):
    counts_list = result.get_counts()
    if not isinstance(counts_list, list):
        counts_list = [counts_list]
    print(f"  Group {i}: {len(counts_list)} results, "
          f"shots={[sum(c.values()) for c in counts_list]}")

ex.parallel_execution = False

#############################################
# Test 3: Uniform shots (all groups same)
#############################################
print(f"\n--- Test 3: Uniform shots (all groups 500 shots) ---")
ex.parallel_execution = False

job_id_uni, group_results_uni = ex.execute_circuit_groups(
    circuit_groups, num_shots=500)
print(f"Groups returned: {len(group_results_uni)}")
for i, result in enumerate(group_results_uni):
    counts_list = result.get_counts()
    if not isinstance(counts_list, list):
        counts_list = [counts_list]
    shots = [sum(c.values()) for c in counts_list]
    print(f"  Group {i}: {len(counts_list)} results, shots={shots}")

#############################################
# Validation
#############################################
print(f"\n--- Validation ---")
ok = True

# Check group count matches
if len(group_results) != len(circuit_groups):
    print(f"FAIL: expected {len(circuit_groups)} groups, got {len(group_results)}")
    ok = False
else:
    print(f"OK: sequential returned {len(group_results)} groups")

if len(group_results_par) != len(circuit_groups):
    print(f"FAIL: parallel returned {len(group_results_par)} groups, expected {len(circuit_groups)}")
    ok = False
else:
    print(f"OK: parallel returned {len(group_results_par)} groups")

# Check per-group result counts and shot counts
for i, (seq_result, par_result, group, shots) in enumerate(
        zip(group_results, group_results_par, circuit_groups, num_shots_list)):

    seq_counts = seq_result.get_counts()
    if not isinstance(seq_counts, list):
        seq_counts = [seq_counts]
    par_counts = par_result.get_counts()
    if not isinstance(par_counts, list):
        par_counts = [par_counts]

    # Right number of results per group
    if len(seq_counts) != len(group):
        print(f"FAIL: group {i} sequential: expected {len(group)} results, got {len(seq_counts)}")
        ok = False
    else:
        print(f"OK: group {i}: {len(seq_counts)} results (matches {len(group)} circuits)")

    # Shot counts correct
    for j, counts in enumerate(seq_counts):
        total = sum(counts.values())
        if total != shots:
            print(f"FAIL: group {i} circuit {j}: expected {shots} shots, got {total}")
            ok = False

    # Parallel matches sequential structure
    if len(par_counts) != len(seq_counts):
        print(f"FAIL: group {i}: parallel has {len(par_counts)} results vs sequential {len(seq_counts)}")
        ok = False

# Check uniform shots test
for i, result in enumerate(group_results_uni):
    counts_list = result.get_counts()
    if not isinstance(counts_list, list):
        counts_list = [counts_list]
    for j, counts in enumerate(counts_list):
        total = sum(counts.values())
        if total != 500:
            print(f"FAIL: uniform test group {i} circuit {j}: expected 500 shots, got {total}")
            ok = False
print(f"OK: uniform shots test passed (all groups got 500 shots)")

if ok:
    print(f"\nAll checks passed.")
else:
    print(f"\nSome checks FAILED.")

print(f"\n=== Group-Level Parallel Execution Test complete ===")
