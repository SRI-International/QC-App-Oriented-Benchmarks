"""
11_execute_parallel.py — Test parallel execution via execute_circuits.

Tests that setting execute.parallel_execution = True causes execute_circuits()
to route through execute_circuits_parallel(). Currently a stub that executes
sequentially, but validates the plumbing end-to-end.

Usage:
    python 11_execute_parallel.py              # qiskit (default)
    python 11_execute_parallel.py -a qiskit
    python 11_execute_parallel.py -a cudaq
"""

import argparse
import time

parser = argparse.ArgumentParser(description="Test parallel execution")
parser.add_argument("--api", "-a", default="qiskit", help="API: qiskit or cudaq")
args = parser.parse_args()

api = args.api
print(f"=== Parallel Execution Test (api={api}) ===\n")

# Initialize and configure backend (includes warmup to prime transpiler)
import qedclib
qedclib.initialize(api)
import execute as ex

ex.set_execution_target(backend_id="qasm_simulator" if api == "qiskit" else None)
ex.verbose = True

# Build test circuits — 5 simple Hadamard circuits of different widths
if api == "qiskit":
    from qiskit import QuantumCircuit

    circuits = []
    for n in [2, 3, 4, 5, 6]:
        qc = QuantumCircuit(n, n)
        for i in range(n):
            qc.h(i)
        for i in range(n):
            qc.measure(i, i)
        qc.name = f"h_{n}q"
        circuits.append(qc)

elif api == "cudaq":
    import cudaq

    circuits = []
    for n in [2, 3, 4, 5, 6]:
        @cudaq.kernel
        def hadamard_kernel(num_qubits: int):
            qubits = cudaq.qvector(num_qubits)
            for i in range(num_qubits):
                h(qubits[i])
            mz(qubits)
        circuits.append([hadamard_kernel, [n]])

num_shots = 1000
print(f"Created {len(circuits)} circuits")

#############################################
# Test 1: Normal (sequential) execution
#############################################
print(f"\n--- Test 1: Normal execution ---")

t0 = time.time()
job_id, result = ex.execute_circuits(circuits, num_shots=num_shots)
elapsed_seq = time.time() - t0

counts_list = result.get_counts()
if not isinstance(counts_list, list):
    counts_list = [counts_list]

print(f"Job ID: {job_id}")
print(f"Elapsed: {elapsed_seq:.3f}s")
print(f"Results: {len(counts_list)} count dicts")
for i, counts in enumerate(counts_list):
    total = sum(counts.values())
    print(f"  Circuit {i}: {len(counts)} outcomes, {total} shots")

#############################################
# Test 2: Parallel execution (Qiskit only)
#
# For cudaq, circuit-level parallel already works via gpus_per_circuit=1
# with MPI — no new flag needed. The parallel_execution flag is Qiskit-only,
# for qubit-mapped parallel on a single QPU.
#############################################
if api == "qiskit":
    print(f"\n--- Test 2: Parallel execution (parallel_execution=True) ---")
    ex.parallel_execution = True

    t0 = time.time()
    job_id, result = ex.execute_circuits(circuits, num_shots=num_shots)
    elapsed_par = time.time() - t0

    counts_list_par = result.get_counts()
    if not isinstance(counts_list_par, list):
        counts_list_par = [counts_list_par]

    print(f"Job ID: {job_id}")
    print(f"Elapsed: {elapsed_par:.3f}s")
    print(f"Results: {len(counts_list_par)} count dicts")
    for i, counts in enumerate(counts_list_par):
        total = sum(counts.values())
        print(f"  Circuit {i}: {len(counts)} outcomes, {total} shots")

    # Reset flag
    ex.parallel_execution = False
else:
    print(f"\n--- Test 2: Skipped (cudaq already has circuit-level parallel via -gpc 1) ---")
    counts_list_par = counts_list  # use same results for validation

#############################################
# Validate: both paths should return same structure
#############################################
print(f"\n--- Validation ---")
ok = True

if len(counts_list) != len(counts_list_par):
    print(f"FAIL: sequential returned {len(counts_list)} results, parallel returned {len(counts_list_par)}")
    ok = False
else:
    print(f"OK: both returned {len(counts_list)} result sets")

for i, (seq, par) in enumerate(zip(counts_list, counts_list_par)):
    seq_total = sum(seq.values())
    par_total = sum(par.values())
    if seq_total != par_total:
        print(f"FAIL: circuit {i} shot count mismatch: seq={seq_total}, par={par_total}")
        ok = False
    else:
        print(f"OK: circuit {i}: {seq_total} shots in both")

if ok:
    print(f"\nAll checks passed.")
else:
    print(f"\nSome checks FAILED.")

print(f"\n=== Parallel Execution Test complete ===")
