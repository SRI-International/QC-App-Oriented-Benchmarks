"""
Test parameterized circuit execution via the params argument to execute_circuits.

Tests both parameter formats:
  - Format A: list of dicts with string keys
  - Format B: tuple of (names_list, values_2d_array)

For Qiskit, also tests native ParameterVector keys (the red-cedar path).

Usage:
  python 12_execute_parameterized.py              # Qiskit (default)
  python 12_execute_parameterized.py --api cudaq  # CUDA-Q
"""

import sys
import numpy as np

api = "qiskit"
if "--api" in sys.argv:
    idx = sys.argv.index("--api")
    api = sys.argv[idx + 1]

from qedclib import initialize
initialize(api)
import execute as ex


def test_qiskit():
    from qiskit.circuit import QuantumCircuit, Parameter

    theta = Parameter('theta')
    phi = Parameter('phi')
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.ry(phi, 1)
    qc.cx(0, 1)
    qc.measure_all()

    ex.set_execution_target("qasm_simulator")

    angles = np.linspace(0, np.pi, 5)

    # Format A: list of dicts with string keys
    print("=== Qiskit Format A: list of dicts (string keys) ===")
    params_a = [{"theta": t, "phi": t / 2} for t in angles]
    job_id, result = ex.execute_circuits([qc], num_shots=1000, params=params_a)
    counts_list = result.get_counts()
    print(f"Executed {len(counts_list)} parameter sets:")
    for i, counts in enumerate(counts_list):
        print(f"  theta={angles[i]:.3f}, phi={angles[i]/2:.3f}: {counts}")

    # Format B: tuple of (names, values)
    print("\n=== Qiskit Format B: tuple of (names, values) ===")
    param_names = ["theta", "phi"]
    param_values = [[t, t / 2] for t in angles]
    params_b = (param_names, param_values)
    job_id, result = ex.execute_circuits([qc], num_shots=1000, params=params_b)
    counts_list = result.get_counts()
    print(f"Executed {len(counts_list)} parameter sets:")
    for i, counts in enumerate(counts_list):
        print(f"  theta={angles[i]:.3f}, phi={angles[i]/2:.3f}: {counts}")

    # Native ParameterVector keys (red-cedar path)
    print("\n=== Qiskit Native: ParameterVector keys (red-cedar path) ===")
    params_native = [{theta: t, phi: t / 2} for t in angles]
    job_id, result = ex.execute_circuits([qc], num_shots=1000, params=params_native)
    counts_list = result.get_counts()
    print(f"Executed {len(counts_list)} parameter sets:")
    for i, counts in enumerate(counts_list):
        print(f"  theta={angles[i]:.3f}, phi={angles[i]/2:.3f}: {counts}")


def test_cudaq():
    import cudaq

    # Create a parameterized kernel
    @cudaq.kernel
    def kernel(theta: float, phi: float):
        q = cudaq.qvector(2)
        ry(theta, q[0])
        ry(phi, q[1])
        cx(q[0], q[1])
        mz(q)

    ex.set_execution_target()

    angles = np.linspace(0, np.pi, 5)

    # Format A: list of dicts with string keys
    print("=== CUDA-Q Format A: list of dicts (string keys) ===")
    params_a = [{"theta": t, "phi": t / 2} for t in angles]
    job_id, result = ex.execute_circuits([[kernel, []]], num_shots=1000, params=params_a)
    counts_list = result.get_counts()
    print(f"Executed {len(counts_list)} parameter sets:")
    for i, counts in enumerate(counts_list):
        print(f"  theta={angles[i]:.3f}, phi={angles[i]/2:.3f}: {counts}")

    # Format B: tuple of (names, values)
    print("\n=== CUDA-Q Format B: tuple of (names, values) ===")
    param_names = ["theta", "phi"]
    param_values = [[t, t / 2] for t in angles]
    params_b = (param_names, param_values)
    job_id, result = ex.execute_circuits([[kernel, []]], num_shots=1000, params=params_b)
    counts_list = result.get_counts()
    print(f"Executed {len(counts_list)} parameter sets:")
    for i, counts in enumerate(counts_list):
        print(f"  theta={angles[i]:.3f}, phi={angles[i]/2:.3f}: {counts}")


if api == "qiskit":
    test_qiskit()
elif api == "cudaq":
    test_cudaq()
else:
    print(f"Unknown API: {api}")
