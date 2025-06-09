"""
QED-C Application-Oriented Benchmarks - Qiskit Version
"""

from qc_app_benchmarks.deutsch_jozsa.qiskit import dj_benchmark
from qc_app_benchmarks.bernstein_vazirani.qiskit import bv_benchmark
from qc_app_benchmarks.hidden_shift.qiskit import hs_benchmark
from qc_app_benchmarks.quantum_fourier_transform.qiskit import qft_benchmark
from qc_app_benchmarks.grovers.qiskit import grovers_benchmark
from qc_app_benchmarks.phase_estimation.qiskit import pe_benchmark
from qc_app_benchmarks.hhl.qiskit import hhl_benchmark
from qc_app_benchmarks.amplitude_estimation.qiskit import ae_benchmark
from qc_app_benchmarks.monte_carlo.qiskit import mc_benchmark
from qc_app_benchmarks.hamiltonian_simulation.qiskit import hamiltonian_simulation_benchmark
from qc_app_benchmarks.vqe.qiskit import vqe_benchmark
from qc_app_benchmarks.shors.qiskit import shors_benchmark


min_qubits = 2
max_qubits = 8
skip_qubits = 1
max_circuits = 3
num_shots = 1000

backend_id = "qasm_simulator"
# backend_id="statevector_simulator"

hub = ""
group = ""
project = ""
provider_backend = None
exec_options = {}

# Deutsch-Jozsa
dj_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    skip_qubits=skip_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Bernstein-Vazirani - Method 1
bv_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    skip_qubits=skip_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    method=1,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Bernstein-Vazirani - Method 2
bv_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    skip_qubits=skip_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    method=2,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Hidden Shift
hs_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Quantum Fourier Transform - Method 1
qft_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    skip_qubits=skip_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    method=1,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Quantum Fourier Transform - Method 2
qft_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    skip_qubits=skip_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    method=2,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Grover
grovers_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    skip_qubits=skip_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Phase Estimation
pe_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    skip_qubits=skip_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# HHL Linear Solver
hhl_benchmark.verbose = False

hhl_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    skip_qubits=skip_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    method=1,
    use_best_widths=True,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Amplitude Estimation
ae_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    skip_qubits=skip_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Monte Carlo
mc_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    skip_qubits=skip_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Hamiltonian Simulation - Method 1
hamiltonian_simulation_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    skip_qubits=skip_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    method=1,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Hamiltonian Simulation - Method 2
hamiltonian_simulation_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    skip_qubits=skip_qubits,
    max_circuits=max_circuits,
    num_shots=num_shots,
    method=2,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# VQE - Method 1
vqe_num_shots = 4098
vqe_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    max_circuits=max_circuits,
    num_shots=vqe_num_shots,
    method=1,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Shor - Method 1
shors_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    max_circuits=1,
    num_shots=num_shots,
    method=1,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)

# Shor - Method 2
shors_benchmark.run(
    min_qubits=min_qubits,
    max_qubits=max_qubits,
    max_circuits=1,
    num_shots=num_shots,
    method=2,
    backend_id=backend_id,
    provider_backend=provider_backend,
    hub=hub,
    group=group,
    project=project,
    exec_options=exec_options,
)
