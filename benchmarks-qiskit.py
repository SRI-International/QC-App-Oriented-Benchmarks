"""
QED-C Application-Oriented Benchmarks - Qiskit Version
"""

from importlib import import_module
import sys

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
dj_benchmark = import_module("deutsch-jozsa.qiskit.dj_benchmark")
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
sys.path.insert(1, "bernstein-vazirani/qiskit")
bv_benchmark = import_module("bernstein-vazirani.qiskit.bv_benchmark")
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
sys.path.insert(1, "hidden-shift/qiskit")
hs_benchmark = import_module("hidden-shift.qiskit.hs_benchmark")
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
sys.path.insert(1, "quantum-fourier-transform/qiskit")
qft_benchmark = import_module("quantum-fourier-transform.qiskit")
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
sys.path.insert(1, "grovers/qiskit")
grovers_benchmark = import_module("grovers.qiskit")
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
sys.path.insert(1, "phase-estimation/qiskit")
pe_benchmark = import_module("phase-estimation.qiskit")
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
sys.path.insert(1, "hhl/qiskit")
hhl_benchmark = import_module("hhl.qiskit")

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
sys.path.insert(1, "amplitude-estimation/qiskit")
ae_benchmark = import_module("amplitude-estimation.qiskit")
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
sys.path.insert(1, "monte-carlo/qiskit")
mc_benchmark = import_module("monte-carlo.qiskit")
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
sys.path.insert(1, "hamiltonian-simulation/qiskit")
hamiltonian_simulation_benchmark = import_module("hamiltonian-simulation.qiskit")
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
sys.path.insert(1, "vqe/qiskit")
vqe_benchmark = import_module("vqe.qiskit")
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
sys.path.insert(1, "shors/qiskit")
shors_benchmark = import_module("shors.qiskit")
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
