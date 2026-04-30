'''
QED-C Benchmark - Simple Example
Demonstrates the three-step benchmark execution: get_circuits, run_circuits, plot_results.
Run from the repository root, or after pip install -e .
'''

from _common.qedc_init import qedc_set_api

# Step 0: Configure the API (once, at startup)
qedc_set_api("qiskit")

# Import the benchmark you want to run
from quantum_fourier_transform import qft_benchmark as qft

# Step 1: Create benchmark circuits
all_qcs, circuit_metrics = qft.get_circuits(min_qubits=3, max_qubits=8, max_circuits=3)

# Step 2: Execute on a backend
qft.run_circuits(all_qcs, num_shots=100, backend_id="qasm_simulator")

# Step 3: Plot results
qft.plot_results(draw_circuits=True, plot_results=True)

# Or, do it all in one call:
# qft.run(min_qubits=3, max_qubits=8, max_circuits=3, num_shots=100, backend_id="qasm_simulator")
