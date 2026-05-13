'''
QED-C Benchmark - Simple Programming Example

Run from anywhere after: pip install -e .
This demonstrates the three-step modularized API:
  1. get_circuits  — generate benchmark circuits
  2. run_circuits  — execute on a backend and compute fidelities
  3. plot_results  — visualize metrics
'''

from qedclib import initialize

# Initialize qedclib with the API to use (once, at startup)
initialize("qiskit")

# Import the benchmark you want to run
from qedcbench.quantum_fourier_transform import qft_benchmark as qft

# Step 1: Create benchmark circuits
all_qcs, circuit_metrics = qft.get_circuits(min_qubits=3, max_qubits=8, max_circuits=3)

# Step 2: Execute on a backend
qft.run_circuits(all_qcs, num_shots=100, backend_id="qasm_simulator")

# Step 3: Plot results
qft.plot_results(draw_circuits=True, plot_results=True)

# Or, do it all in one call:
# qft.run(min_qubits=3, max_qubits=8, max_circuits=3, num_shots=100, backend_id="qasm_simulator")
