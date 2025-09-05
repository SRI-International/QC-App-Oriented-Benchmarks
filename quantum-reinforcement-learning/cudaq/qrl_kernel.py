'''
Quantum Reinforcement Learning Program - CUDA-Q Kernel
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''

import cudaq
import numpy as np

# saved circuits for display
QC_ = None # Quantum Circuit saved for display

############### PQC Circuit Definition for QRL

# Define a parameterized quantum circuit (PQC) for quantum reinforcement learning using CUDA Quantum
@cudaq.kernel
def generate_pqc_circuit(n_qubits: int, n_layers: int, initial_state: list, w_params: list, n_measurements : int):
    """
    Generate a parameterized quantum circuit for QRL.

    Args:
        n_qubits (int): Number of qubits in the circuit.
        n_layers (int): Number of parameterized layers.
        initial_state (list): List representing the initial state of each qubit (0 or 1).
        w_params (list): List of rotation parameters for the circuit.
        n_measurements (int): Number of qubits to measure. If 0, measure all qubits.

    Returns:
        qc: The constructed CUDA Quantum circuit object.
    """

    # If n_measurements is 0, measure all qubits
    if n_measurements == 0:
        n_measurements = n_qubits
        
    # Create a quantum register/vector with n_qubits
    qc = cudaq.qvector(n_qubits)

    # Prepare the initial state using RX rotations if initial_state[i] == 1
    for i in range(n_qubits):
        if initial_state[i]:
            rx(w_params[i], qc[i])
    
    # Add parameterized layers
    for layer in range(n_layers):
        for i in range(n_qubits):
            idx = (layer + 1) * n_qubits + i 
            ry(w_params[idx], qc[i])
            rz(w_params[idx+1], qc[i])
        # Add CZ entangling gates between neighboring qubits
        for i in range(n_qubits-1):
            cz(qc[i], qc[i+1])
    
    # Measure the specified qubits
    for i in n_measurements:
        mz(qc[i])

    # Save the circuit for display if small enough
    global QC_
    if QC_ == None or n_qubits <= 6:
        if n_qubits < 9: QC_ = qc
    return qc

############### Ideal circuit simulation 

# Calculate the noiseless counts

def ideal_simulation(qc):
    """
    Simulate the quantum circuit without noise and return the measurement counts.

    Args:
        qc: The CUDA Quantum circuit to simulate.

    Returns:
        counts: Measurement outcome counts.
    """
    counts = cudaq.sample(qc)
    return counts


############### QRL circuit drawer

# Draw the circuits of this benchmark program
def kernel_draw():
    """
    Print a sample quantum circuit if available.
    """
    print("Sample Circuit:");
    if QC_ != None:
        # Draw the saved circuit using CUDA Quantum's draw function
        print(cudaq.draw(QC_[0], *QC_[1]))
    else:
        print("  ... too large!")