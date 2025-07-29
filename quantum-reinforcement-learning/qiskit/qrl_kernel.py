'''
Quantum Reinforcement Learning Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''

from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import AerSimulator

QC_ = None # Quantum Circuit saved for display

############### PQC Circuit Definition for QRL
def generate_pqc_circuit(n_qubits: int, n_layers: int, initial_state: list, w_params: list, n_measure: int = 0, data_reupload = False):
    """
    Generate a parameterized quantum circuit (PQC) for quantum reinforcement learning.

    Args:
        n_qubits (int): Number of qubits in the circuit.
        n_layers (int): Number of parameterized layers.
        initial_state (list): List representing the initial state of each qubit (0 or 1) which is the bit string of the state.
        w_params (list): List of rotation parameters and initial scaling parameters for the circuit.
        n_measure (int, optional): Number of qubits to measure. If 0, measure all qubits.

    Returns:
        QuantumCircuit: The constructed quantum circuit.
    """
    # Determine which qubits to measure
    if n_measure == 0:
        n_measurements = list(range(n_qubits))
    else:
        n_measurements = list(range(n_measure))
    
    # Create a quantum circuit with n_qubits and classical bits for measurementsa
    qc = QuantumCircuit(n_qubits, len(n_measurements)) 
    
    # Add parameterized layers
    for layer in range(n_layers):
        # Prepare the initial state using RX rotations if initial_state[i] == 1
        if ((layer == 0) or (data_reupload)):
            idx = layer * 3 * n_qubits
            for i in range(len(initial_state)):
                if initial_state[i] == 1:
                    qc.rx(w_params[idx], i)
                idx += 1
            qc.barrier()  # Add a barrier after state preparation
        for i in range(n_qubits):
            qc.ry(w_params[idx], i)
            idx += 1
        for i in range(n_qubits):
            qc.rz(w_params[idx], i)
            idx += 1
        qc.barrier()  # Barrier after single-qubit rotations
        for i in range(n_qubits-1):
            qc.cz(i, i+1)  # Add CZ entangling gates between neighboring qubits
        qc.barrier()  # Barrier after entangling gates
    
    # Measure the specified qubits
    midx = 0
    for i in n_measurements:
        qc.measure(i, midx)
        midx += 1

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
        qc (QuantumCircuit): The quantum circuit to simulate.

    Returns:
        dict: Measurement outcome counts.
    """
    simulator =  AerSimulator()
    qc_trans = transpile(qc, simulator)  # Transpile the circuit for the simulator
    result = simulator.run(qc_trans).result()  # Run the simulation
    counts = result.get_counts()  # Get the measurement counts

    return counts


############### QRL Circuit Drawer

# Draw the circuits of this benchmark program

def kernel_draw():
    """
    Print a sample quantum circuit if available.
    """
    print("Sample Circuit:");
    print(QC_ if QC_ != None else "  ... too large!")