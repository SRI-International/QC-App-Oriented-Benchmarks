import sys
import os

from qiskit import QuantumCircuit 
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer
from qiskit import transpile
import numpy as np

def HamiltonianSimulation_Noiseless(qc, num_qubits, circuit_id: str="0", num_shots=100):
    """
    Simulate a quantum Hamiltonian circuit for a specified number of spins using a noiseless quantum simulator.
    
    This function creates a quantum circuit, transpiles it for optimal execution, and runs it on a quantum simulator.
    It simulates the circuit with a specified number of shots and collects the results to compute the probability 
    distribution of measurement outcomes, normalized to represent probabilities.
    
    Args:
        n_spins (int): The number of spins (qubits) to simulate in the circuit.
    
    Returns:
        dict: A dictionary with keys representing the outcomes and values representing the probabilities of these outcomes.
    
    Note:
        This function uses the 'qasm_simulator' backend from Qiskit's Aer module, which simulates a quantum circuit 
        that measures qubits and returns a count of the measurement outcomes. The function assumes that the circuit 
        creation and the simulator are perfectly noiseless, meaning there are no errors during simulation.
    """
    
    # DEVNOTE: this number might need to change based on number of qubits
    num_shots = 100000
    
    backend = Aer.get_backend("qasm_simulator")
    
    # Transpile and run the circuits  
    transpiled_qc = transpile(qc, backend, optimization_level=0)
    job = backend.run(transpiled_qc, shots=num_shots)
    
    #Uncomment this to use statevector instead (but it's slower)
    #backend = Aer.get_backend("statevector_simulator")
    #job = backend.run(qc, shots=num_shots)
    
    result = job.result()
    counts = result.get_counts(qc)
    # Normalize probabilities for Heisenberg model circuit 
    dist = {}
    for key in counts.keys():
        prob = counts[key] / num_shots
        dist[key] = prob

    return dist