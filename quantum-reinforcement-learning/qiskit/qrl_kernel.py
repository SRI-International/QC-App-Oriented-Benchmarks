'''
Quantum Reinforcement Learning Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''

from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import AerSimulator

QC_ = None # Quantum Circuit

def generate_pqc_circuit(n_qubits, n_layers, initial_state, w_params, n_measurements = []):
    
    if len(n_measurements) == 0:
        n_measurements = list(range(n_qubits))
    
    qc = QuantumCircuit(n_qubits, len(n_measurements))
    
    for i in range(len(initial_state)):
        if initial_state[i] == 1:
            qc.x(i)
    
    for layer in range(n_layers):
        for i in range(n_qubits):
            idx = layer * n_qubits + i
            qc.rx(w_params[idx], i)
            qc.rz(w_params[idx+1], i)
        for i in range(n_qubits-1):
            qc.cz(i, i+1)
    
    midx = 0
    for i in n_measurements:
        qc.measure(i, midx)
        midx += 1

    global QC_
    if QC_ == None or n_qubits <= 6:
        if n_qubits < 9: QC_ = qc
    return qc

def ideal_simulation(qc):
    
    simulator =  AerSimulator()
    qc_trans = transpile(qc, simulator)
    result = simulator.run(qc_trans).result()
    counts = result.get_counts()

    return counts


def get_gradient_cirucits(n_qubits, n_layers, initial_state, w_params, n_measurements, index):
    grads_list = []
    for i in range(w_params):
        w_n_params = w_params.copy()
        w_n_params[i] += np.pi/2
        grads_list.append(generate_pqc_circuit(n_qubits, n_layers, initial_state, w_n_params, n_measurements, index))
        w_n_params[i] -= np.pi
        grads_list.append(generate_pqc_circuit(n_qubits, n_layers, initial_state, w_n_params, n_measurements, index))
    
    return grads_list

############### QRL Circuit Drawer

# Draw the circuits of this benchmark program

def kernel_draw():
    print("Sample Circuit:");
    print(QC_ if QC_ != None else "  ... too large!")
    