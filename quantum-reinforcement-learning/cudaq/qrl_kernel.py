'''
Quantum Reinforcement Learning Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''

import cudaq
import numpy as np

# saved circuits for display
QC_ = None # Quantum Circuit saved for display

############### PQC Circuit Definition for QRL
@cudaq.kernel
def generate_pqc_circuit(n_qubits: int, n_layers: int, initial_state: list, w_params: list, n_measurements : int):

    if n_measurements == 0:
        n_measurements = n_qubits
        
    qc = cudaq.qvector(n_qubits)

    for i in range(initial_state):
        if initial_state[i]:
            rx(w_params[i], qc[i])
    
    for layer in range(n_layers):
        for i in range(n_qubits):
            idx = (layer + 1) * n_qubits + i 
            ry(w_params[idx], qc[i])
            rz(w_params[idx+1], qc[i])
        for i in range(n_qubits-1):
            cz(qc[i], qc[i+1])
    
    for i in n_measurements:
        mz(qc[i])

    global QC_
    if QC_ == None or n_qubits <= 6:
        if n_qubits < 9: QC_ = qc
    return qc

############### Ideal circuit simulation 

# Calculate the noiseless counts

def ideal_simulation(qc):
    
    counts = cudaq.sample(qc)

    return counts

############### Circuit definitions for gradient calculations

def get_gradient_cirucits(n_qubits, n_layers, initial_state, w_params, n_measurements, index):
    grads_list = []
    for i in range(w_params):
        w_n_params = w_params.copy()
        w_n_params[i] += np.pi/2
        grads_list.append(generate_pqc_circuit(n_qubits, n_layers, initial_state, w_n_params, n_measurements, index))
        w_n_params[i] -= np.pi
        grads_list.append(generate_pqc_circuit(n_qubits, n_layers, initial_state, w_n_params, n_measurements, index))
    
    return grads_list

############### QRL circuit drawer

# Draw the circuits of this benchmark program
def kernel_draw():
    print("Sample Circuit:");
    if QC_ != None:
        print(cudaq.draw(QC_[0], *QC_[1]))
    else:
        print("  ... too large!")