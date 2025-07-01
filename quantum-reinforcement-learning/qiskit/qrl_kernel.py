from qiskit import QuantumCircuit

def generate_pqc_circuit(n_qubits, n_layers, initial_state, w_params, n_measurements, index):
    qc = QuantumCircuit(n_qubits)
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
    qc.measure_all()
    return qc

def get_gradient_cirucits(n_qubits, n_layers, initial_state, w_params, n_measurements, index):
    grads_list = []
    for i in range(w_params):
        w_n_params = w_params.copy()
        w_n_params[i] += np.pi/2
        grads_list.append(generate_pqc_circuit(n_qubits, n_layers, initial_state, w_n_params, n_measurements, index))
        w_n_params[i] -= np.pi
        grads_list.append(generate_pqc_circuit(n_qubits, n_layers, initial_state, w_n_params, n_measurements, index))
    
    return grads_list

