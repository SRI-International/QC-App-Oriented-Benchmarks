from qiskit.compiler import transpile

def remove_multiple_registers(qc):
    # Removes multiple quantum registers from a circuit. Required for Fire Opal, since
    # multiple quantum registers aren't supported yet.
    qubit_count = qc.num_qubits
    connectivity = [
        [i, j] for i in range(qubit_count) for j in range(qubit_count) if j != i
    ]
    return transpile(
        qc, layout_method="trivial", optimization_level=0, coupling_map=connectivity
    )
