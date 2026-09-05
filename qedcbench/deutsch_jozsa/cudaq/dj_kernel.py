'''
Deutsch-Jozsa Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import cudaq

# saved circuits for display
QC_ = None

############### Circuit Definition

@cudaq.kernel
def constant_oracle(auxillary_qubit: cudaq.qubit, output: int):
    if output == 1:
        x(auxillary_qubit)


@cudaq.kernel
def balanced_oracle(register: cudaq.qview, auxillary_qubit: cudaq.qubit, input_size: int):
    # map 1's to X gates
    for i_qubit in range(input_size):
        if i_qubit % 2 == 0:
            x(register[input_size - i_qubit - 1])

    for i_qubit in range(input_size):
        x.ctrl(register[input_size - i_qubit - 1], auxillary_qubit)

    for i_qubit in range(input_size):
        if i_qubit % 2 == 0:
            x(register[input_size - i_qubit - 1])


@cudaq.kernel
def dj_kernel(num_qubits: int, oracle_type: int):
    # Size of input is one less than available qubits
    input_size = num_qubits - 1
    qubits = cudaq.qvector(input_size)
    auxillary_qubit = cudaq.qubit()

    h(qubits)
    x(auxillary_qubit)
    h(auxillary_qubit)

    if oracle_type == 0:
        constant_oracle(auxillary_qubit, 0)
    else:
        balanced_oracle(qubits, auxillary_qubit, input_size)

    h(qubits)
    h(auxillary_qubit)
    # uncompute ancilla qubit, not necessary for algorithm
    x(auxillary_qubit)

    mz(qubits)


def DeutschJozsa(num_qubits: int, type: int):
    qc = [dj_kernel, [num_qubits, type]]

    global QC_
    if num_qubits <= 6:
        QC_ = qc

    return qc

############### Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
    print("Sample Circuit:")
    if QC_ is not None:
        try:
            print(cudaq.draw(QC_[0], *QC_[1]))
        except Exception as ex:
            print(f"ERROR attempting to draw the kernel")
            print(ex)
    else:
        print("  ... too large!")
