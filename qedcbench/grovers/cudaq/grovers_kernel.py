'''
Grover's Search Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import cudaq

from typing import List

# saved circuits for display
QC_ = None

############### Circuit Definition

@cudaq.kernel
def add_grover_oracle(qubits: cudaq.qview, num_qubits: int, marked_bits: List[int]):
    for i_qubit, bit in enumerate(marked_bits):
        if bit == 0:
            x(qubits[num_qubits - i_qubit - 1])

    h(qubits[0])
    x.ctrl(qubits[1:num_qubits], qubits[0])
    h(qubits[0])

    for i_qubit, bit in enumerate(marked_bits):
        if bit == 0:
            x(qubits[num_qubits - i_qubit - 1])


@cudaq.kernel
def add_diffusion_operator(qubits: cudaq.qview, num_qubits: int):
    h(qubits)
    x(qubits)

    h(qubits[0])
    x.ctrl(qubits[1:num_qubits], qubits[0])
    h(qubits[0])

    x(qubits)
    h(qubits)


@cudaq.kernel
def grovers_kernel(num_qubits: int, marked_bits: List[int], n_iterations: int):
    qubits = cudaq.qvector(num_qubits)

    # Start with Hadamard on all qubits
    h(qubits)

    # loop over the estimated number of iterations
    for _ in range(n_iterations):
        # add the grover oracle
        add_grover_oracle(qubits, num_qubits, marked_bits)
        # add the diffusion operator
        add_diffusion_operator(qubits, num_qubits)

    # measure all qubits
    mz(qubits)


def GroversSearch(num_qubits: int, marked_item: int, n_iterations: int, use_mcx_shim: bool = False):
    marked_bits = [int(bit) for bit in format(marked_item, f"0{num_qubits}b")[::-1]]
    qc = [grovers_kernel, [num_qubits, marked_bits, n_iterations]]

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
