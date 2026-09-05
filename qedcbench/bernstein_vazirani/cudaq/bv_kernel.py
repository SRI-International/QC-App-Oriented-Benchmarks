'''
Bernstein-Vazirani Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import cudaq

from typing import List

# saved circuits for display
QC_ = None
Uf_ = None

############### BV Circuit Definition

@cudaq.kernel
def oracle(register: cudaq.qview, auxillary_qubit: cudaq.qubit,
           hidden_bits: List[int]):
    input_size = len(hidden_bits)
    for index, bit in enumerate(hidden_bits):
        if bit == 1:
            # apply a `cx` gate with the current qubit as
            # the control and the auxillary qubit as the target.
            x.ctrl(register[input_size - index - 1], auxillary_qubit)


# method 1 is the traditional algorithm with oracle consuming all but one qubit
@cudaq.kernel
def bv_kernel_m1(num_qubits: int, secret_int: int, hidden_bits: List[int]):

    # size of input is one less than available qubits
    input_size = num_qubits - 1

    # Allocate the specified number of qubits - this
    # corresponds to the length of the hidden bitstring.
    qubits = cudaq.qvector(input_size)

    # Allocate an extra auxillary qubit.
    auxillary_qubit = cudaq.qubit()

    # Prepare the auxillary qubit.
    h(auxillary_qubit)
    z(auxillary_qubit)

    # Place the rest of the register in a superposition state.
    h(qubits)

    # Query the oracle.
    oracle(qubits, auxillary_qubit, hidden_bits)

    # Apply another set of Hadamards to the register.
    h(qubits)

    # Apply measurement gates to just the `qubits`
    # (excludes the auxillary qubit).
    mz(qubits)


# method 2 uses mid-circuit measurement to create circuits with only 2 qubits
@cudaq.kernel
def bv_kernel_m2(num_qubits: int, secret_int: int,
                 hidden_bits: List[int]) -> int:

    data = cudaq.qubit()
    auxillary_qubit = cudaq.qubit()

    # put ancilla in |-> state
    x(auxillary_qubit)
    h(auxillary_qubit)

    res = 0
    # perform CX for each qubit that matches a bit in secret string
    for i in range(len(hidden_bits)):
        if hidden_bits[i] == 1:
            h(data)
            cx(data, auxillary_qubit)
            h(data)
        if mz(data):
            res = res + (1 << i)
        # Perform reset operation
        reset(data)
    return res


def BersteinVazirani (num_qubits: int, secret_int: int, hidden_bits: List[int], method: int = 1):

    if method == 2:
        qc = [bv_kernel_m2, [num_qubits, secret_int, hidden_bits],
              {"result_width": num_qubits - 1}]
    else:
        qc = [bv_kernel_m1, [num_qubits, secret_int, hidden_bits]]

    global QC_
    if num_qubits <= 6:
        QC_ = qc

    return qc

############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
    print("Sample Circuit:");
    if QC_ != None:
        try:
            print(cudaq.draw(QC_[0], *QC_[1]))
        except Exception as ex:
            print(f"ERROR attempting to draw the kernel")
            print(ex)
    else:
        print("  ... too large!")
