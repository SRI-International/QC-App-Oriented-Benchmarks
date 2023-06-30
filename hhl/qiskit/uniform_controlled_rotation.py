
"""

Uniformly controlled rotation from arXiv:0407010

"""

import numpy as np
from sympy.combinatorics.graycode import GrayCode

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute


def dot_product(str1, str2):
    """ dot product between 2 binary string """
    
    prod = 0
    for j in range(len(str1)):
        if str1[j] == '1' and str2[j] == '1':
            prod = (prod + 1)%2
    
    return prod


def conversion_matrix(N):
    
    M = np.zeros((N,N))
    n = int(np.log2(N))
    
    gc_list = list(GrayCode(n).generate_gray()) # list of gray code strings
    
    for i in range(N):
        g_i = gc_list[i]
        for j in range(N):
            b_j = np.binary_repr(j, width=n)[::-1]
            M[i,j] = (-1)**dot_product(g_i, b_j)/(2**n)
    
    return M


def alpha2theta(alpha):
    """
    
    alpha : list of angles that get applied controlled on 0,...,2^n-1
    theta : list of angles occuring in circuit construction

    """
    
    N = len(alpha)
    
    M = conversion_matrix(N)
    theta = M @ np.array(alpha)
    
    return theta


def uni_con_rot_recursive_step(qc, qubits, anc, theta):
    """
    
    qc : qiskit QuantumCircuit object
    qubits : qiskit QuantumRegister object
    anc : ancilla qubit register on which rotation acts
    theta : list of angles specifying rotations for 0, ..., 2^(n-1)
    
    """
    
    if type(qubits) == list:
        n = len(qubits)
    else:
        n = qubits.size
    
    # lowest level of recursion
    if n == 1:
        qc.ry(theta[0], anc[0])
        qc.cx(qubits[0], anc[0])
        qc.ry(theta[1], anc[0])
    
    elif n > 1:
        
        qc = uni_con_rot_recursive_step(qc, qubits[1:], anc, theta[0:int(len(theta)/2)])
        qc.cx(qubits[0], anc[0])
        qc = uni_con_rot_recursive_step(qc, qubits[1:], anc, theta[int(len(theta)/2):])
    
    return qc
    
def uniformly_controlled_rot(n, theta):
    qubits = QuantumRegister(n)
    anc_reg = QuantumRegister(1)
    qc = QuantumCircuit(qubits, anc_reg, name = 'INV_ROT')
    qc = uni_con_rot_recursive_step(qc, qubits, anc_reg, theta)
    qc.cx(qubits[0], anc_reg[0])

    return qc


# def uniformly_controlled_rot(qc, qubits, anc, theta):
#     """
    
#     qc : qiskit QuantumCircuit object
#     qubits : qiskit QuantumRegister object
#     anc : ancilla qubit register on which rotation acts
#     theta : list of angles specifying rotations for 0, ..., 2^(n-1)
    
#     """
#     qc = uni_con_rot_recursive_step(qc, qubits, anc, theta)
#     qc.cx(qubits[0], anc[0])
    
#     return qc


def test_circuit(n):
    
    shots = 10000
    C = 0.25
    
    N = 2**n
    # make list of rotation angles
    alpha = [2*np.arcsin(C)]
    for j in range(1,N):
        j_rev = int(np.binary_repr(j, width=n)[::-1],2)
        alpha.append(2*np.arcsin(C*N/j_rev))
        theta = alpha2theta(alpha)
    
    
    for x in range(N): # state prep
    
        #x_bin = np.binary_repr(x, width=n)
        qubits = QuantumRegister(n)
        anc = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qubits, anc, cr)
        # state prep
        x_bin = np.binary_repr(x, width=n)
        for q in range(n):
            if x_bin[n-1-q] == '1':
                qc.x(q)
        qc.barrier()
        qc = uniformly_controlled_rot(qc, qubits, anc, theta)
        qc.barrier()
        qc.measure(anc[0], cr[0])
        
        outcomes = sim_circuit(qc, shots)
        print(round(outcomes['1']/shots, 4))
    

    
def sim_circuit(qc, shots):
    
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=shots).result()
    outcomes = result.get_counts(qc)
    
    
    return outcomes
