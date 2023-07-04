# -*- coding: utf-8 -*-
"""
sparse Hamiltonian simulation

"""

from math import pi
import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute


def Ham_sim(H, t):
    """
        H : sparse matrix
        t : time parameter
        
        returns : QuantumCircuit for e^(-i*H*t)
    """
    
    # read in number of qubits
    N = len(H)
    n = int(np.log2(N))
    
    # read in matrix elements
    #diag_el = H[0,0]
    for j in range(1,N):
        if H[0,j] != 0:
            off_diag_el = H[0,j]
            break
    j_bin = np.binary_repr(j, width=n)
    sign = (-1)**((j_bin.count('1'))%2)
    
    # create registers
    qreg_a = QuantumRegister(n, name='q_a')
    creg = ClassicalRegister(n, name='c')
    qreg_b = QuantumRegister(n, name='q_b') # ancilla register
    anc_reg = QuantumRegister(1, name='q_anc')
    qc = QuantumCircuit(qreg_a, qreg_b, anc_reg, creg)
    
    # apply sparse H oracle gate
    qc = V_gate(qc, H, qreg_a, qreg_b)
    
    # apply W gate that diagonalizes SWAP operator and Toffoli
    for q in range(n):
        qc = W_gate(qc, qreg_a[q], qreg_b[q])
        qc.x(qreg_b[q])
        qc.ccx(qreg_a[q], qreg_b[q], anc_reg[0])
        
    # phase
    qc.p(sign*2*t*off_diag_el, anc_reg[0])
    
    # uncompute
    for q in range(n):
        j = n-1-q
        qc.ccx(qreg_a[j], qreg_b[j], anc_reg[0])
        qc.x(qreg_b[j])
        qc = W_gate(qc, qreg_a[j], qreg_b[j])
    
    # sparse H oracle gate is its own inverse
    qc = V_gate(qc, H, qreg_a, qreg_b)
    
    # measure
    #qc.barrier()
    #qc.measure(qreg_a[0:], creg[0:])
    
    return qc


def control_Ham_sim(n, H, t):
    """
        H : sparse matrix
        t : time parameter
        
        returns : QuantumCircuit for control-e^(-i*H*t)
    """
    qreg = QuantumRegister(n)
    qreg_b = QuantumRegister(n)
    control_reg = QuantumRegister(1)
    anc_reg = QuantumRegister(1)
    qc = QuantumCircuit(qreg, qreg_b, control_reg, anc_reg)
    control = control_reg[0]
    anc = anc_reg[0]

    
    # read in number of qubits
    N = len(H)
    n = int(np.log2(N))
    
    # read in matrix elements
    diag_el = H[0,0]
    for j in range(1,N):
        if H[0,j] != 0:
            off_diag_el = H[0,j]
            break
    j_bin = np.binary_repr(j, width=n)
    sign = (-1)**((j_bin.count('1'))%2)
    
    # use ancilla for phase kickback to simulate diagonal part of H
    qc.x(anc)
    qc.cp(-t*(diag_el+sign*off_diag_el), control, anc)
    qc.x(anc)
    
    # apply sparse H oracle gate
    qc = V_gate(qc, H, qreg, qreg_b)
    
    # apply W gate that diagonalizes SWAP operator and Toffoli
    for q in range(n):
        qc = W_gate(qc, qreg[q], qreg_b[q])
        qc.x(qreg_b[q])
        qc.ccx(qreg[q], qreg_b[q], anc)
        
    # phase
    qc.cp((sign*2*t*off_diag_el), control, anc)
     
    # uncompute
    for q in range(n):
        j = n-1-q
        qc.ccx(qreg[j], qreg_b[j], anc)
        qc.x(qreg_b[j])
        qc = W_gate(qc, qreg[j], qreg_b[j])
    
    # sparse H oracle gate is its own inverse
    qc = V_gate(qc, H, qreg, qreg_b)
    
    return qc


def V_gate(qc, H, qreg_a, qreg_b):
    """
        Hamiltonian oracle V|a,b> = |a,b+v(a)>
    
    """
    
    # index of non-zero off diagonal in first row of H
    n = qreg_a.size
    N = len(H)
    for i in range(1,N):
        if H[0,i] != 0:
            break
    i_bin = np.binary_repr(i, width=n)
    
    for q in range(n):
        a, b = qreg_a[q], qreg_b[q]
        if i_bin[n-1-q] == '1':
            qc.x(a)
            qc.cx(a,b)
            qc.x(a)
        else:
            qc.cx(a,b)
            
    return qc


def W_gate(qc, q0, q1):
    """
    W : |00> -> |00>
        |01> -> |01> + |10>
        |10> -> |01> - |10>
        |11> -> |11>
        
    """
    
    qc.rz(-pi,q1)
    qc.rz(-3*pi/2,q0)
    qc.ry(-pi/2,q1)
    qc.ry(-pi/2,q0)
    qc.rz(-pi,q1)
    qc.rz(-3*pi/2,q0)
    
    qc.cx(q0,q1)
    
    qc.rz(-7*pi/4,q1)
    qc.rx(-pi/2,q0)
    qc.rx(-pi,q1)
    qc.rz(-3*pi/2,q0)
    
    qc.cx(q0,q1)
    
    qc.ry(-pi/2,q1)
    qc.rx(-pi/4,q1)
    
    qc.cx(q1,q0)
    
    qc.rz(-3*pi/2,q0)


    return qc
    
    
def qc_unitary(qc):
    
    simulator = Aer.get_backend('unitary_simulator')
    result = execute(qc, simulator).result()
    U = np.array(result.get_unitary(qc))
    
    return U

def sim_circuit(qc, shots):
    
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=shots).result()
    outcomes = result.get_counts(qc)
    
    return outcomes


def true_distr(H, t, psi0=0):
    
    N = len(H)
    n = int(np.log2(N))
    
    U = expm(-1j*H*t)
    
    psi = U[:,psi0]
    
    probs = np.array([np.abs(amp)**2 for amp in psi])
    probs_bin = {}
    for j, p in enumerate(probs):
        if p > 0:
            j_bin = np.binary_repr(j, width=n)
            probs_bin[j_bin] = p
        
    
    return probs_bin


def generate_sparse_H(n, k, diag_el=0.75, off_diag_el=-0.25):
    """
        n (int) : number of qubits. H will be 2^n by 2^n
        
        k (int) : between 1,...,2^n-1. determines which H will
                  be generated in class of matrices
        
        generates 2-sparse H with 1.5 on diagonal and 0.5 on
        off diagonal

    """
    
    N = 2**n
    k_bin = np.binary_repr(k, width=n)
    
    # initialize H
    H = np.diag(diag_el*np.ones(N))
    
    pairs = []
    tot_indices = []
    for i in range(N):
        i_bin = np.binary_repr(i, width=n)
        j_bin = ''
        for q in range(n):
            if i_bin[q] == k_bin[q]:
                j_bin += '0'
            else:
                j_bin += '1'
        j = int(j_bin,2)
        if i not in tot_indices and j not in tot_indices:
            pairs.append([i,j])
            tot_indices.append(i)
            tot_indices.append(j)
    
    # fill in H
    for pair in pairs:
        i, j = pair[0], pair[1]
        H[i,j] = off_diag_el
        H[j,i] = off_diag_el
    
    return H


def condition_number(A):
    
    evals = np.linalg.eigh(A)[0]
    abs_evals = [abs(e) for e in evals]
    
    if min(abs_evals) == 0.0:
        return 'inf'
    else:
        k = max(abs_evals)/min(abs_evals)
        return k
    
    
