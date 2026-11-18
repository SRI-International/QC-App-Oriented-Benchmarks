'''
HHL benchmark — qiskit-free helper module.
(C) Quantum Economic Development Consortium (QED-C) 2024.

Pure-numpy / stdlib equivalents of the helpers in hhl/qiskit/*.py that the
cudaq port needs (so the cudaq path doesn't pull in qiskit / qiskit_aer /
sympy). The qiskit-side modules still exist and import qiskit for the
qiskit benchmark path.
'''

import math
import numpy as np

from qedclib import metrics


def generate_sparse_H(n, k, diag_el=0.75, off_diag_el=-0.25):
    """Generate a 2-sparse symmetric N x N Hermitian matrix (N = 2**n)
    parameterized by k. Pure numpy."""
    N = 2 ** n
    k_bin = np.binary_repr(k, width=n)
    H = np.diag(diag_el * np.ones(N))
    pairs = []
    tot_indices = []
    for i in range(N):
        i_bin = np.binary_repr(i, width=n)
        j_bin = ''
        for q in range(n):
            j_bin += '0' if i_bin[q] == k_bin[q] else '1'
        j = int(j_bin, 2)
        if i not in tot_indices and j not in tot_indices:
            pairs.append([i, j])
            tot_indices.append(i)
            tot_indices.append(j)
    for pair in pairs:
        i, j = pair[0], pair[1]
        H[i, j] = off_diag_el
        H[j, i] = off_diag_el
    return H


def true_distr(A, b=0):
    """Analytical |x>=A^-1|b> distribution over input bitstrings."""
    N = len(A)
    n = int(np.log2(N))
    b_vec = np.zeros(N)
    b_vec[b] = 1.0
    x = np.linalg.inv(A) @ b_vec
    x_n = x / np.linalg.norm(x)
    probs = np.array([np.abs(xj) ** 2 for xj in x_n])
    distr = {}
    for j, prob in enumerate(probs):
        if prob > 1e-8:
            distr[np.binary_repr(j, width=n)] = prob
    total = sum(distr.values())
    return {key: distr[key] / total for key in distr}


def postselect(outcomes, return_probs=True):
    """Keep only count keys whose leftmost char is '1' (ancilla measured 1);
    drop the leading char and renormalize."""
    mar_out = {}
    for b_str, counts in outcomes.items():
        b_str = b_str.replace(" ", "")
        if b_str[0] == '1':
            mar_out[b_str[1:]] = counts
    ps_shots = sum(mar_out.values())
    shots = sum(outcomes.values())
    rate = ps_shots / shots if shots else 0.0
    if return_probs:
        mar_out = {b_str: round(mar_out[b_str] / ps_shots, 4)
                   for b_str in mar_out}
    return mar_out, rate


def analyze_and_print_result(qc, result, num_qubits, num_shots, s_int=None,
                             verbose=False):
    counts = result.get_counts(qc)

    if verbose:
        print(f"... for circuit = {num_qubits} {s_int}, counts = {counts}")

    post_counts, rate = postselect(counts)
    if not post_counts:
        return counts, 0.0
    num_input_qubits = len(list(post_counts.keys())[0])

    if verbose:
        print(f'... ratio of counts with ancilla measured |1> : {round(rate, 4)}')

    # Decode the (i+1, odi, b) 24-bit packed fields written by the run loop.
    b = s_int & 0xFFFFFF
    off_diag_index = (s_int >> 24) & 0xFFFFFF

    if verbose:
        print(f"... b = {b}, odi = {off_diag_index}")

    diag_el = 0.5
    off_diag_el = -0.25
    A = generate_sparse_H(num_input_qubits, off_diag_index,
                          diag_el=diag_el, off_diag_el=off_diag_el)
    ideal_distr = true_distr(A, b)

    fidelity = metrics.polarization_fidelity(post_counts, ideal_distr)
    return post_counts, fidelity


def alpha2theta_pure(alpha):
    """Pure-Python equivalent of qiskit's
    uniform_controlled_rotation.alpha2theta. Replaces the sympy-based
    GrayCode generator with `gray(i) = i ^ (i >> 1)`."""
    N = len(alpha)
    n = int(np.log2(N))
    M = np.zeros((N, N))
    for i in range(N):
        g_i_bin = np.binary_repr(i ^ (i >> 1), width=n)
        for j in range(N):
            j_bin = np.binary_repr(j, width=n)[::-1]
            prod = sum(1 for k in range(n)
                       if g_i_bin[k] == '1' and j_bin[k] == '1') % 2
            M[i, j] = ((-1) ** prod) / N
    return M @ np.array(alpha)


def ucr_sequence_pure(n, theta):
    """Replicate qiskit `uniformly_controlled_rot(n, theta)` as a flat
    `(ry_angles, ctrl_indices)` pair, in pure Python.
    """
    ry_angles = []
    ctrl_indices = []

    def recurse(qubit_idx_list, theta_slice):
        if len(qubit_idx_list) == 1:
            ry_angles.append(float(theta_slice[0]))
            ctrl_indices.append(qubit_idx_list[0])
            ry_angles.append(float(theta_slice[1]))
        else:
            half = len(theta_slice) // 2
            recurse(qubit_idx_list[1:], theta_slice[:half])
            ctrl_indices.append(qubit_idx_list[0])
            recurse(qubit_idx_list[1:], theta_slice[half:])

    recurse(list(range(n)), list(theta))
    ctrl_indices.append(0)
    return ry_angles, ctrl_indices
