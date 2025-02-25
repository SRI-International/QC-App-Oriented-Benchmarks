import numpy as np
from qiskit import QuantumCircuit

def h(pstring, index):
    result = pstring.copy()
    signchange = False

    if pstring[index] == 'X':
        result[index] = 'Z'
    elif pstring[index] == 'Y':
        signchange = True
    elif pstring[index] == 'Z':
        result[index] = 'X'

    return result, signchange


def s(pstring, index):
    result = pstring.copy()
    signchange = False

    if pstring[index] == 'X':
        result[index] = 'Y'
    elif pstring[index] == 'Y':
        result[index] = 'X'
        signchange = True

    return result, signchange


def sdag(pstring, index):
    result = pstring.copy()
    signchange = False

    if pstring[index] == 'X':
        result[index] = 'Y'
        signchange = True
    elif pstring[index] == 'Y':
        result[index] = 'X'

    return result, signchange


def cx(pstring, i1, i2):
    result = pstring.copy()
    signchange = False

    p1 = pstring[i1]
    p2 = pstring[i2]

    if p1 == '-' and (p2 == 'Z' or p2 == 'Y'):
        result[i1] = 'Z'
    elif p1 == 'Z' and (p2 == 'Z' or p2 == 'Y'):
        result[i1] = '-'
    elif p1 == 'X' and p2 == '-':
        result[i2] = 'X'
    elif p1 == 'X' and p2 == 'X':
        result[i2] = '-'
    elif p1 == 'X' and p2 == 'Y':
        result[i1] = 'Y'
        result[i2] = 'Z'
    elif p1 == 'X' and p2 == 'Z':
        result[i1] = 'Y'
        result[i2] = 'Y'
        signchange = True
    elif p1 == 'Y' and p2 == '-':
        result[i2] = 'X'
    elif p1 == 'Y' and p2 == 'X':
        result[i2] = '-'
    elif p1 == 'Y' and p2 == 'Y':
        result[i1] = 'X'
        result[i2] = 'Z'
        signchange = True
    elif p1 == 'Y' and p2 == 'Z':
        result[i1] = 'X'
        result[i2] = 'Y'

    return result, signchange


def diag_w_1q(pstring, n):
    ops = []
    for i in range(n):
        s = pstring[i]
        if s == 'X':
            ops.append(['H', i])
        elif s == 'Y':
            ops.append(['Sdag', i])
            ops.append(['H', i])

    return ops


def localize_diagonal(pstring, n):
    indices = []
    for i in range(n):
        if pstring[i] != '-':
            indices.append(i)

    if len(indices) < 2:
        return []

    ops = []
    for i in range(len(indices) - 1):
        # ops.append(['CX', indices[i], indices[-1]])
        ops.append(['CX', indices[i], indices[i + 1]])

    return ops


def local_diagonalize(pstring, n):
    ops1 = diag_w_1q(pstring, n)
    ops2 = localize_diagonal(pstring, n)

    return ops1 + ops2


def sort_pstrings(pstrings):
    vals = np.zeros(len(pstrings))

    for ip, p in enumerate(pstrings):
        for j in range(len(p) - 1, -1, -1):
            if p[j] != '-':
                vals[ip] = -j
                break

    indices = np.argsort(vals)
    result = []
    for i in indices:
        result.append(pstrings[i])

    return result


def get_right_most(pstrings, current_qubit):
    # print('get_right_most\t', current_qubit)
    vals = -np.ones(len(pstrings), dtype=int)
    for ip, p in enumerate(pstrings):
        for j in range(current_qubit, -1, -1):
            if p[j] != '-':
                vals[ip] = j
                break
    result_index = np.argmax(vals)
    return result_index, vals[result_index]



def simultaneously_diagonalize(old_pstringlist, barrier=False):
    # assume that the list is somehow sorted
    ops = []

    n = len(old_pstringlist[0])

    pstringlist = old_pstringlist.copy()

    current_qubit = n - 1

    for i in range(len(pstringlist)):

        index, current_qubit = get_right_most(pstringlist, current_qubit)
        new_ops = local_diagonalize(pstringlist[index], current_qubit + 1)
#         print('new ops in sim:', new_ops)
        current_qubit += -1

        if barrier:
            ops = ops + new_ops + [['Barr']]
        else:
            ops = ops + new_ops

    return ops


def diagonalized_pauli_strings(pauli_string_terms, k, n):
    pauli_string_list = transfer_ops(pauli_string_terms)
    pauli_diag_string_terms = []
    ops = kcommutative_diagonalize(pauli_string_list, k, n)
    for term, coeff in zip(pauli_string_list, pauli_string_terms):
#         term_list = list(term)
#         print('old term:', term_list)
        new_op, change = apply_ops(ops, term)
        new_op = [pauli if pauli != '-' else 'I' for pauli in new_op]
#         print('new term:', new_op)
        pauli_diag_string_terms.append((''.join(term), ''.join(new_op), coeff[1], change))

    return pauli_diag_string_terms
              

def apply_ops(ops, pstring):
    signchange = 1
    result = pstring.copy()

    for o in ops:

        change = False
        if o[0] == 'H':
            result, change = h(result, o[1])
        elif o[0] == 'S':
            result, change = s(result, o[1])
        elif o[0] == 'Sdag':
            result, change = sdag(result, o[1])
        elif o[0] == 'CX':
            result, change = cx(result, o[1], o[2])

        if change:
            signchange = -1
    return result, signchange


def kcommutative_diagonalize(pstrings, k, n):
    #assuming that the pstrings k-commute with each other!! It might not work otherwise.
    operation_list = []
    
    for i in range(n//k):
        newlist = []
        for p in pstrings:
            newlist.append(p[i*k:(i+1)*k])
        newops = simultaneously_diagonalize(newlist)
        for o in newops:
            for j in range(1, len(o)):
                o[j] = o[j] + i*k
        operation_list += newops
        
    if n%k > 0:
        newlist = []
        for p in pstrings:
            newlist.append(p[n-(n%k):n])

        newops = simultaneously_diagonalize(newlist)
        for o in newops:
            for j in range(1, len(o)):
                o[j] = o[j] + n-(n%k)

        operation_list += newops

    return operation_list


def transfer_ops(ops):
    ops = [op[0] for op in ops]
    new_ops = []
    for op in ops:
        op = list(op)
        op = ['-' if x == 'I' else x for x in op]
        new_ops.append(op)
    return new_ops


def create_circuits_for_pauli_terms_k_commute(qc, grouping_paulis, k, barrier=False):
    qc_complete = qc.copy()
    pauli_string_list = transfer_ops(grouping_paulis)
#     print(pauli_string_list)
    n = qc_complete.num_qubits
    ops = kcommutative_diagonalize(pauli_string_list, k, n)[::-1]
    measurement_qc = QuantumCircuit(n)
    for i in range(len(ops) - 1, -1, -1):
        o = ops[i]
        if o[0] == 'H':
            measurement_qc.h(n - 1 - o[1])
        elif o[0] == 'S':
            measurement_qc.sdg(n - 1 - o[1])
        elif o[0] == 'Sdag':
            measurement_qc.s(n - 1 - o[1])
        elif o[0] == 'CX':
            measurement_qc.cx(n - 1 -o[1], n - 1 -o[2])
        elif o[0] == 'Barr' and barrier:
            measurement_qc.barrier()
    qc_complete.compose(measurement_qc, qubits=list(range(qc_complete.num_qubits)), inplace=True)
    qc_complete.measure_all()
    return qc_complete
