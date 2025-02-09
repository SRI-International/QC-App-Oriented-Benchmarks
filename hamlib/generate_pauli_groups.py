import math
import numpy as np
# Configure module paths
import sys
sys.path.insert(1, "_common")
sys.path.insert(1, "qiskit")
from hamlib_utils import load_hamlib_file, get_hamlib_sparsepaulilist


X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

pauli_map = {
    'I' : I,
    'X' : X,
    'Z' : Z,
    'Y' : Y
}

def pauli_string_to_string(pauli_string, qubits):
    """Convert a PauliString to a string representation with 'I' for identity."""
    #print(f"... pauli_string_to_string({pauli_string}, {qubits}")
    result = []
    for qubit in qubits:
        if qubit in pauli_string.keys():
            result.append(str(pauli_string[qubit]))
        else:
            result.append('-')
    #print(f"... result = {result}")
    return result

def compute_groups(num_qubits: int, sparse_pauli_list: list, k: int):
    return get_si_sets(num_qubits, sparse_pauli_list, k)

def compute_blocks(qubits, k):
    return [qubits[k * i: k * (i + 1)] for i in range(math.ceil(len(qubits) / k))]

def dict_to_string(input_dic: dict, num_qubits: int):
    """
    Convert the dictionary representation of pauli string to a string
    eg. n = 4, {0: 'Z', 1: 'Z'} -> 'ZZII'
    """
    output_string = ''
    for i in range(num_qubits):
        if i in input_dic.keys():
            output_string += input_dic[i]
        else:
            output_string += 'I'
    return output_string

def get_terms_ordered_by_abscoeff(sparse_pauli_list:list, num_qubits: int) -> list:
    """Returns the terms of the PauliSum ordered by coefficient absolute value.

    Args:
        ham: A PauliSum.
    Returns:
        a list of tuples including a PauliString and its coefficient sorted by the
        absolute value of their coefficient.
    """
    sorted_data = sorted(sparse_pauli_list, key=lambda x: abs(x[1]), reverse=True)
    sorted_pauli_terms = [(dict_to_string(x[0], num_qubits), x[1]) for x in sorted_data]
    return sorted_pauli_terms

def restrict_to(
        pauli: str, qubits: list[int]
) -> str:
    """Returns the Pauli string restricted to the provided qubits.

    Arguments:
        pauli: A Pauli string.
        qubits: A set of qubits.

    Returns:
        The provided Pauli string acting only on the provided qubits.
        Note: This could potentially be empty (identity).
    """
    new_pauli_strings = ''
    for idx, p in enumerate(pauli):
        if idx in qubits:
            new_pauli_strings += p
    return new_pauli_strings

def pauli_string_to_matrix(pauli_str: str) -> np.array:
    """
    Converts a Pauli string (like 'XYZ') to the corresponding matrix.
    """
    result = pauli_map[pauli_str[0]]
    for p in pauli_str[1:]:
        result = np.kron(result, pauli_map[p])
    return result

def if_commute(pauli1: str, pauli2: str, atol=1e-8) -> bool:
    """
    Checks if the pauli1(A) and pauli2(B) commutes.
    if AB-BA < atol, they commute.
    """
    mat1 = pauli_string_to_matrix(pauli1)
    mat2 = pauli_string_to_matrix(pauli2)
    # calculate the commutator
    commutator = mat1 @ mat2 - mat2 @ mat1
    if np.allclose(commutator, np.zeros_like(commutator), atol=atol):
        return True # they commute
    else:
        return False # they don't commute

def commutes(pauli1: str, pauli2: str, blocks) -> bool:
    """Returns True if pauli1 k-commutes with pauli2, else False.

    Arguments:
        pauli1: A Pauli string.
        pauli2: A Pauli string.
        blocks: The block partitioning.

    """

    for block in blocks:
        if not if_commute(restrict_to(pauli1, block), restrict_to(pauli2, block)):
            return False
    return True

def get_si_sets(num_qubits:int, sparse_pauli_list, k: int = 1) -> list[list[tuple]]:
    """Returns grouping from the sorted insertion algorithm [https://quantum-journal.org/papers/q-2021-01-20-385/].

    Args:
        op: The observable to group.
        k: The integer k in k-commutativity.
    """

    blocks = compute_blocks(list(range(num_qubits)), k)

    commuting_sets = []
    for (pstring, coeff) in get_terms_ordered_by_abscoeff(sparse_pauli_list, num_qubits):
        found_commuting_set = False

        for commset in commuting_sets:
            cant_add = False

            for pauli in commset:
                if not commutes(pstring, pauli[0], blocks):
                    cant_add = True
                    break

            if not cant_add:
                commset.append((pstring, coeff))
                found_commuting_set = True
                break

        if not found_commuting_set:
            commuting_sets.append([(pstring, coeff)])

    return commuting_sets

# hamiltonian_name = 'chemistry/electronic/standard/H2'
# hamiltonian_params = { "ham_BK": '' }
# num_qubits = 4
#
# # load the HamLib file for the given hamiltonian name
# load_hamlib_file(filename=hamiltonian_name)
#
# # return a sparse Pauli list of terms queried from the open HamLib file
# sparse_pauli_terms, dataset_name = get_hamlib_sparsepaulilist(num_qubits=num_qubits, params=hamiltonian_params)
# print(f"... sparse_pauli_terms = \n{sparse_pauli_terms}")
#
#
# k = 3
# print(compute_groups(num_qubits, sparse_pauli_terms, k))
