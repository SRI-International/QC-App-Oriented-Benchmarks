import os
import pprint
from qiskit.circuit.library import EfficientSU2
import math
import cirq
import numpy as np
import openfermion as of
import stim
import stimcirq
import h5py
from typing import Set, List, Iterable
import warnings
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error
import matplotlib.pyplot as plt
from qiskit import transpile
import re
from qiskit.quantum_info import SparsePauliOp
from sympy.polys.numberfields.utilities import coeff_search
from qiskit.quantum_info import Operator
from hamlib_utils import  extract_dataset_hdf5
from hamlib_simulation_kernel import process_data
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from cirq import IdentityGate
from qiskit_aer import AerSimulator
from cirq.ops.dense_pauli_string import DensePauliString
from qiskit_ibm_runtime import EstimatorV2 as Estimator

paulis = {
        'X' : np.array([[0, 1], [1, 0]]),
        'Y': np.array([[0, -1j], [1j, 0]]),
        'Z': np.array([[1, 0], [0, -1]]),
        'I': np.array([[1, 0], [0, 1]])
}

Z = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])

def restrict_to(
        pauli: cirq.PauliString, qubits: Iterable[cirq.Qid]
) -> cirq.PauliString:
        """Returns the Pauli string restricted to the provided qubits.

        Arguments:
            pauli: A Pauli string.
            qubits: A set of qubits.

        Returns:
            The provided Pauli string acting only on the provided qubits.
            Note: This could potentially be empty (identity).
        """
        return cirq.PauliString(p.on(q) for q, p in pauli.items() if q in qubits)


def commutes(pauli1: cirq.PauliString, pauli2: cirq.PauliString, blocks) -> bool:
        """Returns True if pauli1 k-commutes with pauli2, else False.

        Arguments:
            pauli1: A Pauli string.
            pauli2: A Pauli string.
            blocks: The block partitioning.

        """

        for block in blocks:
                if not cirq.commutes(restrict_to(pauli1, block), restrict_to(pauli2, block)):
                        return False
        return True


def get_num_qubits(hamiltonian: cirq.PauliSum) -> int:
        return len(hamiltonian.qubits)


def get_terms_ordered_by_abscoeff(ham: cirq.PauliSum) -> List[cirq.PauliString]:
        """Returns the terms of the PauliSum ordered by coefficient absolute value.

        Args:
            ham: A PauliSum.
        Returns:
            a list of PauliStrings sorted by the absolute value of their coefficient.
        """
        return sorted([term for term in ham], key=lambda x: abs(x.coefficient), reverse=True)


def get_si_sets(ham: cirq.PauliSum, k: int = 1) -> List[List[cirq.PauliString]]:
        """Returns grouping from the sorted insertion algorithm [https://quantum-journal.org/papers/q-2021-01-20-385/].

        Args:
            op: The observable to group.
            k: The integer k in k-commutativity.
        """

        qubits = sorted(set(ham.qubits))
        blocks = compute_blocks(qubits, k)

        commuting_sets = []
        for pstring in get_terms_ordered_by_abscoeff(ham):
                found_commuting_set = False

                for commset in commuting_sets:
                        cant_add = False

                        for pauli in commset:
                                if not commutes(pstring, pauli, blocks):
                                        cant_add = True
                                        break

                        if not cant_add:
                                commset.append(pstring)
                                found_commuting_set = True
                                break

                if not found_commuting_set:
                        commuting_sets.append([pstring])

        return commuting_sets


def compute_blocks(qubits, k):
        return [qubits[k * i: k * (i + 1)] for i in range(math.ceil(len(qubits) / k))]


def compute_rhat(groupings):
        r_numerator = 0
        r_denominator = 0
        for group in groupings:
                if isinstance(group, cirq.PauliSum):
                        a_ij = sum([term.coefficient for term in group])
                        r_numerator += abs(a_ij)
                        r_denominator += np.sqrt(abs(a_ij) ** 2)
                else:
                        a_ij = np.array([term.coefficient for term in group])
                        group_sum = np.sum(np.abs(a_ij))
                        group_sum_squares = np.sum(np.abs(a_ij) ** 2)
                        r_numerator += group_sum
                        r_denominator += np.sqrt(group_sum_squares)
        return (r_numerator / r_denominator) ** 2


def read_openfermion_hdf5(fname_hdf5: str, key: str, optype=of.QubitOperator):
        """
        Read any openfermion operator object from HDF5 file at specified key.
        'optype' is the op class, can be of.QubitOperator or of.FermionOperator.
        """

        with h5py.File(fname_hdf5, 'r', libver='latest') as f:
                op = optype(f[key][()].decode("utf-8"))
        return op


def read_qiskit_hdf5(fname_hdf5: str, key: str):
        """
        Read the operator object from HDF5 at specified key to qiskit SparsePauliOp
        format .
        """
        def _generate_string(term) :
                # change X0 Z3 to XIIZ
                indices = [
                (m.group(1), int(m.group(2)))
                for m in re.finditer(r'([A-Z])(\d +)', term )
                ]
                return ''. join (
                [next ((char for char , idx in indices if idx == i), 'I')
                for i in range (max (idx for _ , idx in indices) + 1) ]
                )
        def _append_ids (pstrings) :
                # append Ids to strings
                return [p + 'I' * (max(map(len, pstrings)) - len(p)) for p in pstrings]

        with h5py.File(fname_hdf5, 'r', libver = 'latest') as f:
                pattern = r'\(([^)]+)\) \[([^]]*)\]'
                matches = re.findall(pattern, f[key][()].decode("utf-8"))
                labels = [_generate_string(m[1]) for m in matches]
                coeffs = [float(match[0]) for match in matches]
                op = SparsePauliOp(_append_ids(labels), coeffs)
        return op



def read_qiskit_hdf5_test(fname_hdf5: str, key: str):
        # Open the HDF5 file and read the data
        with h5py.File(fname_hdf5, 'r', libver='latest') as f:
                # Ensure the key exists in the file
                if key in f:
                        # Read the data
                        data = f[key][()].decode("utf-8")

                        # Print the data for inspection
                        print("Data from HDF5 file:")
                        print(data)

                        # Define the regular expression pattern
                        pattern = r'([\d.]+) \[([ ^\]]+) \]'

                        # Find matches
                        matches = re.findall(pattern, data)

                        # Print matches for debugging
                        print("Matches found:")
                        print(matches)
                else:
                        print(f"Key '{key}' not found in the HDF5 file.")



def parse_through_hdf5(func):
        """
        Decorator function that iterates through an HDF5 file and performs
        the action specified by ‘ func ‘ on the internal and leaf nodes in the HDF5 file.
        """

        def wrapper(obj, path='/', key=None):
                if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
                        for ky in obj.keys():
                                func(obj, path, key=ky, leaf=False)
                                wrapper(obj=obj[ky], path=path + ky + ',', key=ky)
                elif type(obj) == h5py._hl.dataset.Dataset:
                        func(obj, path, key=None, leaf=True)

        return wrapper


def get_hdf5_keys(fname_hdf5: str):
        """ Get a list of keys to all datasets stored in the HDF5 file .
        Args
        ----
        fname_hdf5 ( str ) : full path where HDF5 file is stored
        """

        all_keys = []

        @parse_through_hdf5
        def action(obj, path='/', key=None, leaf=False):
                if leaf is True:
                        all_keys.append(path)

        with h5py.File(fname_hdf5, 'r') as f:
                action(f['/'])
        return all_keys


def preprocess_hamiltonian(
        hamiltonian: of.QubitOperator,
        drop_term_if=None,
) -> cirq.PauliSum:
        """Drop identity terms from the Hamiltonian and convert to Cirq format.
        """
        if drop_term_if is None:
                drop_term_if = []

        new = cirq.PauliSum()

        for term in hamiltonian.terms:
                add_term = True

                for drop_term in drop_term_if:
                        if drop_term(term):
                                add_term = False
                                break

                if add_term:
                        key = " ".join(pauli + str(index) for index, pauli in term)
                        new += next(iter(of.transforms.qubit_operator_to_pauli_sum(
                                of.QubitOperator(key, hamiltonian.terms.get(term))
                        )))

        return new


def get_bit(value, bit):
        return value >> bit & 1


def convert_to_stim_strings(group, k, qubits):
        """Convert the group to Stim strings that can be used to generate
        the tableau.

        Args
        ----
            group:
                group of k-commuting Paulis
            k:
                value of k
            qubits:
                qubits hamiltonian acts on
        """

        # Compute the blocks of size k
        blocks = compute_blocks(qubits, k)

        # Compute the Pauli strings for Stim.
        all_strings = []
        for block in blocks:
                block_strings = []
                # print(f'block: {block}')
                for i, ps in enumerate(group):
                        ps = restrict_to(ps, block)
                        dps = ps.dense(block)
                        ss = dps.__str__().replace("I", "_")
                        # if any(s in ss for s in ["X", "Y", "Z"]):
                        #         block_strings.append(ss)
                        # ss = ss[-len(qubits):]
                        block_strings.append(ss)
                all_strings.append(block_strings)
        # print(all_strings)
        return all_strings


def compute_measurement_circuit_depth(stim_strings):
        """Generate the measurement circuits for every block and compute
        their optimized depth.

        Args
        ----
            stim_strings:
                nested list generated with "convert_to_stim_strings"

        Returns
        -------
            optimized depth
        """
        all_depths = []

        for block_strings in stim_strings:
                # Compute tableau and measurement circuit
                if not block_strings:
                        continue
                signs = 0
                result = False

                while not result and signs < 2 ** (len(block_strings[0]) - 1):
                        try:
                                signs += 1
                                stim_tableau = stim.Tableau.from_stabilizers(
                                        [stim.PauliString(
                                                ('-' if get_bit(signs - 1, i) else '+') + stim_str[1:]) for i, stim_str
                                                in enumerate(block_strings)
                                        ],
                                        allow_redundant=True,
                                        allow_underconstrained=True
                                )
                                result = True
                        except ValueError:
                                pass
                if result:
                        stim_circuit = stimcirq.stim_circuit_to_cirq_circuit(
                                stim_tableau.to_circuit(method="elimination")
                        )
                        # Optimize to gate set and compute depth
                        opt_circuit = cirq.optimize_for_target_gateset(
                                stim_circuit, gateset=cirq.CZTargetGateset()
                        )
                        depth = len(cirq.Circuit(opt_circuit.all_operations()))
                        all_depths.append(depth)
                # else:
                #    raise RuntimeWarning('No independent set of stabilizers found.')

        return all_depths


def compute_measurement_circuit(stim_strings):
        """Generate the measurement circuits for every block and compute
        their optimized depth.

        Args
        ----
            stim_strings:
                nested list generated with "convert_to_stim_strings"

        Returns
        -------
            optimized depth
        """
        all_circuits = []
        all_depths = []
        for block_strings in stim_strings:
                # Compute tableau and measurement circuit
                if not block_strings:
                        continue
                signs = 0
                result = False

                while not result and signs < 2 ** (len(block_strings[0]) - 1):
                        try:
                                signs += 1
                                stim_tableau = stim.Tableau.from_stabilizers(
                                        [stim.PauliString(
                                                ('-' if get_bit(signs - 1, i) else '+') + stim_str[1:]) for i, stim_str
                                        in enumerate(block_strings)
                                        ],
                                        allow_redundant=True,
                                        allow_underconstrained=True
                                )
                                result = True
                        except ValueError:
                                pass
                if result:
                        stim_circuit = stimcirq.stim_circuit_to_cirq_circuit(
                                stim_tableau.to_circuit(method="elimination")
                        )
                        qc = QuantumCircuit.from_qasm_str(stim_circuit.to_qasm())
                        qc_t = transpile(qc, optimization_level=3, )
                        print(qc_t)
                        all_circuits.append(qc)
                        opt_circuit = cirq.optimize_for_target_gateset(
                                stim_circuit, gateset=cirq.CZTargetGateset()
                        )
                        depth = len(cirq.Circuit(opt_circuit.all_operations()))
                        all_depths.append(depth)
                # else:
                #    raise RuntimeWarning('No independent set of stabilizers found.')

        return all_circuits


def measurement_circuit_depth(groupings, k, qubits):
        """Compute the maximum circuit depth."""
        depths = []
        for group in groupings:
                blocked_stim_strings = convert_to_stim_strings(group, k, qubits)
                blocked_circuit_depths = compute_measurement_circuit_depth(blocked_stim_strings)
                if blocked_circuit_depths:
                        depths.append(max(blocked_circuit_depths))
        return max(depths)  # Only report the maximum


def threshold_matrix_elements(matrix, threshold=1e-5):
        # Create a copy of the matrix to avoid modifying the original one
        processed_matrix = np.copy(matrix)

        # Apply thresholding to make values exactly 1 or -1
        processed_matrix[np.abs(processed_matrix - 1) <= threshold] = 1
        processed_matrix[np.abs(processed_matrix + 1) <= threshold] = -1

        return processed_matrix


def is_diagonal(matrix):
        """
        Check if a given matrix is a diagonal matrix.

        Parameters:
        matrix (np.ndarray): The matrix to check.

        Returns:
        bool: True if the matrix is diagonal, False otherwise.
        """
        # Convert the matrix to a NumPy array if it isn't already
        matrix = np.array(matrix)

        # Check if the matrix is square
        if matrix.shape[0] != matrix.shape[1]:
                return False

        # Check if all off-diagonal elements are zero
        return np.all(matrix == np.diag(np.diagonal(matrix)))



def transform_observables(observable, unitary):
        # diagonalize the observable given the unitary of the measurement circuit
        # observable_reverse = observable[::-1]
        obs_mat = I if isinstance(observable[0], IdentityGate) else paulis[observable[0]._name]
        for opt in observable[1:]:
                if isinstance(opt, IdentityGate):
                        obs_mat = np.kron(obs_mat, I)
                else:
                        obs_mat = np.kron(obs_mat, paulis[opt._name])
        new_u = unitary @ obs_mat @ unitary.conj().T
        new_u_round = np.round(new_u, 3)
        diag = is_diagonal(new_u_round)
        if diag == False:
                print(f'not diagonalized the observable {observable}')
                # exit()
        return new_u

def measurement_circuit(groupings, k, qubits):
        """Compute the maximum circuit depth."""
        circuits = []
        new_all_obs = []
        old_all_obs = []
        for idx, group in enumerate(groupings):
                observables = []
                for op in group:
                        ps = restrict_to(op, qubits)
                        dps = ps.dense(qubits)
                        observables.append(dps)
                        # print(dps)
                if idx == 0:
                        identity_key = ''.join('I' for _ in range(len(qubits)))
                        if identity_key in pauli_coeffs.keys():
                                observables.append(DensePauliString(identity_key, coefficient=pauli_coeffs[identity_key]))
                old_all_obs.append(observables)
                blocked_stim_strings = convert_to_stim_strings(group, k, qubits)
                measurement_circuits = compute_measurement_circuit(blocked_stim_strings)
                print(blocked_stim_strings)

                circuits.append(measurement_circuits)
                mat = Operator.from_circuit(measurement_circuits[0]).data
                for circuit in measurement_circuits[1:]:
                        circop = Operator.from_circuit(circuit)
                        mat = np.kron(mat, circop.data)
                new_group_obs = []
                for op in observables:
                        new_obs = transform_observables(op, mat)
                        new_group_obs.append(new_obs)
                new_all_obs.append(new_group_obs)
                # print(observables)
                # print('------END------')
        return circuits, new_all_obs, old_all_obs


def expectation_value(observable, counts):
    # calculate the expectation value
    shots = sum(counts.values())
    exp_val = 0
    for outcome, count in counts.items():
        decimal_number = int(outcome, 2)
        sign = observable[decimal_number][decimal_number]
        exp_val += sign * count / shots
    return exp_val


def compute_groups(k):
        return get_si_sets(hamiltonian, k)


if __name__ == "__main__":
        warnings.filterwarnings('ignore')
        data_directory: str = "/home/siyuanniu/LBNL/projects/HamPerf/"
        extension: str = ".hdf5"
        fnames_encodings = {
                # "Li2": '/ham_BK4',
                'H2': '/ham_BK-4',
                # 'BeH': '/ham_BK4',
                # 'LiH': '',
                # "Be2": "/ham_BK-6",
        }

        # Create noise model for noisy simulation
        noise_cx = 0.01
        noise_1q = 0.001
        noise_model = NoiseModel()
        error_cx = depolarizing_error(noise_cx, 2)
        noise_model.add_all_qubit_quantum_error(error_cx, ['cx'])
        error_1q = depolarizing_error(noise_1q, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3'])

        estimator = Estimator(AerSimulator(), options={"default_shots": 10000})

        np.random.seed(42)

        for fname, path in fnames_encodings.items():
                hamiltonian = read_openfermion_hdf5(
                        os.path.join(data_directory, fname + extension),
                        # change key indices we have different encoding ways
                        get_hdf5_keys(os.path.join(data_directory, fname + extension))[2].rstrip(","),
                        # Or # fnames_encodings[fname]
                )
                data = extract_dataset_hdf5(f"/home/siyuanniu/LBNL/projects/HamPerf/{fname + extension}", f"{path}")

                # Get the Hamiltonian operator as SparsePauliOp and its size from the data
                ham_op, num_qubits = process_data(data)
                print(ham_op)

                pauli_coeffs = {}

                for p, c in zip(ham_op.paulis, ham_op.coeffs):
                        pauli_coeffs[p.settings['data']] = c

                # Create ansatz for VQE
                ansatz = EfficientSU2(ham_op.num_qubits).decompose()

                hamiltonian = preprocess_hamiltonian(hamiltonian, drop_term_if=[lambda term: term == ()])
                nqubits = get_num_qubits(hamiltonian)
                qubits = sorted(set(hamiltonian.qubits))
                nterms = len(hamiltonian)

                print(f"Hamiltonian has {nterms} term(s) and acts on {nqubits} qubit(s).")

                grouping_algorithms = {
                        1: "1-qubit-wise commuting",
                        nqubits // 4: f"{nqubits // 4}-qubit-wise commuting",
                        nqubits // 2: f"{nqubits // 2}-qubit-wise commuting",
                        3 * nqubits // 4: f"{3 * nqubits // 4}-qubit-wise commuting",
                        nqubits: "Fully commuting",
                }
                metric_groups = {
                        label: (compute_groups(k), k, qubits) for k, label in grouping_algorithms.items()
                }
                meas_circuits = {}
                for label, groups in metric_groups.items():
                        print('label:', label)
                        meas_circuits[label] =  measurement_circuit(groups[0], groups[1], groups[2])

                for key, value in meas_circuits.items():
                        print(f'---method for {key}-----')
                        circuits, new_all_obs, old_all_obs  = value
                        ansatz.barrier()
                        ansatz_mes_circuits = []

                        for circs in circuits:
                                qubit_idx = 0
                                ansatz_mes = ansatz.copy()
                                for circ in circs:
                                        ansatz_mes.compose(circ, qubits=list(range(qubit_idx, qubit_idx + circ.num_qubits)), inplace=True)
                                        qubit_idx += circ.num_qubits
                                ansatz_mes.measure_active()
                                ansatz_mes_circuits.append(ansatz_mes)

                        #
                        initial_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
                        bound_circuits = [ansatz_mes_circuit.assign_parameters(initial_params) for ansatz_mes_circuit in ansatz_mes_circuits]
                        results = AerSimulator().run(bound_circuits).result().get_counts()

                        # Calculate the expectation value
                        exp = 0
                        for res, new_obs, old_obs in zip(results, new_all_obs, old_all_obs):
                                for new_ob, old_ob in zip(new_obs, old_obs):
                                        old_ob_key = old_ob.__str__()[-len(qubits):]
                                        exp += expectation_value(new_ob, res) * pauli_coeffs[old_ob_key]
                                        # print(f'key: {old_ob_key}, coef: {pauli_coeffs[old_ob_key]}, exp is {exp}')

                        print('expectation value:', exp)

                        #Use qiskit function to calculate and check the results
                pub = (ansatz, [ham_op], [initial_params])
                result = estimator.run(pubs=[pub]).result()

                # Get results for the first (and only) PUB
                energy = result[0].data.evs[0]

                print(f'expectation value calculated by qiskit: {energy}')




