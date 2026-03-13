"""Implementation of the pair unitary CCD (pUCCD) VQE ansatz."""
import itertools
from typing import Any, List, Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.opflow.primitive_ops import PauliSumOp


class PUCCD(object):
    """Class for the pUCCD circuit."""

    @staticmethod
    def generate_mock_hamiltonian(num_qubits: int) -> PauliSumOp:
        """
        Generate a mock pUCCD Hamiltonian for a given number of qubits assuming half-filling.

        This function generates a mock Hamiltonian by constructing Pauli strings with assigned coefficients
        based on the positions of "Z" and "X" operators in the strings. The resulting Hamiltonian is a
        PauliSumOp object.

        Parameters
        ----------
        num_qubits : :obj:`int`
            The number of qubits in the Hamiltonian.

        Returns
        -------
        :obj:`PauliSumOp`
            A mock Hamiltonian represented as a sum of Pauli operators.

        Notes
        -----
        The Hamiltonian is designed to mimic a hydrogen chain lattice in the hard core boson (paired) approximation.
        It should only require three circuits to measure: <Z>, <XX>, and <YY> bases.
        Random coefficients are scaled to be physically reasonable given the term, though YMMV.
        Example Hamiltonian for num_qubits = 3:

        ```
        -0.75 * III
        - 0.2623059170590116 * ZII
        - 0.2417902291649512 * IZI
        + 0.24659271000597804 * IIZ
        + 0.015958046162329265 * XXI
        + 0.015958046162329265 * YYI
        + 0.1392659280805608 * ZZI
        + 0.038832267509665 * XIX
        + 0.038832267509665 * YIY
        + 0.030688157709100217 * ZIZ
        + 0.003677182811813895 * IXX
        + 0.003677182811813895 * IYY
        + 0.17675281046088037 * IZZ
        ```

        """

        def add_term(label, coeff=None):
            if coeff is None:
                if label.count("Z") == 2:
                    coeff = np.random.uniform(0, 0.25)
                elif label.count("Z") == 1:
                    pos = label.find("Z")
                    if pos < num_qubits / 2:
                        coeff = np.random.uniform(-0.5, 0.0)
                    else:
                        coeff = np.random.uniform(0, 0.5)

                # coeff = np.random.uniform(-, 1)
            terms.append((label, coeff))

        labels = ["I", "X", "Y", "Z"]
        terms: List[Any] = list()
        added_terms = set()

        label_str = "I" * num_qubits
        add_term(label_str, coeff=-0.75)

        for length in range(1, num_qubits + 1):
            for positions in itertools.combinations(range(num_qubits), length):
                for paulis in itertools.product(labels[1:], repeat=length):
                    label = ["I"] * num_qubits
                    for pos, pauli in zip(positions, paulis):
                        label[pos] = pauli
                    label_str = "".join(label)

                    if label_str in added_terms:
                        continue

                    if all([x in ("I", "Z") for x in label]) and label.count("Z") in [
                        1,
                        2,
                    ]:
                        add_term(label_str)
                        added_terms.add(label_str)
                    elif all([x in ("I", "X") for x in label]) and label.count("X") == 2:
                        coeff = np.random.uniform(0, 0.05)  # from hydrogen chain
                        add_term(label_str, coeff)
                        added_terms.add(label_str)
                        y_label_str = "".join(["Y" if p == "X" else p for p in label])
                        add_term(y_label_str, coeff)
                        added_terms.add(y_label_str)

        return PauliSumOp.from_list(terms)

    def build_circuit(
        self,
        num_qubits: int,
        num_occ_pairs: Optional[int] = None,
        operator: Optional[PauliSumOp] = None,
        *args,
        **kwargs
    ) -> QuantumCircuit:
        """
        Create the pUCCD ansatz quantum circuit for the VQE algorithm.

        Parameters
        ----------
        num_qubits  :  :obj:`int`
            Number of qubits in the quantum circuit.
        num_occ_pairs  :  :obj:`Optional[int]`
            Number of occupied pairs. If not provided, it will be integer division of qubits by half
        operator  :  :obj:`Optional[int]`
            Quantum operator. Assumes that operator is in computational basis (e.g. diagonal).
            If not provided, it will be generated using ``generate_mock_operator``.

        Returns
        -------
        :obj:`QuantumCircuit`
            The constructed quantum circuit for the pUCCD ansatz.
        """
        if num_occ_pairs is None:
            self.num_occ_pairs = num_qubits // 2  # e.g., half-filling, which is a reasonable chemical case

        # do all possible excitations if not passed a list of excitations directly
        excitation_pairs = []
        for i in range(self.num_occ_pairs):
            for a in range(self.num_occ_pairs, num_qubits):
                excitation_pairs.append([i, a])

        circuit = QuantumCircuit(num_qubits)

        # Hartree Fock initial state
        for occ in range(self.num_occ_pairs):
            circuit.x(occ)

        parameter_vector = ParameterVector("t", length=len(excitation_pairs))

        for idx, pair in enumerate(excitation_pairs):
            # parameter
            theta = parameter_vector[idx]

            # apply excitation
            i, a = pair[0], pair[1]

            # implement the magic gate
            circuit.s(i)
            circuit.s(a)
            circuit.h(a)
            circuit.cx(a, i)

            # Ry rotation
            circuit.ry(theta, i)
            circuit.ry(theta, a)

            # implement M^-1
            circuit.cx(a, i)
            circuit.h(a)
            circuit.sdg(a)
            circuit.sdg(i)

        return circuit
