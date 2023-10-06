from __future__ import annotations

import json

import numpy as np

from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy


def chain(
    n: int = 4,
    r: float = 1.0,
) -> tuple[list[str], np.ndarray, str]:
    """
    Generate the xyz coordinates of a chain of hydrogen atoms.
    """

    xyz = np.zeros((n, 3))
    atoms = ["H"] * n

    z = np.arange(-(n - 1) / 2, (n) / 2) * r
    assert len(z) == n
    assert sum(z) == 0.0

    xyz[:, 2] = z

    description = "H" + str(n) + " 1D chain, " + str(r) + " Angstroms\n"

    return atoms, xyz, description


def as_xyz(atoms, xyz, description="\n"):
    # format as .XYZ
    n = len(atoms)
    pretty_xyz = str(n) + "\n"
    pretty_xyz += description
    for i in range(n):
        pretty_xyz += "{0:s} {1:10.4f} {2:10.4f} {3:10.4f}\n".format(
            atoms[i], xyz[i, 0], xyz[i, 1], xyz[i, 2]
        )

    return pretty_xyz


def molecule_to_problem(
    atoms: list[str],
    xyz: np.ndarray,
) -> ElectronicStructureProblem:
    """
    Creates an ElectronicEnergy object corresponding to a list of atoms and xyz positions.

    This object can be used to access hamiltonian information.
    """

    # define the molecular structure
    hydrogen_molecule = MoleculeInfo(atoms, xyz, charge=0, multiplicity=1)

    # Prepare and run the initial Hartree-Fock calculation on molecule
    molecule_driver = PySCFDriver.from_molecule(hydrogen_molecule, basis="sto3g")
    quantum_molecule = molecule_driver.run()

    # acquire fermionic hamiltonian
    return quantum_molecule


def generate_jordan_wigner_hamiltonian(hamiltonian: ElectronicEnergy) -> PauliSumOp:
    """
    Returns an ElectronicEnergy hamiltonian in the Jordan Wigner encoding basis.
    """
    fermionic_hamiltonian = hamiltonian.second_q_op()

    # create mapper from fermionic to spin basis
    mapper = JordanWignerMapper()

    # create qubit hamiltonian from fermionic one
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    # hamiltonian does not include scalar nuclear energy constant, so we add it back here for consistency in benchmarks
    qubit_hamiltonian += PauliSumOp(
        SparsePauliOp(
            "I" * qubit_hamiltonian.num_qubits,
            coeffs=hamiltonian.nuclear_repulsion_energy,
        )
    )
    return qubit_hamiltonian.reduce()


def generate_paired_qubit_hamiltonian(hamiltonian: ElectronicEnergy) -> PauliSumOp:
    """
    Returns an ElectronicEnergy hamiltonian in the efficient paired electronic basis.
    """

    one_body_integrals = hamiltonian.electronic_integrals.alpha["+-"]
    two_body_integrals = hamiltonian.electronic_integrals.alpha["++--"].transpose(
        (0, 3, 2, 1)
    )
    core_energy = hamiltonian.nuclear_repulsion_energy

    num_orbitals = len(one_body_integrals)

    def I():
        return "I" * num_orbitals

    def Z(p):
        Zp = ["I"] * num_orbitals
        Zp[p] = "Z"
        return "".join(Zp)[::-1]

    def ZZ(p, q):
        ZpZq = ["I"] * num_orbitals
        ZpZq[p] = "Z"
        ZpZq[q] = "Z"
        return "".join(ZpZq)[::-1]

    def XX(p, q):
        XpXq = ["I"] * num_orbitals
        XpXq[p] = "X"
        XpXq[q] = "X"
        return "".join(XpXq)[::-1]

    def YY(p, q):
        YpYq = ["I"] * num_orbitals
        YpYq[p] = "Y"
        YpYq[q] = "Y"
        return "".join(YpYq)[::-1]

    terms = [((I(), core_energy))]  # nuclear repulsion is a constant

    # loop to create paired electron Hamiltonian
    # last term is from p = p case in (I - Zp) * (I - Zp)* (pp|qq)
    gpq = (
        one_body_integrals
        - 0.5 * np.einsum("prrq->pq", two_body_integrals)
        + np.einsum("ppqq->pq", two_body_integrals)
    )
    for p in range(num_orbitals):
        terms.append((I(), gpq[p, p]))
        terms.append((Z(p), -gpq[p, p]))
        for q in range(num_orbitals):
            if p != q:
                terms.append(
                    (
                        I(),
                        0.5 * two_body_integrals[p, p, q, q]
                        + 0.25 * two_body_integrals[p, q, q, p],
                    )
                )
                terms.append(
                    (
                        Z(p),
                        -0.5 * two_body_integrals[p, p, q, q]
                        - 0.25 * two_body_integrals[p, q, q, p],
                    )
                )
                terms.append(
                    (
                        Z(q),
                        -0.5 * two_body_integrals[p, p, q, q]
                        + 0.25 * two_body_integrals[p, q, q, p],
                    )
                )
                terms.append(
                    (
                        ZZ(p, q),
                        0.5 * two_body_integrals[p, p, q, q]
                        - 0.25 * two_body_integrals[p, q, q, p],
                    )
                )
                terms.append((XX(p, q), 0.25 * two_body_integrals[p, q, p, q]))
                terms.append((YY(p, q), 0.25 * two_body_integrals[p, q, p, q]))

    qubit_hamiltonian = PauliSumOp.from_list(terms)  # pass in list of terms

    return qubit_hamiltonian.reduce()  # Qiskit can collect and simplify terms


def make_pauli_dict(hamiltonian):
    op = hamiltonian.primitive.to_list()

    n_terms = len(op)
    coeffs = []
    paulis = []

    for i in range(n_terms):
        coeffs.append(complex(op[i][1]).real)
        paulis.append(op[i][0])

    pauli_dict = dict(zip(paulis, coeffs))

    return pauli_dict


def make_energy_orbital_dict(hamiltonian):
    d = {
        "hf_energy": float(hamiltonian.reference_energy),
        "nuclear_repulsion_energy": float(hamiltonian.nuclear_repulsion_energy),
        "spatial_orbitals": int(hamiltonian.num_spatial_orbitals),
        "alpha_electrons": int(hamiltonian.num_alpha),
        "beta_electrons": int(hamiltonian.num_beta),
    }

    return d


if __name__ == "__main__":
    """
    generate various hydrogen lattice pUCCD hamiltonians
    """
    for shape in [chain]:
        for n in [2, 4, 6, 8, 10, 12]:
            for r in [0.75, 1.00, 1.25]:
                file_name = f"h{n:03}_{shape.__name__}_{r:06.2f}"
                file_name = file_name.replace(".", "_")
                file_name = file_name + ".json"
                print(f"Working on {shape.__name__} for n={n} and r={r}")

                # get lattice info from a particular shape
                atoms, xyz, description = shape(n, r)
                # print to console the lattice info
                print(as_xyz(atoms, xyz, description))
                # create hamiltonian from lattice info
                electronic_problem = molecule_to_problem(atoms, xyz)

                fermionic_hamiltonian = electronic_problem.hamiltonian

                # generate information on number of orbitals and nuclear + HF energies
                orbital_energy_info = make_energy_orbital_dict(electronic_problem)

                # generate hamiltonian in paired basis (i.e., hard-core boson)
                paired_hamiltonian = make_pauli_dict(
                    generate_paired_qubit_hamiltonian(fermionic_hamiltonian)
                )

                # generate hamiltonian in spin-orbital (Jordan-Wigner) basis
                jordan_wigner_hamiltonian = make_pauli_dict(
                    generate_jordan_wigner_hamiltonian(fermionic_hamiltonian)
                )

                # begin JSON file creation
                # this JSON file format can be changed to add/remove categories in the future

                # create a dict and merge it with the orbital energy dict
                d = {
                    "atoms": atoms,
                    "x": xyz[:, 0].tolist(),
                    "y": xyz[:, 1].tolist(),
                    "z": xyz[:, 2].tolist(),
                    "paired_hamiltonian": paired_hamiltonian,
                    "jordan_wigner_hamiltonian": jordan_wigner_hamiltonian,
                }

                d = {**d, **orbital_energy_info}

                # save file in the working directory
                with open(file_name, "w") as f:
                    json.dump(d, f, indent=4)
                # end JSON file creation
