import json

import numpy as np

from qiskit.opflow.primitive_ops import PauliSumOp

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper


def ring(n: int = 10, r: float = 1.0) -> tuple[list[str], np.ndarray, str]:
    """Generate the xyz coordinates of a ring of hydrogen atoms."""
    xyz = np.zeros((n, 3))
    atoms = ["H"] * n

    angle = 2 * np.pi / n
    rad_denom = 2 * np.sin(np.pi / n)
    rad = r / rad_denom

    x = [rad * np.cos(angle * i) for i in range(n)]
    y = [rad * np.sin(angle * i) for i in range(n)]

    xyz[:, 0] = x
    xyz[:, 1] = y

    description = "H" + str(n) + " ring, " + str(r) + " Angstroms\n"

    return atoms, xyz, description


def h10_pyramid(n: int = 10, r: float = 1.0) -> tuple[list[str], np.ndarray, str]:
    """
    Generate the xyz coordinates of a pyramid of 10 (and only 10) hydrogen atoms.
    """
    # hardcoded for n = 10 right now
    xyz = np.zeros((10, 3))
    atoms = ["H"] * 10

    hpyr = np.sqrt(2.0 / 3) * r
    htri = np.cos(np.pi / 6.0) * r

    xyz[0, 0] = 0.0
    xyz[1, 0] = -0.5 * r
    xyz[2, 0] = 0.5 * r
    xyz[3, 0] = -r
    xyz[4, 0] = 0.0
    xyz[5, 0] = r
    xyz[6, 0] = 0.0
    xyz[7, 0] = -0.5 * r
    xyz[8, 0] = 0.5 * r
    xyz[9, 0] = 0.0

    xyz[0, 1] = (4.0 / 3) * htri
    xyz[1, 1] = (1.0 / 3) * htri
    xyz[2, 1] = (1.0 / 3) * htri
    xyz[3, 1] = -(2.0 / 3) * htri
    xyz[5, 1] = -(2.0 / 3) * htri
    xyz[6, 1] = (2.0 / 3) * htri
    xyz[7, 1] = -(1.0 / 3) * htri
    xyz[8, 1] = -(1.0 / 3) * htri
    xyz[9, 1] = 0.0

    xyz[0, 2] = -hpyr
    xyz[1, 2] = -hpyr
    xyz[2, 2] = -hpyr
    xyz[3, 2] = -hpyr
    xyz[4, 2] = -hpyr
    xyz[5, 2] = -hpyr
    xyz[6, 2] = 0.0
    xyz[7, 2] = 0.0
    xyz[8, 2] = 0.0
    xyz[9, 2] = hpyr

    description = "H10 pyramid, " + str(r) + " Angstroms\n"

    return atoms, xyz, description


def h10_sheet(n: int = 10, r: float = 1.0) -> tuple[list[str], np.ndarray, str]:
    """
    Generate the xyz coordinates of a sheet of 10 (and only 10) hydrogen atoms.
    """
    # hardcoded for n = 10 right now
    xyz = np.zeros((10, 3))
    atoms = ["H"] * 10

    htri = np.cos(np.pi / 6.0) * r

    xyz[0, 0] = -r
    xyz[1, 0] = 0.0
    xyz[2, 0] = r
    xyz[3, 0] = -(3.0 / 2) * r
    xyz[4, 0] = -r / 2
    xyz[5, 0] = r / 2
    xyz[6, 0] = (3.0 / 2) * r
    xyz[7, 0] = -r
    xyz[8, 0] = 0.0
    xyz[9, 0] = r

    xyz[0, 1] = htri
    xyz[1, 1] = htri
    xyz[2, 1] = htri
    xyz[3, 1] = 0.0
    xyz[4, 1] = 0.0
    xyz[5, 1] = 0.0
    xyz[6, 1] = 0.0
    xyz[7, 1] = -htri
    xyz[8, 1] = -htri
    xyz[9, 1] = -htri

    description = "H10 2D sheet, " + str(r) + " Angstroms\n"

    return atoms, xyz, description


def chain(
    n: int = 10,
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
        pretty_xyz += "{0:s} {1:10.4f} {2:10.4f} {3:10.4f}\n".format(atoms[i], xyz[i, 0], xyz[i, 1], xyz[i, 2])

    return pretty_xyz


def molecule_to_hamiltonian(
    atoms: list[str],
    xyz: np.ndarray,
) -> ElectronicEnergy:
    """
    Creates an ElectronicEnergy object corresponding to a list of atoms and xyz positions.

    This object can be used to access hamiltonian information.
    """

    # define the molecular structure
    hydrogen_molecule = MoleculeInfo(atoms, xyz, charge=0, multiplicity=1)

    # Prepare and run the initial Hartree-Fock calculation on molecule
    molecule_driver = PySCFDriver.from_molecule(hydrogen_molecule, basis="sto3g")
    # print("Spin:", molecule_driver.spin)
    # print("Atom:", molecule_driver.atom)
    quantum_molecule = molecule_driver.run()

    # acquire fermionic hamiltonian
    return quantum_molecule.hamiltonian


def generate_spin_qubit_hamiltonian(hamiltonian: ElectronicEnergy) -> PauliSumOp:
    """
    Returns an ElectronicEnergy hamiltonian in the spin basis. (Not in the spatial basis.)
    """
    fermionic_hamiltonian = hamiltonian.second_q_op()

    # create mapper from fermionic to spin basis
    mapper = JordanWignerMapper()

    # create qubit hamiltonian from fermionic one
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    return qubit_hamiltonian


def generate_paired_electron_hamiltonian(hamiltonian: ElectronicEnergy) -> PauliSumOp:
    """
    Returns an ElectronicEnergy hamiltonian in the efficient paired electronc basis.
    """

    one_body_integrals = hamiltonian.electronic_integrals.alpha["+-"]
    two_body_integrals = hamiltonian.electronic_integrals.alpha["++--"].transpose((0, 3, 2, 1))
    core_energy = quantum_molecule.hamiltonian.nuclear_repulsion_energy

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
        one_body_integrals - 0.5 * np.einsum("prrq->pq", two_body_integrals) + np.einsum("ppqq->pq", two_body_integrals)
    )
    for p in range(num_orbitals):
        terms.append((I(), gpq[p, p]))
        terms.append((Z(p), -gpq[p, p]))
        for q in range(num_orbitals):
            if p != q:
                terms.append((I(), 0.5 * two_body_integrals[p, p, q, q] + 0.25 * two_body_integrals[p, q, q, p]))
                terms.append((Z(p), -0.5 * two_body_integrals[p, p, q, q] - 0.25 * two_body_integrals[p, q, q, p]))
                terms.append((Z(q), -0.5 * two_body_integrals[p, p, q, q] + 0.25 * two_body_integrals[p, q, q, p]))
                terms.append((ZZ(p, q), 0.5 * two_body_integrals[p, p, q, q] - 0.25 * two_body_integrals[p, q, q, p]))
                terms.append((XX(p, q), 0.25 * two_body_integrals[p, q, p, q]))
                terms.append((YY(p, q), 0.25 * two_body_integrals[p, q, p, q]))

    qubit_hamiltonian = PauliSumOp.from_list(terms)  # pass in list of terms

    return qubit_hamiltonian.reduce()  # Qiskit can collect and simplify terms


if __name__ == "__main__":
    """
    generate various hydrogen lattice pUCCD hamiltonians
    """
    #    for shape in [chain, ring, h10_sheet, h10_pyramid]:
    for shape in [chain]:
        for n in [10]:
            for r in [1.0]:
                # define json file name and print. for h10 sheet/pyramid n = 10
                # so no need to set/print it
                if shape in [h10_sheet, h10_pyramid]:
                    file_name = f"h{shape.__name__}_{r}.json"
                    print(f"Working on {shape.__name__}")
                else:
                    file_name = f"h{n}_{shape.__name__}_{r}.json"
                    print(f"Working on {shape.__name__} for n={n} and r={r}")

                # get lattice info from a particular shape
                atoms, xyz, description = shape(n, r)
                # print to console the lattice info
                print(as_xyz(atoms, xyz, description))
                # create hamiltonian from lattice info
                hamiltonian = molecule_to_hamiltonian(atoms, xyz)

                # generate hamiltonian in spin basis-
                # spin_qubit_hamiltonian = generate_spin_qubit_hamiltonian(hamiltonian)

                # generate hamiltonian in paired spin basis
                spin_pair_hamiltonian = generate_spin_qubit_hamiltonian(hamiltonian)

                # begin JSON file creation
                op = spin_pair_hamiltonian.primitive.to_list()

                n_terms = len(op)
                coeffs = []
                paulis = []

                for i in range(n_terms):
                    coeffs.append(complex(op[i][1]).real)
                    paulis.append(op[i][0])

                pauli_dict = dict(zip(paulis, coeffs))

                d = {
                    "positions": {
                        "x": xyz[:, 0].tolist(),
                        "y": xyz[:, 1].tolist(),
                        "z": xyz[:, 2].tolist(),
                    },
                    "hamiltonian": pauli_dict,
                }
                with open(file_name, "w") as f:
                    json.dump(d, f)
                # end JSON file creation
