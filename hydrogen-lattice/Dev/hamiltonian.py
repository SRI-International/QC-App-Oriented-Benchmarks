import json

import numpy as np

from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper


def ring(n: int = 10, r: float = 1.0) -> tuple[list[str], np.ndarray, str]:
    """
    Generate the xyz coordinates of a ring of hydrogen atoms.
    """
    xyz = np.zeros((n, 3))
    atoms = ['H'] * n

    angle = 2 * np.pi / n
    rad_denom = 2 * np.sin(np.pi / n)
    rad = r / rad_denom

    x = [rad * np.cos(angle * i) for i in range(n)]
    y = [rad * np.sin(angle * i) for i in range(n)]

    xyz[:, 0] = x
    xyz[:, 1] = y

    description = "H" + str(n) + " ring, " + str(r) + " Angstroms\n"

    return atoms, xyz, description


def h10_pyramid(
        n: int = 10, r: float = 1.0) -> tuple[list[str], np.ndarray, str]:
    """
    Generate the xyz coordinates of a pyramid of 10 (and only 10) hydrogen atoms.
    """
    # hardcoded for n = 10 right now
    xyz = np.zeros((10, 3))
    atoms = ['H'] * 10

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


def h10_sheet(n: int = 10,
              r: float = 1.0) -> tuple[list[str],
                                       np.ndarray,
                                       str]:
    """
    Generate the xyz coordinates of a sheet of 10 (and only 10) hydrogen atoms.
    """
    # hardcoded for n = 10 right now
    xyz = np.zeros((10, 3))
    atoms = ['H'] * 10

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


def chain(n: int = 10, r: float = 1.0,) -> tuple[list[str], np.ndarray, str]:
    """
    Generate the xyz coordinates of a chain of hydrogen atoms.
    """

    xyz = np.zeros((n, 3))
    atoms = ['H'] * n

    z = np.arange(-(n - 1) / 2, (n) / 2) * r
    assert len(z) == n
    assert sum(z) == 0.0

    xyz[:, 2] = z

    description = "H" + str(n) + " 1D chain, " + str(r) + " Angstroms\n"

    return atoms, xyz, description


def as_xyz(atoms, xyz, description='\n'):
    # format as .XYZ
    n = len(atoms)
    pretty_xyz = str(n) + '\n'
    pretty_xyz += description
    for i in range(n):
        pretty_xyz += "{0:s} {1:10.4f} {2:10.4f} {3:10.4f}\n".format(
                      atoms[i], xyz[i, 0], xyz[i, 1], xyz[i, 2])

    return pretty_xyz


def generate_qubit_hamiltonian(
    atoms: list[str],
    xyz: np.ndarray,
) -> PauliSumOp:
    """
    Create a hamiltonian from a list of atoms and their xyz positions. 
    """

        
    # define the molecular structure
    hydrogen_molecule = MoleculeInfo(atoms, xyz, charge=0, multiplicity=1)

    # Prepare and run the initial Hartree-Fock calculation on molecule
    molecule_driver = PySCFDriver.from_molecule(
        hydrogen_molecule, basis="sto3g")
    # print("Spin:", molecule_driver.spin)
    # print("Atom:", molecule_driver.atom)
    quantum_molecule = molecule_driver.run()

    # acquire fermionic hamiltonian
    fermionic_hamiltonian = quantum_molecule.hamiltonian.second_q_op()

    # create mapper from fermionic to spin basis
    mapper = JordanWignerMapper()

    # create qubit hamiltonian from fermionic one
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    return qubit_hamiltonian


if __name__ == '__main__':

    for shape in [chain, ring, h10_sheet, h10_pyramid]:
        for n in [10]:
            for r in [1.0]:

                # get lattice info from a particular shape
                atoms, xyz, description = shape(n, r)
                print(xyz)
                # log in console
                print(as_xyz(atoms, xyz, description))

                # generate hamiltonian in spin basis
                qubit_hamiltonian = generate_qubit_hamiltonian(atoms, xyz)

                # begin putting spin basis hamiltonian into json
                op = qubit_hamiltonian.primitive.to_list()

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
                    "hamiltonian": pauli_dict
                }

                with open('h' + str(n) + '_' + str(r) + '_' + str(shape.__name__) + '.json', 'w') as f:
                    json.dump(d, f)
                # end putting spin basis hamiltonian into json
