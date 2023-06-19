import numpy as np 
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.problems.second_quantization.electronic import (
    ElectronicStructureProblem,
)
from qiskit_nature.second_q.mappers import JordanWignerMapper
import json


def chain(n=10, r=1.0,file=True):

    xyz = np.zeros((n, 3))
    atoms = ['H'] * n

    z = np.arange(-(n-1)/2, (n)/2) * r
    assert len(z) == n
    assert sum(z) == 0.0

    xyz[:, 2] = z

    description = "H" + str(n) + " 1D chain, " + str(r) + " Angstroms\n"
    
    if file:
        with open('h'+str(n)+'_'+str(r)+'_'+'chain'+'.json','w') as f:
            print(as_xyz(atoms,xyz,description),file=f)
    
    return atoms, xyz

def as_xyz(atoms, xyz, description='\n'):
    # format as .XYZ
    n = len(atoms)
    pretty_xyz = str(n) + '\n'
    pretty_xyz += description
    for i in range(n):
        pretty_xyz += "{0:s} {1:10.4f} {2:10.4f} {3:10.4f}\n".format(
                      atoms[i], xyz[i,0], xyz[i,1], xyz[i,2]) 

    return pretty_xyz

#old code: uses deprecated code that can keep if desired (but code below also

#def generate_qubit_hamiltonian(n,r):
#    # Generate chain atoms and positions
#    atoms, xyz = chain(n,r)
#
#    # Put atoms + posititions into a formatted list
#    geometry =  [[atom, position] for atom, position in zip(atoms, xyz)]
#
#    print(geometry)
#    # Define the molecular structure (in this case, it's a hydrogen molecule)
#    hydrogen_molecule = Molecule(
#            geometry=geometry, charge=0, multiplicity=1
#            )
#    # Prepare and run the initial Hartree-Fock calculation on molecule
#    molecule_driver = ElectronicStructureMoleculeDriver(
#            hydrogen_molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF
#            )
#    quantum_molecule = molecule_driver.run()
#    # Get nuclear repulsion + frozen core energy (constant)
#    nuclear_repulsion_energy = quantum_molecule.get_property("ElectronicEnergy").nuclear_repulsion_energy
#    # Define the electronic structure problem for Quantum Processing Unit (QPU)
#    electronic_structure_problem = ElectronicStructureProblem(molecule_driver)
#    # Generate the second-quantized operators (fermionic hamiltonian)
#    fermionic_hamiltonian = electronic_structure_problem.second_q_ops()["ElectronicEnergy"]
#    # Get some useful information of number qubits and electrons, etc.
#    grouped_property = electronic_structure_problem.grouped_property_transformed
#    particle_number = grouped_property.get_property("ParticleNumber")
#    num_particles = (particle_number.num_alpha, particle_number.num_beta)
#    num_spin_orbitals = particle_number.num_spin_orbitals
#    # Map fermionic Hamiltonian to qubit Hamiltonian. We use JW encoding.
#    qubit_converter = QubitConverter(JordanWignerMapper())
#    qubit_hamiltonian = qubit_converter.convert(
#            fermionic_hamiltonian, num_particles=num_particles
#            )
#
#    return qubit_hamiltonian

def generate_qubit_hamiltonian(n,r, file=False):
    # Generate chain atoms and positions
    atoms, xyz = chain(n,r)
    #print(atoms, xyz)

    #define the molecular structure 
    hydrogen_molecule = MoleculeInfo(atoms, xyz, charge=0, multiplicity=1)

    # Prepare and run the initial Hartree-Fock calculation on molecule
    molecule_driver = PySCFDriver.from_molecule(hydrogen_molecule, basis= "sto3g")
    #print("Spin:", molecule_driver.spin)
    #print("Atom:", molecule_driver.atom)
    quantum_molecule = molecule_driver.run()

    # acquire fermionic hamiltonian 
    fermionic_hamiltonian = quantum_molecule.hamiltonian.second_q_op()

    #create mapper from fermionic to spin basis
    mapper = JordanWignerMapper()

    #create qubit hamiltonian from fermionic one
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
    
    if file:

        op = qubit_hamiltonian.primitive.to_list()
        n_terms = len(op)
        coeffs = []
        paulis = []

        for i in range(n_terms):
            coeffs.append(complex(op[i][1]).real)
            paulis.append(op[i][0])

        pauli_dict = dict(zip(paulis,coeffs))
        pauli_json = json.dumps(pauli_dict)

        with open('h'+str(n)+'_'+str(r)+'_'+'chain'+'.json','w') as f:
            f.write(pauli_json)
    
    return qubit_hamiltonian


if __name__ == '__main__':
    n = 2
    R = 1.00
    print(generate_qubit_hamiltonian(n,R,True))
