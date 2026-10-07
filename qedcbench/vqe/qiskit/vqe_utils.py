from qiskit_nature.drivers import PySCFDriver, UnitsType
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.transformers import ActiveSpaceTransformer
from qiskit_nature.operators.second_quantization import FermionicOp


# Function that converts a list of single and double excitation operators to Pauli operators
def convertExcitationToPauli(singles, doubles, norb, t1_amp, t2_amp):

    # initialize Pauli list
    pauli_list = []

    # qubit converter
    qubit_converter = QubitConverter(JordanWignerMapper())
    
    # loop over singles
    for index, single in enumerate(singles):
        t1 = t1_amp[index] * 1j * (FermionicOp(f"-_{single[0]} +_{single[1]}", norb) \
           - FermionicOp(f"-_{single[0]} +_{single[1]}", norb).adjoint())
        #print(t1)
        qubit_op = qubit_converter.convert(t1)
        for p in qubit_op:
            pauli_list.append(p)
        #pauli_list.append(qubit_op[0])

    # loop over doubles
    for index, double in enumerate(doubles):
        t2 = t2_amp[index] * 1j * (FermionicOp(f"-_{double[0]} +_{double[1]} -_{double[2]} +_{double[3]}", norb) \
           - FermionicOp(f"-_{double[0]} +_{double[1]} -_{double[2]} +_{double[3]}", norb).adjoint())
        qubit_op = qubit_converter.convert(t2)
        #print(qubit_op)
        for p in qubit_op:
            pauli_list.append(p)
        #pauli_list.append(qubit_op[0])

    # return Pauli list
    return pauli_list


# Get the inactive energy and the Hamiltonian operator in an active space
def GetHamiltonians(mol, n_orbs, na, nb):
    
    # construct the driver
    driver = PySCFDriver(molecule=mol, unit=UnitsType.ANGSTROM, basis='sto6g')

    # the active space transformer (use a (2, 2) active space)
    transformer = ActiveSpaceTransformer(num_electrons=(na+nb), num_molecular_orbitals=int(n_orbs/2))

    # the electronic structure problem
    problem = ElectronicStructureProblem(driver, [transformer])

    # get quantum molecule
    q_molecule = driver.run()

    # reduce the molecule to active space
    q_molecule_reduced = transformer.transform(q_molecule)

    # compute inactive energy
    core_energy = q_molecule_reduced.energy_shift["ActiveSpaceTransformer"]

    # add nuclear repulsion energy
    core_energy += q_molecule_reduced.nuclear_repulsion_energy

    # generate the second-quantized operators
    second_q_ops = problem.second_q_ops()

    # construct a qubit converter
    qubit_converter = QubitConverter(JordanWignerMapper())

    # qubit Operations 
    qubit_op = qubit_converter.convert(second_q_ops[0])

    # return the qubit operations
    return qubit_op, core_energy

