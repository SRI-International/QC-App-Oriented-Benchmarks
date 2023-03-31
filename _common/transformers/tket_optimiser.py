from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import (  # type: ignore
    BasePass,
    auto_rebase_pass,
    RemoveRedundancies,
    SequencePass,
    SynthesiseTket,
    CXMappingPass,
    DecomposeBoxes,
    FullPeepholeOptimise,
    CliffordSimp,
    SimplifyInitial,
    KAKDecomposition,
)
from pytket.architecture import Architecture
from pytket.placement import NoiseAwarePlacement
from pytket.extensions.qiskit.qiskit_convert import (
    process_characterisation,
    get_avg_characterisation,
)
from pytket import OpType, Circuit
from qiskit.circuit.quantumcircuit import QuantumCircuit

def rebase_pass():
    return auto_rebase_pass({OpType.CX, OpType.X, OpType.SX, OpType.Rz})

def gen_pass_list(backend, quick=False, approximate=False):

    # Obtain device data for noise aware placement.
    characterisation = process_characterisation(backend)
    averaged_errors = get_avg_characterisation(characterisation)

    # Initialise pass list and perform thorough optimisation.
    pass_list = [DecomposeBoxes()]
    if not quick: pass_list.append(FullPeepholeOptimise())

    # Add noise aware placement and routing to pass list.
    coupling_map = backend.configuration().coupling_map
    if coupling_map:
        arch = Architecture(coupling_map)
        pass_list.append(
            CXMappingPass(
                arch,
                NoiseAwarePlacement(
                    arch,
                    averaged_errors["node_errors"],
                    averaged_errors["edge_errors"],
                    averaged_errors["readout_errors"],
                ),
                directed_cx=False,
            )
        )

    if (not quick) and approximate: pass_list.append(KAKDecomposition(cx_fidelity=0.9))
    if not quick: pass_list.extend([CliffordSimp(False), SynthesiseTket()])

    # Rebase to backend gate set and perform basic optimisation
    pass_list.append(rebase_pass())

    if not quick:
        pass_list.extend(
            [
                RemoveRedundancies(),
                SimplifyInitial(allow_classical=False, create_all_qubits=True),
            ]
        )

    return pass_list


def approximate_optimisation(circuit:QuantumCircuit, backend) -> list[QuantumCircuit]:
    """Perform thourough but generic optimisation using 
    TKET optimisation tool. Gates are approximately synthesised
    to further reduce gate count.

    :param circuit: Circuit to be optimised.
    :type circuit: QuantumCircuit
    :param backend: Backend which circuit should be optimised to.
    :type backend: BaseBackend
    :return: Optimised circuit.
    :rtype: list[QuantumCircuit]
    """
    print("  ... performing high TKET approximate optimisation.")

    pass_list = gen_pass_list(backend, quick=False, approximate=True)

    # Optimise circuit using constructed pass list
    tk_circuit = qiskit_to_tk(circuit)
    SequencePass(pass_list).apply(tk_circuit)
    circuit = tk_to_qiskit(tk_circuit)

    return [circuit]

def high_optimisation(circuit:QuantumCircuit, backend) -> list[QuantumCircuit]:
    """Perform thourough but generic optimisation using 
    TKET optimisation tool.

    :param circuit: Circuit to be optimised.
    :type circuit: QuantumCircuit
    :param backend: Backend which circuit should be optimised to.
    :type backend: BaseBackend
    :return: Optimised circuit.
    :rtype: list[QuantumCircuit]
    """
    print("  ... performing high TKET optimisation.")

    pass_list = gen_pass_list(backend, quick=False, approximate=False)

    # Optimise circuit using constructed pass list
    tk_circuit = qiskit_to_tk(circuit)
    SequencePass(pass_list).apply(tk_circuit)
    circuit = tk_to_qiskit(tk_circuit)

    return [circuit]

def quick_optimisation(circuit:QuantumCircuit, backend) -> list[QuantumCircuit]:
    """Perform basic compilation to build valid circuit for backend.

    :param circuit: Circuit to be compiled.
    :type circuit: QuantumCircuit
    :param backend: Backend to compile to.
    :type backend: BaseBackend
    :return: Compiled circuit.
    :rtype: list[QuantumCircuit]
    """
    print("  ... performing quick TKET optimisation.")

    pass_list = gen_pass_list(backend, quick=True, approximate=False)

    # Optimise circuit using constructed pass list
    tk_circuit = qiskit_to_tk(circuit)
    SequencePass(pass_list).apply(tk_circuit)
    circuit = tk_to_qiskit(tk_circuit)

    return [circuit]