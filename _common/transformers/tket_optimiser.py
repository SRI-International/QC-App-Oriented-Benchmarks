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

def tket_transformer_generator(quick=False, cx_fidelity=1.0):
    """Generator for transformer using TKET passes

    :param quick: Perform quick optimisation, defaults to False
    :type quick: bool, optional
    :param cx_fidelity: Estimated CX gate fidelity, defaults to 1.0
    :type cx_fidelity: float, optional
    """

    def transformation_method(circuit:QuantumCircuit, backend) -> list[QuantumCircuit]:
        """Transformer using TKET optimisation passes.

        :param circuit: Circuit to be optimised
        :type circuit: QuantumCircuit
        :param backend: Backed on which circuit is run.
        :return: List of transformed circuits
        :rtype: list[QuantumCircuit]
        """

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

        if not quick:
            pass_list.append(KAKDecomposition(cx_fidelity=0.9))
            pass_list.extend([CliffordSimp(False), SynthesiseTket()])

        # Rebase to backend gate set and perform basic optimisation
        pass_list.append(rebase_pass())

        if not quick:
            pass_list.extend(
                [
                    RemoveRedundancies(),
                    SimplifyInitial(allow_classical=False, create_all_qubits=True),
                ]
            )

        tk_circuit = qiskit_to_tk(circuit)
        SequencePass(pass_list).apply(tk_circuit)
        circuit = tk_to_qiskit(tk_circuit)

        return [circuit]

    return transformation_method