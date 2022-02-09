from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import (  # type: ignore
    BasePass,
    RebaseCustom,
    RemoveRedundancies,
    SequencePass,
    SynthesiseTket,
    CXMappingPass,
    DecomposeBoxes,
    FullPeepholeOptimise,
    CliffordSimp,
    SimplifyInitial,
)
from pytket.routing import Architecture, NoiseAwarePlacement
from pytket.extensions.qiskit.qiskit_convert import (
    process_characterisation,
    get_avg_characterisation,
)
from pytket.extensions.qiskit.backends.ibm import _tk1_to_x_sx_rz
from pytket import OpType, Circuit

def rebase_pass():
    return RebaseCustom(
        {OpType.CX},
        Circuit(2).CX(0, 1),
        {OpType.X, OpType.SX, OpType.Rz},
        _tk1_to_x_sx_rz,
    )

def high_optimisation(circuit, backend):
    print("  ... performing high TKET optimisation.")

    characterisation = process_characterisation(backend)
    averaged_errors = get_avg_characterisation(characterisation)
    
    passlist = [DecomposeBoxes()]
    passlist.append(FullPeepholeOptimise())

    coupling_map = backend.configuration().coupling_map
    arch = Architecture(coupling_map)
    passlist.append(
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

    passlist.extend([CliffordSimp(False), SynthesiseTket()])
    passlist.append(rebase_pass())
    passlist.append(RemoveRedundancies())
    passlist.append(
        SimplifyInitial(allow_classical=False, create_all_qubits=True)
    )

    tk_circuit = qiskit_to_tk(circuit)

    SequencePass(passlist).apply(tk_circuit)
    circuit = tk_to_qiskit(tk_circuit)

    return [circuit]

def quick_optimisation(circuit, backend):
    print("  ... performing quick TKET optimisation.")

    characterisation = process_characterisation(backend)
    averaged_errors = get_avg_characterisation(characterisation)
    
    passlist = [DecomposeBoxes()]

    coupling_map = backend.configuration().coupling_map
    arch = Architecture(coupling_map)
    passlist.append(
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

    passlist.append(rebase_pass())

    tk_circuit = qiskit_to_tk(circuit)

    SequencePass(passlist).apply(tk_circuit)
    circuit = tk_to_qiskit(tk_circuit)

    return [circuit]