"""
Amplitude Estimation Benchmark Program via Phase Estimation Utils - Braket
"""
from braket.circuits import Circuit

# Function that generates a circuit object corresponding to the adjoint of a given circuit
# Adapted from AWS Braket QAA Example: 
# https://github.com/aws/amazon-braket-examples/blob/main/examples/advanced_circuits_algorithms/QAA/utils_circuit.py
def adjoint(circuit):
    # Define the adjoint circuit
    adjoint_circ = Circuit()

    # Loop through original circuit gates
    for instruction in circuit.instructions:
        # Save the operator name and target
        op_name = instruction.operator.name
        target = instruction.target
        angle = None

        # If the operator has an attribute called 'angle', save that too
        if hasattr(instruction.operator, "angle"):
            angle = instruction.operator.angle
        
        # To make use of native gates, we'll define the adjoint for each
        if op_name == "H":
            adjoint_gate = Circuit().h(target)
        elif op_name == "I":
            adjoint_gate = Circuit().i(target)
        elif op_name == "X":
            adjoint_gate = Circuit().x(target)
        elif op_name == "Y":
            adjoint_gate = Circuit().y(target)
        elif op_name == "Z":
            adjoint_gate = Circuit().z(target)
        elif op_name == "S":
            adjoint_gate = Circuit().si(target)
        elif op_name == "Si":
            adjoint_gate = Circuit().s(target)
        elif op_name == "T":
            adjoint_gate = Circuit().ti(target)
        elif op_name == "Ti":
            adjoint_gate = Circuit().t(target)
        elif op_name == "V":
            adjoint_gate = Circuit().vi(target)
        elif op_name == "Vi":
            adjoint_gate = Circuit().v(target)
        elif op_name == "Rx":
            adjoint_gate = Circuit().rx(target, -angle)
        elif op_name == "Ry":
            adjoint_gate = Circuit().ry(target, -angle)
        elif op_name == "Rz":
            adjoint_gate = Circuit().rz(target, -angle)
        elif op_name == "PhaseShift":
            adjoint_gate = Circuit().phaseshift(target, -angle)
        elif op_name == "CNot":
            adjoint_gate = Circuit().cnot(*target)
        elif op_name == "Swap":
            adjoint_gate = Circuit().swap(*target)
        elif op_name == "ISwap":
            adjoint_gate = Circuit().pswap(*target, -np.pi / 2)
        elif op_name == "PSwap":
            adjoint_gate = Circuit().pswap(*target, -angle)
        elif op_name == "XY":
            adjoint_gate = Circuit().xy(*target, -angle)
        elif op_name == "CPhaseShift":
            adjoint_gate = Circuit().cphaseshift(*target, -angle)
        elif op_name == "CPhaseShift00":
            adjoint_gate = Circuit().cphaseshift00(*target, -angle)
        elif op_name == "CPhaseShift01":
            adjoint_gate = Circuit().cphaseshift01(*target, -angle)
        elif op_name == "CPhaseShift10":
            adjoint_gate = Circuit().cphaseshift10(*target, -angle)
        elif op_name == "CY":
            adjoint_gate = Circuit().cy(*target)
        elif op_name == "CZ":
            adjoint_gate = Circuit().cz(*target)
        elif op_name == "XX":
            adjoint_gate = Circuit().xx(*target, -angle)
        elif op_name == "YY":
            adjoint_gate = Circuit().yy(*target, -angle)
        elif op_name == "ZZ":
            adjoint_gate = Circuit().zz(*target, -angle)
        elif op_name == "CCNot":
            adjoint_gate = Circuit().ccnot(*target)
        elif op_name == "CSwap":
            adjoint_gate = Circuit().cswap(*target)
        
        # If the gate is a custom unitary, we'll create a new custom unitary
        else:
            # Extract the transpose of the unitary matrix for the unitary gate
            adjoint_matrix = instruction.operator.to_matrix().T.conj()

            # Define a gate for which the unitary matrix is the adjoin found above.
            # Add an "H" to the display name.
            adjoint_gate = Circuit().unitary(
                matrix=adjoint_matrix,
                targets=instruction.target,
                display_name="".join(instruction.operator.ascii_symbols) + "H",
            )
        # Add the new gate to the adjoint circuit. Note the order of operators here:
        # (AB)^H = B^H A^H, where H is adjoint, thus we prepend new gates, rather than append.
        adjoint_circ = adjoint_gate.add(adjoint_circ)
    
    return adjoint_circ

# Function to apply MCX gate to Ctrl_Q, implemented using ancilla qubits
# Adapted from AWS Braket QAA Example: 
# https://github.com/aws/amazon-braket-examples/blob/main/examples/advanced_circuits_algorithms/QAA/utils_qaa.py
def applyMCX(circuit):
    qubits = circuit.qubits

    # Dynamically add ancilla qubits, starting on the next unused qubit in the circuit
    ancilla_start = max(qubits) + 1

    # Apply CCNOT on first two qubits
    circuit.ccnot(qubits[0], qubits[1], ancilla_start)

    # Now add a CCNOT from each of the next register qubits, comparing with the ancilla we just added.
    # Target on a new ancilla. If len(qubits) is 2, this does not execute.
    for ii, qubit in enumerate(qubits[2:]):
        circuit.ccnot(qubit, ancilla_start + ii, ancilla_start + ii + 1)

    # TODO ... Finish MCX gate implementation. Do we need to use ancilla qubits?
    

