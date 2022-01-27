"""
Amplitude Estimation Benchmark Program via Phase Estimation Utils - Braket
"""
import numpy as np
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

# Function to apply MCX gate without ancilla qubits
# Adapted from Barenco et al: https://arxiv.org/pdf/quant-ph/9503016.pdf
def MCX(num_state_qubits):
    qc = Circuit()
    m = num_state_qubits

    # First check to see if m = 1 or m = 2, in which case this simplifies to 
    # CNOT or CCNOT
    if m == 2:
        qc.ccnot(control1=0, control2=1, target=2)
        return qc
    elif m == 1:
        qc.cnot(control=0, target=1)
        return qc

    ## ELSE: use gray code method from Barenco et al to generate appropriate gates
    ## Probably some room for improvement... it produces an equivalent circuit, but 
    ## with extra CNOT gates.

    # Generate list of qubit indices for the circuit
    qubits = list(range(num_state_qubits + 1))

    # Using the notation from Barenco et al, U = X, thus we want to select
    # V such that V^(m + 1) = U = X => V = (2^(m - 1))-root(X)
    def n_root_X_matrix(n):
        factor = np.exp(1j * (np.pi / n))
        print(factor)

        matrix = 0.5 * np.array([
            [1 + factor, 1 - factor],
            [1 - factor, 1 + factor]
        ])

        return np.matrix(matrix)
    
    # V and V_dagger unitary matrix
    V_unitary_matrix = n_root_X_matrix(2 ** (m - 1))
    V_dagger_unitary_matrix = V_unitary_matrix.H

    # Generate gray code sequence for number of control qubits
    def generateGrayCode(n):
        if (n <= 0): return

        gray = []

        gray.append("0")
        gray.append("1")

        i = 2
        j = 0
        while True:
            if i >= 1 << n:
                break
            
            for j in range(i - 1, -1, -1):
                gray.append(gray[j])
            
            for j in range(i):
                gray[j] = "0" + gray[j]

            for j in range(i, 2*i):
                gray[j] = "1" + gray[j]
            
            i = i << 1
        
        return gray

    # Generate gray code and remove first element (000)
    grayCode = generateGrayCode(m)[1:]

    # Use gray code to apply appropriate gates in order
    # Note: even/odd parity of bitstring determines whether or not
    # to apply V or V_dagger.
    for sequence in grayCode:
        # Gates to apply for a given sequence
        applyToQubits = []
        odd_parity = False
        for i in range(m):
            # Reverse sequence to match paper, does it matter?
            if sequence[::-1][i] == "1":
                odd_parity = ~odd_parity
                applyToQubits.append(qubits[i])
            elif sequence[i] == "0":
                pass
        
        # If it's only one qubit, apply controlled V
        if len(applyToQubits) == 1:
            qc.add_circuit(controlled_unitary(
                control=applyToQubits[0], 
                targets=[num_state_qubits], 
                unitary=V_unitary_matrix, 
                display_name="V"
            ))
        elif len(applyToQubits) > 1:
            # Else, we need to iterate through and apply CNOT gates
            for qubit_index in range(len(applyToQubits) - 1):
                qc.cnot(control=applyToQubits[qubit_index], target=applyToQubits[len(applyToQubits) - 1])

            # Apply controlled V/V_dagger on last qubit line
            if odd_parity:
                qc.add_circuit(controlled_unitary(
                    control=applyToQubits[-1], 
                    targets=[num_state_qubits],
                    unitary=V_unitary_matrix, 
                    display_name="V"
                ))    
            else:
                qc.add_circuit(controlled_unitary(
                    control=applyToQubits[-1], 
                    targets=[num_state_qubits], 
                    unitary=V_dagger_unitary_matrix,
                    display_name="V_dagger"
                ))
            
            # Apply cnot gates in reverse order to undo operations
            for qubit_index in reversed(range(len(applyToQubits) - 1)):
                qc.cnot(control=applyToQubits[qubit_index], target=applyToQubits[len(applyToQubits) - 1])

    return qc


# Function to create a controlled arbitrary unitary gate
# Adapted from AWS Braket QPE Example:
# https://github.com/aws/amazon-braket-examples/blob/main/examples/advanced_circuits_algorithms/QPE/utils_qpe.py
def controlled_unitary(control, targets, unitary, display_name):
    # Define projectors onto the computational basis
    p0 = np.array([[1.0, 0.0], [0.0, 0.0]])

    p1 = np.array([[0.0, 0.0], [0.0, 1.0]])

    # Instantiate circuit object
    circ = Circuit()

    # Construct numpy matrix
    id_matrix = np.eye(len(unitary))
    controlled_matrix = np.kron(p0, id_matrix) + np.kron(p1, unitary)

    # Set all target qubits
    target_qubits = [control] + targets

    # Add controlled unitary
    circ.unitary(matrix=controlled_matrix, targets=target_qubits, display_name=display_name)

    return circ