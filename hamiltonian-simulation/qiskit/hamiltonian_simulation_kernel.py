'''
Hamiltonian Simulation Benchmark Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

'''
There are multiple Hamiltonians and three methods defined for this kernel.
The Hamiltonian name is specified in the "hamiltonian" argument.
The "method" argument indicates the type of fidelity comparison that will be done. 
In this case, method 3 is used to create a mirror circuit for scalability.
'''

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
# Saved circuits and subcircuits for display
QC_ = None
XX_ = None
YY_ = None
ZZ_ = None
XXYYZZ_ = None

# Mirror Gates of the previous four gates
XX_mirror_ = None
YY_mirror_ = None
ZZ_mirror_ = None
XXYYZZ_mirror_ = None

# For validating the implementation of XXYYZZ operation (saved for possible use in drawing)
_use_XX_YY_ZZ_gates = False


############### Circuit Definition
def initial_state(n_spins: int, initial_state: str = "checker") -> QuantumCircuit:
    """
    Initialize the quantum state.
    
    Args:
        n_spins (int): Number of spins (qubits).
        initial_state (str): The chosen initial state. By default applies the checkerboard state, but can also be set to "ghz", the GHZ state.

    Returns:
        QuantumCircuit: The initialized quantum circuit.
    """
    qc = QuantumCircuit(n_spins, name = "InitialState")

    if initial_state.strip().lower() == "checkerboard" or initial_state.strip().lower() == "neele":
        # Checkerboard state, or "Neele" state
        for k in range(0, n_spins, 2):
            qc.x([k])
    elif initial_state.strip().lower() == "ghz":
        # GHZ state: 1/sqrt(2) (|00...> + |11...>)
        qc.h(0)
        for k in range(1, n_spins):
            qc.cx(k-1, k)

    return qc

############## Heisenberg Circuit
def Heisenberg(n_spins: int, K: int, t: float, tau: float, w: float, h_x: list[float], h_z: list[float],
            use_XX_YY_ZZ_gates: bool = False) -> QuantumCircuit:
    
    qr = QuantumRegister(n_spins)
    qc = QuantumCircuit(qr, name = "Heisenberg")
    # Loop over each Trotter step, adding gates to the circuit defining the Hamiltonian
    for k in range(K):
        # Pauli spin vector product
        [qc.rx(2 * tau * w * h_x[i], qr[i]) for i in range(n_spins)]
        [qc.rz(2 * tau * w * h_z[i], qr[i]) for i in range(n_spins)]
        qc.barrier()
        
        # Basic implementation of exp(i * t * (XX + YY + ZZ))
        if use_XX_YY_ZZ_gates:
            # XX operator on each pair of qubits in linear chain
            for j in range(2):
                for i in range(j%2, n_spins - 1, 2):
                    qc.append(xx_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

            # YY operator on each pair of qubits in linear chain
            for j in range(2):
                for i in range(j%2, n_spins - 1, 2):
                    qc.append(yy_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

            # ZZ operation on each pair of qubits in linear chain
            for j in range(2):
                for i in range(j%2, n_spins - 1, 2):
                    qc.append(zz_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
                    
        else:
            # Optimized XX + YY + ZZ operator on each pair of qubits in linear chain
            for j in reversed(range(2)):
                for i in reversed(range(j % 2, n_spins - 1, 2)):
                    qc.append(xxyyzz_opt_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
        qc.barrier()

    return qc

########### TFIM circuit
def tfim(n_spins: int, K: int, tau: float, use_XX_YY_ZZ_gates: bool)-> QuantumCircuit:
    h = 0.2  # Strength of transverse field
    qr = QuantumRegister(n_spins)
    qc = QuantumCircuit(qr, name = "TFIM")
    for k in range(K):
        for i in range(n_spins):
            qc.rx(2 * tau * h, qr[i])
        qc.barrier()

        for j in range(2):
            for i in range(j % 2, n_spins - 1, 2):
                qc.append(zz_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
        qc.barrier()
    return qc

############## Apply random Pauli gates to all the qubits.
def create_random_paulis(n_spins)-> QuantumCircuit:
    """Create a quantum oracle that applies random Pauli gates to n qubits."""
    qr = QuantumRegister(n_spins)
    qc = QuantumCircuit(qr, name = "RandomPaulis")
    
    for i in range(n_spins):
        gate = np.random.choice(['x','y','z'])
        if gate == 'x':
            qc.x(i)
        elif gate == 'y':
            qc.y(i)
        elif gate == 'z':
            qc.z(i)
        qc.barrier()   
    return qc

############# Resultant Pauli after applying quasi inverse Hamiltonain and random Pauli to Hamiltonian.
def ResultantPauli(n_spins)-> QuantumCircuit:
    """Create a quantum oracle that is the result of applying quasi inverse Hamiltonain and random Pauli to Hamiltonian."""
    qr = QuantumRegister(n_spins)
    qc = QuantumCircuit(qr)
    for n in range(n_spins):
        qc.x(n)      # You can apply any Pauli, but you must also change the state you are comparing with.
        qc.barrier()
    return qc

############ Quasi Inverse Heisenberg    
def QuasiHamiltonian(hamiltonian_circuit, random_pauli_oracle, res_pauli, n_spins)-> QuantumCircuit:
                qr = QuantumRegister(n_spins)
                qc = QuantumCircuit(qr, name = "QuasiHamiltonian")
                hamiltonian_circuit_inverse = hamiltonian_circuit.inverse()
                random_pauli_oracle_inverse = random_pauli_oracle.inverse()
                
                qc.append(random_pauli_oracle_inverse,qr)
                qc.append(hamiltonian_circuit_inverse ,qr)
                
                qc.append(res_pauli,qr)

                return qc

def HamiltonianSimulation(n_spins: int, K: int, t: float,
            hamiltonian: str, w: float, hx: list[float], hz: list[float],
            use_XX_YY_ZZ_gates: bool = False,
            method: int = 1) -> QuantumCircuit:
    """
    Construct a Qiskit circuit for Hamiltonian simulation.

    Args:
        n_spins (int): Number of spins (qubits).
        K (int): The Trotterization order.
        t (float): Duration of simulation.
        hamiltonian (str): Which hamiltonian to run. "heisenberg" by default but can also choose "TFIM". 
        w (float): Strength of two-qubit interactions for heisenberg hamiltonian. 
        hx (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 
        hz (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 

    Returns:
        QuantumCircuit: The constructed Qiskit circuit.
    """
    global _use_XX_YY_ZZ_gates
    _use_XX_YY_ZZ_gates = use_XX_YY_ZZ_gates
    
    num_qubits = n_spins
    secret_int = f"{K}-{t}"

    # Allocate qubits
    qr = QuantumRegister(n_spins)
    cr = ClassicalRegister(n_spins)
    qc = QuantumCircuit(qr, cr, name=f"hamsim-{num_qubits}-{secret_int}")
    tau = t / K

    h_x = hx[:n_spins]
    h_z = hz[:n_spins]

    hamiltonian = hamiltonian.strip().lower()


    if hamiltonian == "heisenberg": 
        init_state = "checkerboard"
        # apply initial state
        qc.append(initial_state(n_spins, init_state), qr)
        qc.barrier()
        heisenberg_circuit = Heisenberg(n_spins, K, t, tau, w, h_x, h_z, use_XX_YY_ZZ_gates)
        qc.append(heisenberg_circuit, qr)
        qc.barrier()
 
        if (method == 3):
            random_pauli_oracle = create_random_paulis(n_spins)
            qc.append(random_pauli_oracle, qr)
            qc.barrier()
            res_pauli = ResultantPauli(n_spins)
            Quasi_heisenberg = QuasiHamiltonian(heisenberg_circuit, random_pauli_oracle, res_pauli, n_spins)
            qc.append(Quasi_heisenberg, qr)


    elif hamiltonian == "tfim":

        init_state = "ghz"
        #apply initial state
        qc.append(initial_state(n_spins, init_state), qr)
        qc.barrier()
        tfim_circuit = tfim(n_spins, K, tau, use_XX_YY_ZZ_gates)
        qc.append(tfim_circuit, qr)
        qc.barrier()
        
        if (method == 3):
            random_pauli_oracle = create_random_paulis(n_spins)
            qc.append(random_pauli_oracle, qr)
            qc.barrier()
            res_pauli = ResultantPauli(n_spins)
            Quasi_tfim = QuasiHamiltonian(tfim_circuit, random_pauli_oracle, res_pauli, n_spins)
            qc.append(Quasi_tfim, qr)

    else:
        raise ValueError("Invalid Hamiltonian specification.")

    # Measure all qubits
    for i_qubit in range(n_spins):
        qc.measure(qr[i_qubit], cr[i_qubit])

    # Save smaller circuit example for display
    global QC_
    if QC_ is None or n_spins <= 6:
        if n_spins < 9:
            QC_ = qc

    # Collapse the sub-circuits used in this benchmark (for Qiskit)
    qc2 = qc.decompose()
            
    return qc2
    

############### XX, YY, ZZ Gate Implementations

def xx_gate(tau: float) -> QuantumCircuit:
    """
    Simple XX gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The XX gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="xx_gate")
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    
    global XX_
    XX_ = qc
    
    return qc

def yy_gate(tau: float) -> QuantumCircuit:
    """
    Simple YY gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The YY gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="yy_gate")
    qc.s(qr[0])
    qc.s(qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.sdg(qr[0])
    qc.sdg(qr[1])

    global YY_
    YY_ = qc

    return qc

def zz_gate(tau: float) -> QuantumCircuit:
    """
    Simple ZZ gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The ZZ gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="zz_gate")
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])

    global ZZ_
    ZZ_ = qc

    return qc

def xxyyzz_opt_gate(tau: float) -> QuantumCircuit:
    """
    Optimal combined XXYYZZ gate (with double coupling) on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The optimal combined XXYYZZ gate circuit.
    """
    alpha = tau
    beta = tau
    gamma = tau
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="xxyyzz_opt")
    qc.rz(3.1416 / 2, qr[1])
    qc.cx(qr[1], qr[0])
    qc.rz(3.1416 * gamma - 3.1416 / 2, qr[0])
    qc.ry(3.1416 / 2 - 3.1416 * alpha, qr[1])
    qc.cx(qr[0], qr[1])
    qc.ry(3.1416 * beta - 3.1416 / 2, qr[1])
    qc.cx(qr[1], qr[0])
    qc.rz(-3.1416 / 2, qr[0])

    global XXYYZZ_
    XXYYZZ_ = qc

    return qc

    


############### Mirrors of XX, YY, ZZ Gate Implementations   
def xx_gate_mirror(tau: float) -> QuantumCircuit:
    """
    Simple XX mirror gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The XX_mirror_ gate circuit.
    """
    qr = QuantumRegister(2, 'q')
    qc = QuantumCircuit(qr, name="xx_gate_mirror")
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(-3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])

    global XX_mirror_
    XX_mirror_ = qc

    return qc

def yy_gate_mirror(tau: float) -> QuantumCircuit:
    """
    Simple YY mirror gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The YY_mirror_ gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="yy_gate_mirror")
    qc.s(qr[0])
    qc.s(qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(-3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.sdg(qr[0])
    qc.sdg(qr[1])

    global YY_mirror_
    YY_mirror_ = qc

    return qc   

def zz_gate_mirror(tau: float) -> QuantumCircuit:
    """
    Simple ZZ mirror gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The ZZ_mirror_ gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="zz_gate_mirror")
    qc.cx(qr[0], qr[1])
    qc.rz(-3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])

    global ZZ_mirror_
    ZZ_mirror_ = qc

    return qc

def xxyyzz_opt_gate_mirror(tau: float) -> QuantumCircuit:
    """
    Optimal combined XXYYZZ mirror gate (with double coupling) on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The optimal combined XXYYZZ_mirror_ gate circuit.
    """
    alpha = tau
    beta = tau
    gamma = tau
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="xxyyzz_opt_mirror")
    qc.rz(3.1416 / 2, qr[0])
    qc.cx(qr[1], qr[0])
    qc.ry(-3.1416 * beta + 3.1416 / 2, qr[1])
    qc.cx(qr[0], qr[1])
    qc.ry(-3.1416 / 2 + 3.1416 * alpha, qr[1])
    qc.rz(-3.1416 * gamma + 3.1416 / 2, qr[0])
    qc.cx(qr[1], qr[0])
    qc.rz(-3.1416 / 2, qr[1])

    global XXYYZZ_mirror_
    XXYYZZ_mirror_ = qc

    return qc


############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw(hamiltonian: str = "heisenberg", use_XX_YY_ZZ_gates: bool = False, method: int = 1):
                          
    # Print a sample circuit
    print("Sample Circuit:")
    print(QC_ if QC_ is not None else "  ... too large!")

    if hamiltonian == "heisenberg": 
        if use_XX_YY_ZZ_gates:
            print("\nXX, YY, ZZ = ")
            print(XX_)
            print(YY_)
            print(ZZ_)
            if method == 3:
                print("\nXX, YY, ZZ mirror = ")
                print(XX_mirror_)
                print(YY_mirror_)
                print(ZZ_mirror_)
        else:
            print("\nXXYYZZ_opt = ")
            print(XXYYZZ_)  
            if method == 3:
                print("\nXXYYZZ_opt_mirror = ")
                print(XXYYZZ_mirror_)
    
    if hamiltonian == "tfim": 
        print("\nZZ = ")
        print(ZZ_)
        if method == 3:
            print("\nZZ mirror = ")
            print(ZZ_mirror_)
    