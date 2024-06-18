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

# DEVNOTE: the global variables below will be converted to class instance variables later

# Saved circuits and subcircuits for display
QC_ = None
QCI_ = None
QCR_ = None
QCRP_ = None
QCRS_ = None

QC2_ = None
XX_ = None
YY_ = None
ZZ_ = None
XXYYZZ_ = None

# Mirror Gates of the previous four gates
QC2D_ = None
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

########### TFIM hamiltonian circuit
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
    qc = QuantumCircuit(qr, name = "ResultantPaulis")
    for n in range(n_spins):
        qc.x(n)      # You can apply any Pauli, but you must also change the state you are comparing with.
        
    qc.barrier()
    
    return qc

############ Quasi Inverse Heisenberg 
########### ~H P H = R ==> ~H = R H' P'  ; ~H is QuasiHamiltonian, P is Random Pauli, H is Hamiltonian, R is resultant circuit that appends on the initial state
def QuasiHamiltonian(hamiltonian_circuit, random_pauli_oracle, res_pauli, n_spins)-> QuantumCircuit:
    qr = QuantumRegister(n_spins)
    qc = QuantumCircuit(qr, name = "QuasiHamiltonian")
    hamiltonian_circuit_inverse = hamiltonian_circuit.inverse()
    random_pauli_oracle_inverse = random_pauli_oracle.inverse()
    
    qc.append(random_pauli_oracle_inverse,qr)             
    qc.append(hamiltonian_circuit_inverse ,qr)
    
    qc.append(res_pauli,qr) 

    return qc

#Inverse of Heisenberg model. mirror gates are applied.
def HeisenbergInverse(n_spins: int, K: int, t: float, tau: float, w: float, h_x: list[float], h_z: list[float],
            use_XX_YY_ZZ_gates: bool = False) -> QuantumCircuit:
    
    qr = QuantumRegister(n_spins)
    qc = QuantumCircuit(qr, name = "HeisenbergInverse")
    # Add mirror gates for negative time simulation
    for k in range(K): 
        # Basic implementation of exp(-i * t * (XX + YY + ZZ)):
        if use_XX_YY_ZZ_gates:
            # regular inverse of XX + YY + ZZ operators on each pair of quibts in linear chain
            # XX operator on each pair of qubits in linear chain
            for j in range(2):
                for i in range(j%2, n_spins - 1, 2):
                    qc.append(zz_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

            # YY operator on each pair of qubits in linear chain
            for j in range(2):
                for i in range(j%2, n_spins - 1, 2):
                    qc.append(yy_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

            # ZZ operation on each pair of qubits in linear chain
            for j in range(2):
                for i in range(j%2, n_spins - 1, 2):
                    qc.append(xx_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

        else:
            # optimized Inverse of XX + YY + ZZ operator on each pair of qubits in linear chain
            for j in range(2):
                for i in range(j % 2, n_spins - 1, 2):
                    qc.append(xxyyzz_opt_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
        qc.barrier()

        # the Pauli spin vector product
        [qc.rz(-2 * tau * w * h_z[i], qr[i]) for i in range(n_spins)]
        [qc.rx(-2 * tau * w * h_x[i], qr[i]) for i in range(n_spins)]
        qc.barrier()
    return qc

#########Inverse of tfim hamiltonian
def tfimInverse(n_spins: int, K: int, tau: float, use_XX_YY_ZZ_gates: bool)-> QuantumCircuit:
    h = 0.2
    qr = QuantumRegister(n_spins)
    qc = QuantumCircuit(qr, name = "tfimInverse")
    for k in range(K):
        for j in range(2):
            for i in range(j % 2, n_spins - 1, 2):
                qc.append(zz_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
        qc.barrier()
        for i in range(n_spins):
            qc.rx(-2 * tau * h, qr[i])
        qc.barrier()
    return qc


def HamiltonianSimulation(n_spins: int, K: int, t: float,
            hamiltonian: str, w: float, hx: list[float], hz: list[float],
            use_XX_YY_ZZ_gates: bool = False,
            method: int = 1, random_pauli_flag: bool = True) -> QuantumCircuit:
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
    global QC_, QCI_, QCRP_, QCRS_, QC2_, QC2D_
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
    
        # append the initial state circuit to the quantum circuit
        init_state = "checkerboard"
        QCI_ = initial_state(n_spins, init_state)
        qc.append(QCI_, qr)
        qc.barrier()
        
        # append the Hamiltonian-specific circuit
        QC2_ = heisenberg_circuit = Heisenberg(n_spins, K, t, tau, w, h_x, h_z, use_XX_YY_ZZ_gates) 
        qc.append(heisenberg_circuit, qr)
        qc.barrier()
 
        if (method == 3):
            if random_pauli_flag:
            
                QCRP_ = random_pauli_oracle = create_random_paulis(n_spins)
                qc.append(random_pauli_oracle, qr)    #append the random pauli on the circuit
                qc.barrier()
                
                QCRS_ = res_pauli = ResultantPauli(n_spins) # create a resultant pauli that we want to apply to initial state.
                
                QC2D_ = Quasi_heisenberg = QuasiHamiltonian(heisenberg_circuit, random_pauli_oracle, res_pauli, n_spins) # create a QuasiHamiltonian. 
                qc.append(Quasi_heisenberg, qr)    #append the Quasi Hamiltonian on the circuit
            
            else:
                #if random_pauli_flag is False, just use traditional mirror circuit, i.e. Apply Inverse of Hamiltonian to the Hamiltonian to give Inverse.
                QC2D_ = heisenberg_inverse_circuit = HeisenbergInverse(n_spins, K, t, tau, w, h_x, h_z, use_XX_YY_ZZ_gates)
                qc.append(heisenberg_inverse_circuit, qr)
                qc.barrier()


    elif hamiltonian == "tfim":

        # append the initial state circuit to the quantum circuit
        init_state = "ghz"
        QCI_ = initial_state(n_spins, init_state)  
        qc.append(QCI_, qr)
        qc.barrier()
        
        # append the Hamiltonian-specific circuit
        QC2_ = tfim_circuit = tfim(n_spins, K, tau, use_XX_YY_ZZ_gates)
        qc.append(tfim_circuit, qr)
        qc.barrier()
        
        if (method == 3):
        
            if random_pauli_flag == True:
                QCRP_ = random_pauli_oracle = create_random_paulis(n_spins)
                qc.append(random_pauli_oracle, qr)  #append the random pauli on the circuit
                qc.barrier()
                
                QCRS_ = res_pauli = ResultantPauli(n_spins) # create a resultant pauli that we want to apply to initial state.
                
                QC2D_ = Quasi_tfim = QuasiHamiltonian(tfim_circuit, random_pauli_oracle, res_pauli, n_spins) # create a QuasiHamiltonian.
                qc.append(Quasi_tfim, qr) #append the Quasi Hamiltonian on the circuit
            else:
                #if random_pauli_flag is False, just use traditional mirror circuit, i.e. Apply Inverse of Hamiltonian to the Hamiltonian to give Inverse.
                QC2D_ = tfim_inverse_circuit = tfimInverse(n_spins, K, tau, use_XX_YY_ZZ_gates)
                qc.append(tfim_inverse_circuit, qr)
                qc.barrier()

    else:
        raise ValueError("Invalid Hamiltonian specification.")

    # Measure all qubits
    for i_qubit in range(n_spins):
        qc.measure(qr[i_qubit], cr[i_qubit])

    # Save smaller circuit example for display
    if QC_ is None or n_spins <= 6:
        if n_spins < 9:
            QC_ = qc

    # Collapse the sub-circuits used in this benchmark (for Qiskit)
    qc2 = qc.decompose().decompose()
            
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
    qc = QuantumCircuit(qr, name="XX")
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
    qc = QuantumCircuit(qr, name="YY")
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
    qc = QuantumCircuit(qr, name="ZZ")
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
    qc = QuantumCircuit(qr, name="XXYYZZ")
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
    qc = QuantumCircuit(qr, name="XX\u2020")
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
    qc = QuantumCircuit(qr, name="YY\u2020")
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
    qc = QuantumCircuit(qr, name="ZZ\u2020")
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
    qc = QuantumCircuit(qr, name="XXYYZZ\u2020")
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
def kernel_draw(hamiltonian: str = "heisenberg", use_XX_YY_ZZ_gates: bool = False, method: int = 1,random_pauli_flag: bool = True):
                          
    # Print a sample circuit
    print("Sample Circuit:")
    print(QC_ if QC_ is not None else "  ... too large!")
    
    # we don't restrict save of large sub-circuits, so skip printing if num_qubits too large
    if QCI_ is not None and QCI_.num_qubits > 6:
        print("... subcircuits too large to print") 
        return
        
    print("  Initial State:")
    if QCI_ is not None: print(QCI_)
    
    print(f"  Hamiltonian ({QC2_.name if QC2_ is not None else '?'}):")
    if QC2_ is not None: print(QC2_)
      
    if QC2D_ is not None:
        print("  Quasi-Hamiltonian:")
        print(QC2D_)

    if hamiltonian == "heisenberg": 
        if use_XX_YY_ZZ_gates:
            print("\nXX, YY, ZZ = ")
            print(XX_)
            print(YY_)
            print(ZZ_)
            if method == 3:
                print("\nXX, YY, ZZ \u2020 = ")
                print(XX_mirror_)
                print(YY_mirror_)
                print(ZZ_mirror_)
        else:
            print("\nXXYYZZ = ")
            print(XXYYZZ_)  
            if method == 3:
                print("\nXXYYZZ\u2020 = ")
                print(XXYYZZ_mirror_)
    
    if hamiltonian == "tfim": 
        print("\nZZ = ")
        print(ZZ_)
        if method == 3:
            print("\nZZ\u2020 = ")
            print(ZZ_mirror_)
    
    if QCRP_ is not None:
        print("  Random Paulis:")
        print(QCRP_)
        
    if QCRS_ is not None:
        print("  Resultant Paulis:")
        print(QCRS_)
        