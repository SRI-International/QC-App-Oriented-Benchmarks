'''
Hamiltonian Simulation Benchmark Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

'''
There are multiple Hamiltonians and three methods defined for this kernel.
Hamiltonians are applied via a base class HamiltonianKernel and derived classes for specific hamiltonians.
The Hamiltonian name is specified in the "hamiltonian" argument.
The "method" argument indicates the type of fidelity comparison that will be done. 
In this case, method 3 is used to create a mirror circuit for scalability.
'''
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
import math

pi = math.pi

# Gates to be saved for printing purpose.
# DEVNOTE: these are global for now, for convenience; improve later
XX_ = None
YY_ = None
ZZ_ = None
XXYYZZ_ = None
XX_mirror_ = None
YY_mirror_ = None
ZZ_mirror_ = None
XXYYZZ_mirror_ = None
XXYYZZ_quasi_mirror_ = None


# For validating the implementation of XXYYZZ operation (saved for possible use in drawing)
_use_XX_YY_ZZ_gates = False

## use initial state in the abstract class
class HamiltonianKernel(object):

    # class variables saved for printing
    MAX_PRINT_SIZE = 6  # default maximum size to print circuit
    QCI_ = None         # Initial Circuit
    QC_ = None          # Total Circuit
    QCH_ = None         # Hamiltonian 
    QC2D_ = None        # Mirror Circuit
    QCRS_ = None        # Resultant Pauli
    
    def __init__(self, n_spins, K, t, hamiltonian, w, hx, hz, use_XX_YY_ZZ_gates, method, random_pauli_flag, init_state):
        self.n_spins = n_spins
        self.K = K
        self.t = t
        self.tau = t / K
        self.hamiltonian = hamiltonian
        self.w = w
        self.h_x = hx
        self.h_z = hz   
        self.use_XX_YY_ZZ_gates = use_XX_YY_ZZ_gates
        self.random_pauli_flag = random_pauli_flag
        self.method = method
        
        # DEVNOTE: this shouldn't be here, instead if None, we should not add an initial state
        if init_state == None:
            if hamiltonian == "tfim": 
                init_state = "ghz"
            else:
                init_state = "checkerboard"
                
        self.init_state = init_state

        self.qr = QuantumRegister(n_spins)
        self.cr = ClassicalRegister(n_spins)
        self.qc = QuantumCircuit(self.qr, self.cr, name = hamiltonian)

    def overall_circuit(self):
    
        # create initial state and append to the overall circuit
        i_state = self.initial_state()
        self.qc.append(i_state, self.qr)
        
        # create Hamiltonian evolution circuit and append to the overall circuit
        hamiltonian_circuit = self.create_hamiltonian()
        self.qc.append(hamiltonian_circuit, self.qr)

        # if mirrored, create the inverse circuit and append to overall circuit
        inverse_circuit = None
        if self.method == 3:
            #checks if random pauli flag is true to apply quasi inverse.
            if self.random_pauli_flag:
                inverse_circuit = self.create_quasi_inverse_hamiltonian() 
                self.qc.append(inverse_circuit, self.qr)
            else:
                #applies regular inverse.
                inverse_circuit = self.create_inverse_hamiltonian() 
                self.qc.append(inverse_circuit, self.qr)
        
        # Measure all qubits
        for i_qubit in range(self.n_spins):
            self.qc.measure(self.qr[i_qubit], self.cr[i_qubit])

        # Save circuits and subcircuits for possible display (explicitly set class variables)
        if self.n_spins <= self.MAX_PRINT_SIZE:
            HamiltonianKernel.QC_ = self.qc
            HamiltonianKernel.QCI_ = i_state
            HamiltonianKernel.QCH_ = hamiltonian_circuit
            HamiltonianKernel.QC2D_ = inverse_circuit

        # Collapse the sub-circuits used in this benchmark (for Qiskit)
        qc2 = self.qc.decompose().decompose()      

        return qc2

    #apply initial state to the quantum circuit
    def initial_state(self) -> QuantumCircuit:
        #Initialize the quantum state.
        qr = QuantumRegister(self.n_spins)
        qc = QuantumCircuit(qr, name = "InitialState")
        if self.init_state == "checkerboard" or self.init_state == "neele":
            # Checkerboard state, or "Neele" state
            for k in range(0, self.n_spins, 2):
                qc.x([k])
        elif self.init_state == "ghz":
            # GHZ state: 1/sqrt(2) (|00...> + |11...>)
            qc.h(0)
            for k in range(1, self.n_spins):
                qc.cx(k-1, k)

        return qc

    def create_hamiltonian(self) -> QuantumCircuit:
        pass

    def create_inverse_hamiltonian(self) -> QuantumCircuit:
        pass

    def create_quasi_inverse_hamiltonian(self) -> QuantumCircuit:
        pass
    
    ### List of random paulis to apply if method == 3.
    def random_paulis_list(self):
        """Create a list of random paulis to apply to mirror circuit."""
        pauli_tracker_list = []
        for i in range(self.n_spins):
            gate = np.random.choice(["x","z"])
            if gate == "x":
                pauli_tracker_list.append("x")
            if gate == "z":
                pauli_tracker_list.append("z")                
        return pauli_tracker_list

    #### Resultant Pauli after applying quasi inverse Hamiltonian
    def ResultantPauli(self)-> QuantumCircuit:
        """Create a quantum oracle that is the result of applying quasi inverse Hamiltonain and random Pauli to Hamiltonian."""
        qr = QuantumRegister(self.n_spins)
        qc = QuantumCircuit(qr, name = "ResultantPaulis")
        for n in range(self.n_spins):
            qc.x(n)      # You can apply any Pauli, but you must also change the state you are comparing with.
            
        qc.barrier()
        self.QCRS_ = qc
        return qc

    #### Draw the circuits of this kernel
    def kernel_draw(self):
                
        # Print a sample circuit
        print("Sample Circuit:")
        if self.QC_ is not None:
            print(self.QC_)
        else:
            print("  ... circuit too large to print!")
            return False    
            
        if self.QCI_ is not None:
            print("  Initial State:")
            print(self.QCI_)
        
        if self.QCH_ is not None:        
            print(f"  Hamiltonian ({self.QCH_.name if self.QCH_ is not None else '?'}):")
            print(self.QCH_)
        
        if self.QC2D_ is not None:
            print(f"  Inverse Hamiltonian ({self.QC2D_.name if self.QC2D_ is not None else '?'}):")
            print(self.QC2D_)
            
        if self.QCRS_ is not None:
            print("  Resultant Paulis:")
            print(self.QCRS_)
        
        # reset these variables after printing;
        # (since we don't intialize them anywhere, they could be retained incorrectly on subsequent runs)
        HamiltonianKernel.QC_ = HamiltonianKernel.QCI_ = HamiltonianKernel.QCH_ = HamiltonianKernel.QC2D_ = HamiltonianKernel.QCRS_ = None

        return True
 
########################
####*****************###
########################

# Derived class for Heisenberg.
class HeisenbergHamiltonianKernel(HamiltonianKernel):

    #apply Heisenberg hamiltonian.
    def create_hamiltonian(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_spins)
        qc = QuantumCircuit(qr, name="Heisenberg")
        for k in range(self.K):
            [qc.rx(2 * self.tau * self.w * self.h_x[i], qr[i]) for i in range(self.n_spins)]
            [qc.rz(2 * self.tau * self.w * self.h_z[i], qr[i]) for i in range(self.n_spins)]
            qc.barrier()

            if self.use_XX_YY_ZZ_gates:
                for j in range(2):
                    for i in range(j % 2, self.n_spins - 1, 2):
                        qc.append(xx_gate(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])
                for j in range(2):
                    for i in range(j % 2, self.n_spins - 1, 2):
                        qc.append(yy_gate(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])
                for j in range(2):
                    for i in range(j % 2, self.n_spins - 1, 2):
                        qc.append(zz_gate(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])
            else:
                for j in range(2):
                    for i in range(j % 2, self.n_spins - 1, 2):
                        qc.append(xxyyzz_opt_gate(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])
            qc.barrier()
  
        return qc

    #apply inverse of the hamiltonian to simulate negative time evolution.
    def create_inverse_hamiltonian(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_spins)
        qc = QuantumCircuit(qr, name="Heisenberg\u2020")
        for k in range(self.K): 
            if self.use_XX_YY_ZZ_gates:
                for j in range(2):
                    for i in range(j % 2, self.n_spins - 1, 2):
                        qc.append(zz_gate_mirror(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])
                for j in reversed(range(2)):
                    for i in reversed(range(j % 2, self.n_spins - 1, 2)):
                        qc.append(yy_gate_mirror(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])
                for j in range(2):
                    for i in range(j % 2, self.n_spins - 1, 2):
                        qc.append(xx_gate_mirror(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])
            else:
                for j in reversed(range(2)):
                    for i in reversed(range(j % 2, self.n_spins - 1, 2)):
                        qc.append(xxyyzz_opt_gate_mirror(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])
            qc.barrier()
            [qc.rz(-2 * self.tau * self.w * self.h_z[i], qr[i]) for i in range(self.n_spins)]
            [qc.rx(-2 * self.tau * self.w * self.h_x[i], qr[i]) for i in range(self.n_spins)]
            qc.barrier()

        return qc

    #create quasi inverse hamiltonian to simulate negative time evolution with randomized paulis applied.
    def create_quasi_inverse_hamiltonian(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_spins)
        qc = QuantumCircuit(qr, name = "Quasi-Heisenberg\u2020")

        # Apply random paulis
        pauli_list = self.random_paulis_list()

        for i, gate in enumerate(pauli_list):
            if gate == "x":
                qc.x(qr[i])
            else:
                qc.z(qr[i])

        qc.barrier()

        self.QCRS_ = res_pauli = self.ResultantPauli() # create a resultant pauli that we want to apply to initial state.
                
        for k in range(self.K): 
            # Basic implementation of exp(-i * t * (XX + YY + ZZ)):
            if self.use_XX_YY_ZZ_gates:
                # regular inverse of XX + YY + ZZ operators on each pair of quibts in linear chain
                # XX operator on each pair of qubits in linear chain
                for j in range(2):
                    for i in range(j%2, self.n_spins - 1, 2):
                        qc.append(zz_gate_mirror(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])

                # YY operator on each pair of qubits in linear chain
                for j in range(2):
                    for i in range(j%2, self.n_spins - 1, 2):
                        qc.append(yy_gate_mirror(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])

                # ZZ operation on each pair of qubits in linear chain
                for j in range(2):
                    for i in range(j%2, self.n_spins - 1, 2):
                        qc.append(xx_gate_mirror(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])

            else:
                # optimized Inverse of XX + YY + ZZ operator on each pair of qubits in linear chain
                for j in reversed(range(2)):
                    
                    #Keep a track of what pauli is applied at the first part of mirror circuit.
                    if j == 0 and k == 0:
                        if self.n_spins % 2 == 1:  
                            if pauli_list[0] == "x":
                                qc.x(qr[0])
                            if pauli_list[0] == "z":
                                qc.z(qr[0])
                                                                ###applying a little twirl to prevent compiler from creating identity.
                        if self.n_spins % 2 == 0:
                            if pauli_list[0] == "x":
                                qc.x(qr[0])
                            if pauli_list[0] == "z":
                                qc.z(qr[0])
                            if pauli_list[self.n_spins-1] == "x":
                                qc.x(qr[self.n_spins-1])
                            if pauli_list[self.n_spins-1] == "z":
                                qc.z(qr[self.n_spins-1])

                    for i in reversed(range(j % 2, self.n_spins - 1, 2)):
                        if k == 0 and j == 1:
                            gate_i = pauli_list[i]
                            gate_next = pauli_list[(i + 1) % self.n_spins]
                            qc.append(xxyyzz_opt_gate_quasi_mirror(self.tau, gate_i, gate_next).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])
                                                                                
                        else:
                            qc.append(xxyyzz_opt_gate_mirror(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])     

            qc.barrier()

            # the Pauli spin vector product
            [qc.rz(-2 * self.tau * self.w * self.h_z[i], qr[i]) for i in range(self.n_spins)]
            [qc.rx(-2 * self.tau * self.w * self.h_x[i], qr[i]) for i in range(self.n_spins)]
            qc.barrier()

        qc.append(self.QCRS_,qr)
        #self.QC2D_ = qc
        return qc

    # draw circuit and apply extra circuit printing specific to this kernel type.
    def kernel_draw(self):
    
        if super().kernel_draw() == False:
            return
 
        if self.use_XX_YY_ZZ_gates:
                print("\nXX, YY, ZZ = ")
                print(XX_)
                print(YY_)
                print(ZZ_)
                if self.method == 3:
                    print("\nXX, YY, ZZ \u2020 = ")
                    print(XX_mirror_)
                    print(YY_mirror_)
                    print(ZZ_mirror_)
        else:
            print("\nXXYYZZ = ")
            print(XXYYZZ_)  
            if self.method == 3:
                print("\nXXYYZZ\u2020 = ")
                print(XXYYZZ_mirror_)             
            if self.random_pauli_flag:
                print("Qusai Inverse XXYYZZ:")
                print(XXYYZZ_quasi_mirror_)


########################
####*****************###
########################

#Derived Class for TFIM.
class TfimHamiltonianKernel(HamiltonianKernel):

    #apply tfim hamiltonian.
    def create_hamiltonian(self) -> QuantumCircuit:
        self.h = 1
        qr = QuantumRegister(self.n_spins)
        qc = QuantumCircuit(qr, name="TFIM")
        for k in range(self.K):
            for i in range(self.n_spins):
                qc.rx(2 * self.tau * self.h, qr[i])
            qc.barrier()
            for j in range(2):
                for i in range(j % 2, self.n_spins - 1, 2):
                    qc.append(zz_gate(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])
            qc.barrier()

        return qc

    #apply inverse of the hamiltonian to simulate negative time evolution.
    def create_inverse_hamiltonian(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_spins)
        qc = QuantumCircuit(qr, name="TFIM\u2020")
        for k in range(self.K):
            for j in range(2):
                for i in range(j % 2, self.n_spins - 1, 2):
                    qc.append(zz_gate_mirror(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])
            qc.barrier()
            for i in range(self.n_spins):
                qc.rx(-2 * self.tau * self.h, qr[i])
            qc.barrier()

        return qc

    #create quasi inverse hamiltonian to simulate negative time evolution with randomized paulis applied.
    def create_quasi_inverse_hamiltonian(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_spins)
        qc = QuantumCircuit(qr, name="Quasi-TFIM\u2020")
        for k in range(self.K):
            for j in range(2):
                for i in range(j % 2, self.n_spins - 1, 2):
                    qc.append(zz_gate_mirror(self.tau).to_instruction(), [qr[i], qr[(i + 1) % self.n_spins]])
            qc.barrier()
            for i in range(self.n_spins):
                qc.rx(-2 * self.tau * self.h, qr[i])
            qc.barrier()

        return qc
    
    # draw circuit and apply extra circuit printing specific to this kernel type.
    def kernel_draw(self):
    
        if super().kernel_draw() == False:
            return
            
        print("\nZZ = ")
        print(ZZ_)
        if self.method == 3:
            print("\nZZ\u2020 = ")
            print(ZZ_mirror_)


########################
####*****************###
########################

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
    qc.rz(pi * tau, qr[1])
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
    qc.rz(pi * tau, qr[1])
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
    qc.rz(pi * tau, qr[1])
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
    qc.rz(pi / 2, qr[1])
    qc.cx(qr[1], qr[0])
    qc.rz(pi * gamma - pi / 2, qr[0])
    qc.ry(pi / 2 - pi * alpha, qr[1])
    qc.cx(qr[0], qr[1])
    qc.ry(pi * beta - pi / 2, qr[1])
    qc.cx(qr[1], qr[0])
    qc.rz(-pi / 2, qr[0])

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
    qc.rz(-pi * tau, qr[1])
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
    qc.rz(-pi * tau, qr[1])
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
    qc.rz(-pi * tau, qr[1])
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
    qc.rz(pi / 2, qr[0])
    qc.cx(qr[1], qr[0])
    qc.ry(-pi * beta + pi / 2, qr[1])
    qc.cx(qr[0], qr[1])
    qc.ry(-pi / 2 + pi * alpha, qr[1])
    qc.rz(-pi * gamma + pi / 2, qr[0])
    qc.cx(qr[1], qr[0])
    qc.rz(-pi / 2, qr[1])

    global XXYYZZ_mirror_
    XXYYZZ_mirror_ = qc

    return qc


def xxyyzz_opt_gate_quasi_mirror(tau: float, pauli1: str, pauli2: str) -> QuantumCircuit:
    """
    Optimal combined XXYYZZ quasi mirror gate (with double coupling) on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The optimal combined XXYYZZ_mirror_ gate circuit.
    """
    alpha = tau
    beta = tau
    gamma = tau
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="XXYYZZ~Q")

    if pauli1 == "x":
        qc.h(qr[0])
        qc.z(qr[0])
        qc.rx(pi / 2, qr[0])   #### X(Random Pauli) --- X --- Rz is equivalent to X ------ H - Z - H ----Rz
        qc.h(qr[0])                #### which is equivalent to X ------ H - Z - Rx ----H
        
    if pauli1 == "z":
        qc.h(qr[0])
        qc.x(qr[0])  
        qc.rx(pi / 2, qr[0])   #### X(Random Pauli) --- Z --- Rz is equivalent to Z ------ H - X - H ----Rz
        qc.h(qr[0])                #### #### which is equivalent to Z ------ H - X - Rx ----H

    if pauli2 == "x":
        qc.x(qr[1])
    if pauli2 == "z":
        qc.z(qr[1])

    qc.cx(qr[1], qr[0])
    qc.ry(-pi * beta + pi / 2, qr[1])
    qc.cx(qr[0], qr[1])
    qc.ry(-pi / 2 + pi * alpha, qr[1])
    qc.rz(-pi * gamma + pi / 2, qr[0])
    qc.cx(qr[1], qr[0])
    qc.rz(-pi / 2, qr[1])

    global XXYYZZ_quasi_mirror_
    XXYYZZ_quasi_mirror_ = qc

    return qc


