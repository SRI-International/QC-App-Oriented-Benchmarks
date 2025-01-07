
import numpy as np

import evolution_exact

H_terms = [
    ('XXII', 0.5),
    ('IYYI', 0.3),
    ('IIZZ', 0.4),
    ('XYII', 0.2),
    ('IIYX', 0.6),
    ('IZXI', 0.1),
    ('XIII', 0.7)
]

print("")
print(H_terms)

# initialize 0 state based on width of first term's Pauli string
dimension = len(H_terms[0][0])
initial_state = np.zeros((2**dimension), dtype=complex)
initial_state[0] = 1  # Set the amplitude for |00> state

# for testing string initialization
# initial_state = "checkerboard"
# initial_state = "0000"
# initial_state = ""

print(initial_state)

#############################

# TEST 1
# s/b energy = 0.4

total_evolution_time = 0.5

# Compute the theoretical energy using an exact computation
theoretical_energies_exact, time_values = evolution_exact.compute_theoretical_energies(
        initial_state,
        H_terms,
        total_evolution_time,
        total_evolution_time)

print("")
print(f"For evolution time = {total_evolution_time}:")
print(f"  Theoretical energy (exact): {theoretical_energies_exact[0]}")
print("")

#############################

# TEST 2

try:
    from qiskit import QuantumCircuit 
    from qiskit_algorithms import TimeEvolutionProblem, SciPyRealEvolver
    from qiskit.quantum_info import Statevector, SparsePauliOp

    def convert_to_sparse_pauli_op(pauli_terms):
        """
        Convert an array of (coefficient, pauli string) tuples into a SparsePauliOp.
        
        Args:
        pauli_terms (list): List of tuples, each containing (coeff, pauli)
        
        Returns:
        SparsePauliOp: Qiskit SparsePauliOp representation of the Hamiltonian
        """
             
        if (pauli_terms is None):
            return None
        
        coeffs = []
        paulis = []

        for pauli_string, coeff in pauli_terms:
            coeffs.append(coeff)
            paulis.append(pauli_string)
        
        return SparsePauliOp(paulis, coeffs)
    
    sparse_pauli_op = convert_to_sparse_pauli_op(H_terms)
    print(sparse_pauli_op)
      
    # Compute the theoretical energy using an exact computation
    theoretical_energies_exact, time_values = evolution_exact.compute_theoretical_energies_spo_sv(
            initial_state,
            sparse_pauli_op,
            total_evolution_time,
            total_evolution_time)

    print("")
    print(f"For evolution time = {total_evolution_time}:")
    print(f"  Theoretical energy (exact): {theoretical_energies_exact[0]}")
    print("")
    
    
except Exception as ex:
    print("WARNING: Qiskit-dependent compute observable value functions are not available")
    print(f"Exception: {ex}") 
    