"""
This script is an example illustrating the process of estimating the accuracy of the pUCCD algorithm on mock hydrogen chains. 
The script simulates hydrogen chains of different lengths (num_qubits), constructs the corresponding pUCCD circuits, and then computes their expectation values using a noiseless simulation.
"""
import sys
sys.path[1:1] = [ "Dev", "hydrogen-lattice/Dev" ]
sys.path[1:1] = [ "../../Dev", "../../Dev", "../../hydrogen-lattice/Dev/" ]

import numpy as np
import ansatz,simulator
from ansatz import PUCCD 
from simulator import Simulator 



# Create an instance of the Simulator class for noiseless simulations
ideal_backend = Simulator()

# Initialize an empty list to accumulate simulation data for hydrogen chains of different lengths
simulation_data = []

# Instantiate the pUCCD algorithm
puccd = PUCCD()

# Define the number of shots (number of repetitions of each quantum circuit)
# For the noiseless simulation, we use 10,000 shots.
# For the statevector simulator, this would be set to None.
shots = 10_000


# Loop over hydrogen chains with different numbers of qubits (from 2 to 4 in this example)
for num_qubits in range(2, 5):
    # Construct the pUCCD circuit for the current mock hydrogen chain
    circuit = puccd.build_circuit(num_qubits)

    # Assign small random parameters to the circuit. These would typically be optimized during a VQE process
    # For simplicity, we use either -1e-3 or 1e-3 here.
    circuit.assign_parameters(
        [np.random.choice([-1e-3, 1e-3]) for _ in range(len(circuit.parameters))],
        inplace=True,
    )

    # Generate a mock Hamiltonian for the pUCCD algorithm. In a real-world scenario, you would replace this with a 
    # Hamiltonian from a physical system or from a pre-generated file.
    operator = puccd.generate_mock_hamiltonian(num_qubits)

    # Compute and print the expectation value of the circuit with respect to the Hamiltonian,
    # using the noiseless backend
    ideal_energy = ideal_backend.compute_expectation(
        circuit, operator=operator, shots=shots
    )
    print(ideal_energy)
