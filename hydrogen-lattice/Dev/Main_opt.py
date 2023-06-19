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
from qiskit.algorithms.optimizers import COBYLA



# Create an instance of the Simulator class for noiseless simulations
ideal_backend = simulator.Simulator()

# Initialize an empty list to accumulate simulation data for hydrogen chains of different lengths
simulation_data = []

# Instantiate the pUCCD algorithm
puccd = PUCCD()

# Define the number of shots (number of repetitions of each quantum circuit)
# For the noiseless simulation, we use 10,000 shots.
# For the statevector simulator, this would be set to None.
shots = 10_000

def compute_energy(circuit, operator, shots):
    # Compute the expectation value of the circuit with respect to the Hamiltonian for optimization
    ideal_energy = ideal_backend.compute_expectation(circuit, operator=operator, shots=shots)
    return ideal_energy

# Loop over hydrogen chains with different numbers of qubits (from 2 to 4 in this example)
for num_qubits in range(2, 5):
    # Construct the pUCCD circuit for the current mock hydrogen chain
    circuit = puccd.build_circuit(num_qubits)

    operator = puccd.generate_mock_hamiltonian(num_qubits)

    # Initialize the parameters with -1e-3 or 1e-3
    initial_parameters = [np.random.choice([-1e-3, 1e-3]) for _ in range(len(circuit.parameters))]
    circuit.assign_parameters(initial_parameters, inplace=True)

    # Initialize the COBYLA optimizer
    optimizer = COBYLA(maxiter=1000, tol=1e-6,disp=False)
    
    # Optimize the circuit parameters using the optimizer
    optimized_parameters = optimizer.minimize(lambda parameters: 
        compute_energy(circuit, operator, shots=shots), x0 =  np.random.choice([-1e-3, 1e-3]))

    # Extract the parameter values from the optimizer result
    optimized_values = optimized_parameters.x

    # Create a dictionary of {parameter: value} pairs
    parameter_values = {param: value for param, value in zip(circuit.parameters, optimized_values)}

    # Assign the optimized values to the circuit parameters
    circuit.assign_parameters(parameter_values, inplace=True)
    
    ideal_energy = ideal_backend.compute_expectation(circuit, operator=operator, shots=shots)
    print(ideal_energy)
