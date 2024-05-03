"""
This script is an example illustrating the process of estimating the accuracy of the pUCCD algorithm on real hamiltonians. 
The script simulates hydrogen chains of different lengths (num_qubits), constructs the corresponding pUCCD circuits, and then computes their expectation values using a noiseless simulation.
"""


# Note this version doesn't have classical optimization, so it's not going to be as accurate as the other one


import sys

sys.path[1:1] = [ "_common", "_common/Dev", "hydrogen-lattice/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit", "../../hydrogen-lattice/_common/" ]

import numpy as np
from ansatz import PUCCD
from simulator import Simulator
from pathlib import Path
import common
import os
from qiskit.opflow.primitive_ops import PauliSumOp
import matplotlib.pyplot as plt

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

# Get the path of the instance for accessing instance files using current folder
current_folder_path = Path(os.path.dirname(os.path.abspath(__file__)))
instance_folder_path = Path(os.path.join(current_folder_path.parent, "_common", "instances"))

# Get a list of problem files and solution files in the instance folder
problem_files = list(instance_folder_path.glob("**/*.json"))
solution_files = list(instance_folder_path.glob("**/*.sol"))

# Extract the number of qubits from the problem file names and store them in a list
num_qubits = [int(problem.name.split("_")[0][1:]) for problem in problem_files]

# Print the number of qubits
print(f"num_qubits: {num_qubits}")

# Create an empty list to store PauliSumOps
pauli_ops = []

""" Loop through each problem file path and read the paired instance from the problem file and
 Create a list of tuples, each containing an operator and its coefficient  and then 
 convert the list to a PauliSumOp objects and add them to the pauli_ops list"""
 
# Reading paired instances from the problem files
for file_path in problem_files:

    ops, coefs = common.read_paired_instance(file_path)
    hamiltonians = list(zip(ops, coefs))
    pauli_ops.append(PauliSumOp.from_list(hamiltonians))
    
 
solutions = []

# Reading energes from the solution files
for file_path in solution_files:
    method_names, values = common.read_puccd_solution(file_path)
    solution = list(zip(method_names, values))
    solutions.append(solution)   
    
print("Note - This wont be accurate as we are not using optimization")
    
# Loop over hydrogen chains with different numbers of qubits (from 2 to 4 in this example)
for index, pauli_op in enumerate(pauli_ops):
    
    # Construct the pUCCD circuit for the current number of qubits
    circuit = puccd.build_circuit(num_qubits[index])
    
    # Set the PauliSumOp object as the operator or Hamiltonian of the circuit
    operator = pauli_op

    # Assign small random parameters to the circuit. These would typically be optimized during a VQE process
    # For simplicity, we use either -1e-3 or 1e-3 here.
    circuit.assign_parameters(
        [np.random.choice([-1e-3, 1e-3]) for _ in range(len(circuit.parameters))],
        inplace=True,
    )

    """ Compute and print the expectation value of the circuit with respect to the Hamiltonian,
     using the noiseless backend """
    print(f"\nThis is for {num_qubits[index]} qubits")
    ideal_energy = ideal_backend.compute_expectation(circuit, operator=operator, shots=shots)
    print(f"PUCCD calculated energy  : {ideal_energy}")

    # classical_energy is calculated using np.linalg.eigvalsh 
    solution = solutions[index]
    doci_energy = float(next(value for key, value in solution if key == 'doci_energy'))
    fci_energy = float(next(value for key, value in solution if key == 'fci_energy'))
    
    print(f"DOCI calculated energy : {doci_energy}")
    print(f"FCI calculated energy : {fci_energy}")
    
    plt.figure()
    plt.plot(num_qubits, [doci_energy] * len(num_qubits), 'r*', label='DOCI Energy for given Hamiltonian')
    plt.plot(num_qubits, [fci_energy] * len(num_qubits), 'g^', label='FCI Energy for given Hamiltonian')
    plt.plot(num_qubits, [ideal_energy] * len(num_qubits), 'bo', label='PUCCD Energy for given Hamiltonian')

    plt.xlabel('Number of Qubits')
    plt.ylabel('Energy')
    plt.title('Energy Comparison: Quantum vs. Classical')
    plt.legend()

    # Generate the text to display
    energy_text = f'Quantum Energy: {ideal_energy:.2f}  |  DOCI Energy: {doci_energy:.2f}  |  \
    FCI Energy: {fci_energy:.2f}'

    # Add the text annotation at the top of the plot
    plt.annotate(energy_text, xy=(0.5, 0.97), xycoords='figure fraction', ha='center', va='top')
    plt.show()