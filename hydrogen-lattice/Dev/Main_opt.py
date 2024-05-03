"""
This script is an example illustrating the process of estimating the accuracy of the pUCCD algorithm on real hamiltonians
The script simulates hydrogen chains of different lengths (num_qubits), constructs the corresponding pUCCD circuits, 
and then computes their expectation values using a noiseless simulation.
"""
from pathlib import Path

import numpy as np
from qiskit.opflow.primitive_ops import PauliSumOp
import matplotlib.pyplot as plt
from ansatz import PUCCD
from simulator import Simulator
import os

from scipy.optimize import minimize
import sys

# Create a list to store the plots
plots = []

sys.path[1:1] = [ "_common", "_common/Dev", "hydrogen-lattice/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit", "../../hydrogen-lattice/_common/" ]

import common
# Create an instance of the Simulator class for noiseless simulations
ideal_backend = Simulator()

# # Initialize an empty list to accumulate simulation data for hydrogen chains of different lengths
# simulation_data = []

# Instantiate the pUCCD algorithm
puccd = PUCCD()

# Define the number of shots (number of repetitions of each quantum circuit)
# For the noiseless simulation, we use 10,000 shots.
# For the statevector simulator, this would be set to None.
shots = 10_000

# Initialize an empty list to store the lowest energy values
lowest_energy_values = []

# objective Function to compute the energy of a circuit with given parameters and operator
def compute_energy(circuit, operator, shots, parameters): 
    
    # Bind the parameters to the circuit
    bound_circuit = circuit.assign_parameters(parameters)
    
    # Compute the expectation value of the circuit with respect to the Hamiltonian for optimization
    energy = ideal_backend.compute_expectation(bound_circuit, operator=operator, shots=shots)
    
    # Append the energy value to the list
    lowest_energy_values.append(energy)
    
    return energy


# Get the path of the instance for accessing instance files using current folder
current_folder_path = Path(os.path.dirname(os.path.abspath(__file__)))
instance_folder_path = Path(os.path.join(current_folder_path.parent, "_common", "instances"))

# Get a list of problem files and solution files in the instance folder
problem_files = list(instance_folder_path.glob("**/*.json"))
solution_files = list(instance_folder_path.glob("**/*.sol"))

# Extract the number of qubits from the problem file names and store them in a list
num_qubits_array = list(set([int(problem.name.split("_")[0][1:]) for problem in problem_files]))

# Print the number of qubits 
# Dev Note :- Here you cannot see multiple same qubit numbers even though you have multiple same qubit file for optimization
print(f"num_qubits array: {num_qubits_array}")

    

""" We will loop through 2,4 as of now but will change it as required in original implementation of run method
      and even initialize according to it """
min_qubits = min(num_qubits_array) #2
max_qubits = max(num_qubits_array) #4
max_circuits = 3

# For paired electrons it should be incremented 2 at a time
for num_qubits in range(min_qubits , max_qubits + 1, 2):
    
    instance_count = 0
    # Reading instances from the problem files
    for file_path in problem_files:
        # To optimize the loop we can only loop at our qubit number to avoid looping for all instances
        if int(file_path.name.split("_")[0][1:]) > num_qubits:
            break
        elif int(file_path.name.split("_")[0][1:]) == num_qubits and instance_count < max_circuits:
            
            # Not using ennumerate as may even have loop of other files
            instance_count += 1      
                      
            # Building PUCCD Ansatz circuit for currnt number of qubits
            circuit = puccd.build_circuit(num_qubits)
            
            # Initialize initial_parameters as an ndarray with shape (n,)
            initial_parameters = np.random.random(size=len(circuit.parameters))
            
            # Here we are reading the paired instances from the problem files
            ops, coefs = common.read_paired_instance(file_path)
            operator = PauliSumOp.from_list(list(zip(ops, coefs)))
            """Set the maximum number of iterations, tolerance, and display options and initialize 
            the COBYLA optimizerOptimize the circuit parameters using the optimizer"""
            optimized_parameters = minimize(
                lambda parameters: compute_energy(circuit, operator, shots=shots, parameters=parameters),
                x0=initial_parameters.ravel(),
                method='COBYLA',
                tol=1e-3,
                options={'maxiter': 100, 'disp': False}
            )
            
            # Extract the parameter values from the optimizer result
            optimized_values = optimized_parameters.x
            
            # Create a dictionary of {parameter: value} pairs
            parameter_values = {param: value for param, value in zip(circuit.parameters, optimized_values) }
            
            # Assign the optimized values to the circuit parameters
            circuit.assign_parameters(parameter_values, inplace=True)
            
            radius = os.path.basename(file_path).split('_')[2][:4]
            ideal_energy = ideal_backend.compute_expectation(circuit, operator=operator, shots=shots)
            print(f"\nBelow Energies are for problem file {os.path.basename(file_path)} is for {num_qubits} qubits and radius {radius} of paired hamiltionians")
            print(f"PUCCD calculated energy : {ideal_energy}")
        
            # classical_energy is calculated using np.linalg.eigvalsh 
            # solution is calculated using sol file created using sol file of current instance
            sol_file_name = os.path.splitext(file_path)[0] + ".sol"
            method_names, values = common.read_puccd_solution(sol_file_name)
            solution = list(zip(method_names, values))
            
            # Doci_energy and Fci energy is extracted from Solution file
            print(f"\nBelow Classical Energies are in solution file {os.path.basename(sol_file_name)} is {num_qubits} qubits and radius {radius} of paired hamiltionians")
            doci_energy = float(next(value for key, value in solution if key == 'doci_energy'))
            fci_energy = float(next(value for key, value in solution if key == 'fci_energy'))
            
            print(f"DOCI calculated energy : {doci_energy}")
            print(f"FCI calculated energy : {fci_energy}")
            
            # pLotting each instance of qubit count given 
            plt.figure()
            plt.plot(range(len(lowest_energy_values)), lowest_energy_values, label='Quantum Energy')
            plt.axhline(y=doci_energy, color='r', linestyle='--', label='DOCI Energy for given Hamiltonian')
            plt.axhline(y=fci_energy, color='g', linestyle='solid', label='FCI Energy for given Hamiltonian')
            plt.xlabel('Number of Iterations')
            plt.ylabel('Energy')
            plt.title('Energy Comparison: Quantum vs. Classical')
            plt.legend()
            # Generate the text to display
            energy_text = f'Ideal Energy: {ideal_energy:.2f} | DOCI Energy: {doci_energy:.2f} | FCI Energy: {fci_energy:.2f} | Num of Qubits: {num_qubits} | Radius: {radius}'

            # Add the text annotation at the top of the plot
            plt.annotate(energy_text, xy=(0.5, 0.97), xycoords='figure fraction', ha='center', va='top')
            plt.show()
            
    lowest_energy_values.clear()
 