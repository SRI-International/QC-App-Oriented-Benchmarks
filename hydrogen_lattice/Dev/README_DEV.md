# Hydrogen Lattice Simulation using VQE

Current folder dev ( Development in progress ) contains multiple files which are used to execute an example of a simulation to estimate the accuracy of finding the lowest energy state. The simulation aims to assess the performance of the algorithm by simulating hydrogen chains of different lengths and computing their expectation values.

## Background

The VQE algorithm is a quantum computing algorithm used for solving optimization problems. It combines classical optimization techniques with a quantum computing component to find the minimum energy of a given Hamiltonian. In the context of quantum chemistry, VQE is commonly used to estimate the ground state energy of molecules.


## Purpose

The purpose of this simulation is to evaluate how accurately the VQE algorithm with the pUCCD Ansatz can estimate the energy of a hydrogen chain. By varying the length of the hydrogen chain, we can observe how the algorithm performs on different system sizes and evaluate the ideal energy with respect to given hamiltonian
## Simulation Steps

1. **Simulation Setup**:
   - We initialize the simulation environment, including importing the necessary libraries and setting up the required dependencies.

2. **Constructing the pUCCD Circuit (ansatz.py)**:
  - For each hydrogen chain length ( Each qubit count used ), we construct the corresponding pUCCD Ansatz     circuit. This circuit represents the parameterized quantum circuit used in the VQE algorithm to estimate the energy.
   - The pUCCD Ansatz circuit is built using quantum gates and operations. It includes both single and two-qubit gates to capture electron correlations.

3. **Assigning Random Parameters**:
   - In order to run the pUCCD circuit, we need to assign parameters to the circuit using one of below two methods

   ***Method 1*:**
   - These parameters determine the behavior of the circuit and are typically optimized during a VQE   (Variational Quantum Eigensolver) optimization process (Main_opt.py)

   ***Method 2*:**
   - In this simulation, we use random values (one value from [-1e-3, 1e-3] )for simplicity to assess the algorithm's performance.

4. **Computing Expectation Values**:
   - Using a noiseless simulation backend, we compute the expectation value of the pUCCD circuit with respect to a mock Hamiltonian. 
   - The expectation value represents the estimated energy of the hydrogen chain. 
   - The Hamiltonian is a mathematical representation of the system's energy, and we use it to calculate the expectation value.


5. **Results and Analysis**:
   - Finally, we present the computed expectation values for each hydrogen chain length 
   - This information can provide insights into the algorithm's capabilities and limitations. 
   - By comparing the computed expectation values to known values, we can assess the accuracy of the algorithm.

## How to Run Algorithm

The benchmark programs may be run directly by running Main_opt.py for version with optimization or running Main.py which Includes parameter value from a fixed set of parameters. Apart from that one can run manually in a command shell. In a command window or shell, change directory to the application you would like to execute. Then, simply execute a line similar to the following, to begin execution of the main program for the application:

To execute version which includes classical optimization
```
    cd hydrogen-lattice/Dev
    python Main_opt.py
```

To execute version which has fixed set of parameters
```
    cd hydrogen-lattice/Dev
    python Main.py
```




## Conclusion

This simulation provides an opportunity to evaluate the accuracy of the pUCCD algorithm on mock hydrogen chains. By simulating different system sizes, we can gain insights into the algorithm's performance and its potential applications in quantum chemistry. The results can guide further research and development of quantum algorithms for electronic structure calculations.
