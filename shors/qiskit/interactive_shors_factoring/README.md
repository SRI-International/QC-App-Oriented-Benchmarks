# Interactive Shors Factoring Algorithm - Qiskit

Implementation of Shor's Factoring Algorithm in Qiskit

This is a clone of the code in ShorsAlgQiskit
at https://github.com/ttlion/ShorAlgQiskit

The code is reproduced here to provide an interactive version of Shor's algorithm. The algorithm in 
used in this interactive version is the same version used to benchmark Shor's period finding algorithm
and perform Shor's factoring algorithm. 

```
TU Delft - AP3421 Project 18/19. Authors: Rui Maia (4942728) and Tiago Leao (4937589)

In this directory there are the Python files using the SDK Qiskit to implement Shor's Algorithm.

The algorithm was implemented using the description in the paper:

Stephane Beauregard (2003), Circuit for Shors algorithm using 2n+3 qubits, Quantum Information
and Computation, Vol. 3, No. 2 (2003) pp. 175-185. Also on quant-ph/0205095

This paper is referred the "base paper".

All the files work with Qiskit version 0.7.0, that is a python library/package for Quantum Computing.
To use Qiskit, Python must be installed and then Qiskit can be installed through:
pip install qiskit
If, when running this project, the Qiskit version installed is a later version, the project may naturally
not work if some functions change. If it does not work, please uninstall that later Qiskit version through
pip uninstall qiskit
And then install the Qiskit version used in this project, through:
pip install qiskit==0.7.0

This directory contains:
-> this readme
-> Shor_Normal_QFT.py
	- Has the implementation of Shor's Algorithm using as quantum circuit the first
	  simplification of the base paper, the version using the "normal" QFT,
	  that has a top register with 2n qubits
-> Shor_Sequential_QFT.py
	- Has the implementation of Shor's Algorithm using as quantum circuit the second
	  simplification explained in the base paper, the version using the "sequential" QFT,
	  that has a top register with only 1 qubit and does 2n measurements until getting
	  at the end of the circuit, being the results of those measurements the x_final,
	  not needing to actually apply a QFT at the end of the circuit
-> Test_QFT.py
	- Test file for user to test correctness of QFT implementation
-> Test_Mult.py
	- Test file for user to test correctness of the modular multiplication component
-> Test_classical_before_quantum.py
	- Test file for user to test correctness of the classical data treatment before the quantum circuit
-> Test_classical_after_quantum.py
	- Test file for user to test correctness of the classical data treatment after the quantum circuit
```
