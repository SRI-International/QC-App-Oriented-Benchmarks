# HHL Algorithm - Linear Equation Solver

The HHL (HHL) quantum algorithm [[1]](#references) may demonstrate a quantum speedup over its classical counterpart for solving linear equations. 

The HHL offers a more complex algorithm that uses a combination of Quantum Phase Estimation and Quantum Fourier Transform along with state initialization and a new and unique, scalable, inverse rotation algorithm. The combination of the component routines taken together provides an extension to the benchmark suite that fills a gap between the QFT and Amplitude Estimation algorithms.



This is also a benchmark that can measure fidelity of circuit execution, but can also provide an application specific metric based on how well the algorithm solves the linear equation.

NOTE: The remainder of this README needs to be modifed with content for HHL.

## Problem outline

Insert HHL picture here. 

Given the linear equation <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}A\vec{x}=\vec{b}\end{align*}\"/>, where A is a N_b x N_b matrix. In the quantum circuit that will be performed <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}N_{b}=2^{n_{b}}\end{align*}\"/> where n_b will be the number of qubits in the circuit. The restrictions on A is that it has to be an s-sparse Hermitian matrix, meaning that A has at most s elements in each row/column.

## Benchmarking
The HHL benchmark is available in the hhl folder in the master repo. It has a file to build the Hamiltonian Simulation circuit, a file to create the uniformally controlled rotational, and a file to execute the entire algorithm. To evalute the solution, the HHL algorithm is compared to the classical solution to the Ax=b, and the fidelity is computed. 
The HHL algorithm is benchmarked by running a sample sparse A matrix with a scalable number of qubits.Each circuit is repeated a number of times denoted by `num_shots`. We then run the algorithm for numbers of qubits between `min_qubits` and `max_qubits`, inclusive. The test returns the averages of the circuit creation times, average execution times, fidelities, and circuit depths, like all of the other algorithms. For this algorithm's fidelity calculation, as we always have a single correct state, we compare the returned measurements against the distribution that has the single state with 100% probability using our [noise-normalized fidelity calculation](../_doc/POLARIZATION_FIDELITY.md).

## Classical algorithm
Classically, linear equations can be solved a variety of ways, mainly taking the inverse of the matrix or applying Guassian Elimination. A refesher of Guassian Elimination can be found here: https://en.wikipedia.org/wiki/Gaussian_elimination. However methods like Guassian Elimination can take <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}O(n^3)"> time.

## Quantum algorithm

The HHL algorithm has five main components, namely state preparation, Quantum Phase Estimation, Controlled RY Rotation for the ancilla qubit, Inverse Quantum Phase Estimation, and measurement.


The state starts with the initial state, <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}|\Psi_{1}\rangle=|b\rangle_{b}|0...0\rangle_{c}|0\rangle_{a}\end{align*}\"/> and the final state <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}|\Psi_{f}\rangle=|x\rangle_{b}|0\rangle_{c}|1\rangle_{a}\end{align*}\"/>


### General Quantum Circuit
The following circuit is the general quantum circuit for the HHL algorithm with <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n_b"> b-register qubits, <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> c-register(clock) qubits and 1 ancilla qubit. 

<p align="center">
   <img align=center src="../_doc/images/bernstein-vazirani/bv_circuit.png"  width="600" />
</p>

*Fig 1. Diagram of general quantum circuit for Bernstein-Vazirani Algorithm [[2]](#references)*

References [[2]](#references) and [[3]](#references) both have overviews of the mathematical details of the 
algorithm, but the key points will be reproduced here.

### Algorithmic Visualization

<p align="center">
<img align=center src="../_doc/images/bernstein-vazirani/bv_gif.gif"  width="700" />
</p>

*Fig 2. Visualization of quantum circuit executing for Bernstein-Vazirani Algorithm. The visualization
demonstrates how each qubit and state evolves throughout the algorithm. Visualization created
with IBM's Quantum Composer and can be analyzed
[here](https://quantum-computing.ibm.com/composer/files/new?initial=N4IgdghgtgpiBcIBCA1ABAQQDYHMD2ATgJYAuAFlCADQgCOEAzpYgPIAKAogHICKGAygFk0AJgB0ABgDcAHTBEwAYywBXACYw0MujCxEARgEYxCxdtlg5tAjBxpaAbQBsAXQuKbdxc7dy5%2BiAJiGAJ7BwlfMACgohCww0jo4NDHEUTA5LCAZnSYuMcAFkiADzCAVkiyMIiLKscE2rC0xsccloci9oqLJNiU8NzM%2BsG%2BppH8hzb-DNHC8f7uuSI1asjl%2BMjFUtSXKkdF%2BRXHGqWjhwbTsfdtyd39%2BdWemYmLqOf%2B5um8-qm377DOl8hg4DnUBu1XmDPmAwb8wYCYeUSkiLLBGCobKs0ABaAB8aG8JzAaIYGM0wxx%2BO8rxJZLGlIJDmhtMxrRcDO8vxZ5I67LxjM61BAGgYHiIAAcSEQ8GAECAQABfIA).*

[//]: # (For more information about reading these circuit diagrams, visit internal documentation or link to qiskit circuit composer. We likely need to include information about Bloch sphere reading to really make this a useful visualization.)

### Algorithm Steps

The steps for the HHL algorithm are the following, with the state after each step, <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_n\rangle">:

1. Initialize two quantum registers and the ancilla qubit. The b-register register has <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n_b"> data qubits initialized to the b-vector which has <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2^{n_b}"> components. The c-register is initialized to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0...0\rangle"/> with n qubits. The ancilla qubit is initialized to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|1\rangle"/>.
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\Psi_{1}\rangle=|b\rangle_{b}|0...0\rangle_{c}|0\rangle_{a}"/>
   </p>
   
2. Apply the Hadamard gate to create an equal superposition of the clock qubits, creating an equal superposition state <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\frac{1}{\sqrt{2^{n}}}\sum_{x=0}^{2^n-1}|x\rangle"/> in the c-register.

   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\Psi_{2}\rangle=|b\rangle\frac{1}{\sqrt{2^{n}}}(|0\rangle{+}|1\rangle)^{{\otimes}n}|0\rangle_{a}"/>
   </p>
3. Apply controlled rotation portion of the QPE where <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U=e^{iAt}">. SInce U is unitary, <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U|b\rangle=e^{2{\pi}i\phi}|b\rangle"> . After the controlled U rotation,

   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\Psi_{3}\rangle=|b\rangle\frac{1}{\sqrt{2^{n}}}\sum_{j=0}^{2^n-1}e^{2{\pi}i{\phi}k}|k\rangle|0\rangle_{a}"/>
   </p>
4. Apply IQFT to clock qubits. More information for IQFT can be seen in the QFT repo. Resultant state can be written as
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\Psi_{4}\rangle=|b\rangle|N\phi\rangle|0\rangle_{a}"/>
   </p>

As mentioned before since <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U=e^{iAt}">, <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U|b\rangle=e^{i{\lambda_j}t}|u_j\rangle"> if <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}b=|u_j\rangle">. As <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U|b\rangle=e^{2{\pi}i\phi}|b\rangle">m you can equate <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}i\lambda_{j}t=2{\pi}i\phi"> to get

   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\Psi_{4}\rangle=b_j|u_j\rangle|\frac{N{\lambda_j}t}{2\pi}\rangle|0\rangle_{a}"/>
   </p>

   
Setting <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\tilde{\lambda_j}=\frac{N{\lambda_j}t}{2\pi}"> with general case <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}b=\sum_{j=0}^{2^{{n_b}-1}}|u_j\rangle">

   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\Psi_{4}\rangle=\sum_{j=0}^{2^{{n_b}-1}}b_j|u_j\rangle|~\lambda_j\rangle|0\rangle_{a}"/>
   </p>

5. Apply controlled RY roation on ancilla qubit.

   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\Psi_{5}\rangle=\sum_{j=0}^{2^{{n_b}-1}}b_j|u_j\rangle|\tilde{\lambda_j}\rangle(\sqrt{1-\frac{C^2}{\tilde{\lambda_j^2}}}|0\rangle_{a}+\frac{C}{\tilde{\lambda_j}}|1\rangle_a)"/>
   </p>

   
6. From now, I'll omit normalization factors for simplicity in reading. Keep making measurements till the ancilla qubit collapses to the <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|1\rangle"> state.

   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\Psi_{6}\rangle=\sum_{j=0}^{2^{{n_b}-1}}b_j|u_j\rangle|\tilde{\lambda_j}\rangle\frac{C}{\tilde{\lambda_j}}|1\rangle_a"/>
   </p>

Now the eigenvalue is in the denominator and one can see the |x> in the expression <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\lambda_i^{-1}b_{i}|u_i\rangle">

7,8. Because A is a diagonalizable matrix, we are able to apply the inverse Quantum Fourier Transform to each of the b_j components to get after applying IQFT and the controlled A rotation

   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\Psi_{8}\rangle=\sum_{j=0}^{2^n_b-1}\frac{b_{j}C}{\tilde{\lambda_j}}|u_j\rangle\sum_{y=0}^{2^n-1}|y\rangle|1\rangle_{a}"/>
   </p>  

   We can then substitute |x> in:
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\Psi_{8}\rangle=|x\rangle\sum_{y=0}^{2^n-1}|y\rangle|1\rangle_{a}"/>
   </p>


9. Apply the Hadamard gate to all the qubits in the c-register.
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\Psi_{8}\rangle=|x\rangle|0\rangle_{c}^{{\otimes}n}|1\rangle_{a}"/>
   </p>
   
   
10. Measure the b-register qubits.

## Gate Implementation
The novelty in the QED's benchmark is the unique implementations of the controlled A rotation and the controlled RY rotation.



- **RY rotation**:
For a given bitstring <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,s"/>, the quantum oracle <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,U_f"/> for the function <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}f(x)=s\cdot\,x\end{align*}\"> is implemented as a product of CNOT gates according to
<p align="center">
    <img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\,U_f=\bigotimes_{i:s_i=1}\text{CNOT}_{i,a}\">
</p>

where <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,\text{CNOT}_{i,a}\"> is a CNOT gate controlled on data qubit <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,i"/> and targeting the ancilla qubit.

<p align="center">
<img align=center src="../_doc/images/bernstein-vazirani/u_f.png" width="200"/>
</p>

## Circuit Methods
This benchmark contains two methods for generating the Bernstein-Vazirani circuit.

- **Method 1**: Generate the Bernstein-Vazirani circuit traditionally using the quantum circuit described in the [General Quantum Circuit](#general-quantum-circuit) section.
This method benchmarks the following circuit:

   <p align="center">
   <img align=center src="../_doc/images/bernstein-vazirani/bv1_qiskit_circ.png"  width="600" />
   </p>

- **Method 2**: Generate the Bernstein-Vazirani circuit using only two qubits and mid-circuit measurements. This method
mathematically is the same as method 1 but reduces the total number of qubits to two. This circuit is generated with the following 
steps:
  
1. Initialize the two qubits to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle|-\rangle">
2. Repeat the following subcircuit *n* times:
   * Measure the first qubit 
   * Reset the first qubit to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle">
   * For a given bitstring *s*, if <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s_n=1"> apply a hadamard to the first qubit, a CNOT gate where the first qubit is the control qubit, followed by another hadamard.
  
To learn more about this method, refer to 
Qiskit's implementation blog post [[4]](#references). This is currently only implemented in Qiskit. The following
is an example of the circuit benchmarked for this method: 
  
   <p align="center">
   <img align=center src="../_doc/images/bernstein-vazirani/bv2_qiskit_circ.png"  width="800" />
   </p>

Note for this method, the following plots (which are generated with each benchmark)
plots some metric versus the circuit width (number of qubits). For method 2, this circuit width is a virtual circuit width since the 
physical circuit width is two. For example, for the virtual circuit width = 4, this represents the corresponding
two qubit circuit used with mid circuit measurements to represent the quantum circuit with 4 qubits.

   <p align="center">
   <img align=center src="../_doc/images/bernstein-vazirani/bv_fidelity_width.png"  width="500" />
   </p>

## References

[1] Aram W. Harrow, Avinatan Hassidim, Seth Lloyd. (2008).
    Quantum algorithm for solving linear systems of equations.
    (https://arxiv.org/abs/0811.3171)

[2] Hector Jose Morell Jr, Anika Zaman, Hiu Yung Wong. (2023).
    A Step-by-Step HHL Algorithm Walkthrough to Enhance Understanding of Critical Quantum Computing Concepts.
    (https://arxiv.org/abs/2108.09004)
    
[3] Andrew M. Childs, Richard Cleve, Enrico Deotto, Edward Farhi, Sam Gutmann, Daniel A. Spielman. (2010).
    Exponential algorithmic speedup by quantum walk.
    (https://arxiv.org/abs/quant-ph/0209131)
    
[4] Mikko Mottonen, Juha J. Vartiainen, Ville Bergholm, Martti M. Salomaa. (2004).
    Transformation of quantum states using uniformly controlled rotations.
    (https://arxiv.org/abs/quant-ph/0407010)

[5] Yudong Cao, Anmer Daskin, Steven Frankel, Sabre Kais. (2011).
    Quantum Circuit Design for Solving Linear Systems of Equations
    (https://arxiv.org/abs/1110.2232v2)
    
[6] Yonghae Lee, Jaewoo Joo, and Soojoon Lee. (2019).
    Hybrid quantum linear equation algorithm and its experimental test on IBM Quantum Experience.
    (https://www.nature.com/articles/s41598-019-41324-9)

[7] Ana Martin, Ruben Ibarrondo, Mikel Sanz. (2022).
    Digital-analog co-design of the Harrow-Hassidim-Lloyd algorithm.
    (https://arxiv.org/abs/2207.13528)
    
