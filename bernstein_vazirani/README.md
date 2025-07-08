# Bernstein-Vazirani - Prototype Benchmark Program

The Bernstein-Vazirani (BV) algorithm [[1]](#references) is an extension of the [Deustch-Jozsa algorithm](../deutsch-jozsa/), also implemented in this repository. 
The BV algorithm demonstrates a linear quantum speedup over its classical counterpart for analyzing a function. 

## Problem outline
Suppose we are given access to a black-box Boolean function defined as <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,f(x)=s\cdot\,x\quad(\text{mod}\;2),"/> 
where <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}s\in\{0,1\}^n\end{align*}\"/> 
is a hidden bit string of length <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,n"/>. 
Thus the function is guaranteed to return a bitwise product with a hidden string <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s">. The black-box implements this function by mapping <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,(x,y)\mapsto\big(x,y\oplus\,f(x)\big)\"/>. 
The goal is to determine the hidden string <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,s"/>.

## Benchmarking
The Bernstein-Vazirani algorithm is benchmarked by running `max_circuits` circuits for different hidden bitstrings `s_int`, chosen uniformly at random from <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\{0,1\}^N"> for <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}N"> qubits. Each circuit is repeated a number of times denoted by `num_shots`. We then run the algorithm for numbers of qubits between `min_qubits` and `max_qubits`, inclusive. The test returns the averages of the circuit creation times, average execution times, fidelities, and circuit depths, like all of the other algorithms. For this algorithm's fidelity calculation, as we always have a single correct state, we compare the returned measurements against the distribution that has the single state with 100% probability using our [noise-normalized fidelity calculation](../_doc/POLARIZATION_FIDELITY.md).

## Classical algorithm
Classically, to solve this problem requires appling <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,f"/> to each bitstring belonging to a basis for <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,\{0,1\}^n"/>.
For example, the standard basis is <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,\{(1,0,\dots,0),(0,1,0,\dots,0),\dots,(0,\dots,0,1)\} "/>. 
Each query from a basis bitstring reveals information about one of the bits in <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s"> due to the linearity of the function <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f">.
To determine the hidden string <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s"> requires <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,n"/> calls to the function <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,f"/>.
Thus this classical algorithm takes <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}O(n)"> time.

## Quantum algorithm
Using a quantum algorithm, the problem can be solved with only one call to the oracle, implying a runtime of <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}O(1)">.
This requires <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f"> to be implemented as a quantum
oracle. The quantum oracle for the function <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,f"/> 
is a unitary <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,U_f"/> that acts on a 
<img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,n"/> data qubits 
and 1 ancilla qubit such that

<p align="center">
    <img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\,U_f|x\rangle|y\rangle=|x\rangle|y\otimes\,f(x)\rangle.\">
</p>

The bitstring <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,s"/> determines the oracle <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\,U_f"/>.

### General Quantum Circuit
The following circuit is the general quantum circuit for the BV algorithm with <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> data qubits
and 1 ancilla qubit. The circuit is identical to the Deutsch-Jozsa circuit with the only difference 
being in the definition and implementation of the oracle <img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}U_f">.

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

The steps for the BV algorithm are the following, with the state after each step, <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_n\rangle">:

1. Initialize two quantum registers. The first register has <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> data qubits  initialized to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle"/> and the 
   second register has one ancilla qubit initialized to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|1\rangle"/>.
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_0\rangle=|0\rangle^{\otimes{n}}|1\rangle"/>
   </p>
   
2. Apply the Hadamard gate to all qubits, creating an equal superposition state <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\frac{1}{\sqrt{2^{n}}}\sum_{x=0}^{2^n-1}|x\rangle"/> in the first register and
   <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|-\rangle=\frac{1}{\sqrt{2}}\big(|0\rangle-1\rangle\big)"> in the second.
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle=\frac{1}{\sqrt{2^{n+1}}}\sum_{x=0}^{2^n-1}|x\rangle(|0\rangle{-}|1\rangle)"/>
   </p>
   

3. Apply the oracle <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_f"> to the data and ancilla qubits. Recall
   <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_f|x\rangle|y\rangle=|x\rangle|y\otimes{f}(x)\rangle"> thus
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=\frac{1}{\sqrt{2^{n+1}}}\sum_{x=0}^{2^n-1}|x\rangle(|{f(x)}\rangle{-}|1\oplus{f(x)}\rangle)"/>
   </p>
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=\frac{1}{\sqrt{2^{n+1}}}\sum_{x=0}^{2^n-1}(-1)^{f(x)}|x\rangle(|0\rangle{-}|1\rangle)"/>
   </p>
4. Apply the Hadamard gate to all <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> data qubits in the first register.
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_3\rangle=\frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^{f(x)}\bigg{[}\sum_{z=0}^{2^n-1}(-1)^{x\cdot{z}}\bigg{]}|z\rangle\frac{(|0\rangle{-}|1\rangle)}{\sqrt{2}}"/>
   </p>
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_3\rangle=\frac{1}{2^n}\sum_{x=0}^{2^n-1}\sum_{z=0}^{2^n-1}(-1)^{x\cdot{s}+x\cdot{z}}|z\rangle|-\rangle"/>
   </p>
  
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_3\rangle=|s\rangle|-\rangle"/>
   </p>
   
5. Measure the data qubits and the result is the bitstring <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s">.

## Gate Implementation
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

[1] Ethan Bernstein and Umesh Vazirani. (1997).
    Quantum Complexity Theory
    [`doi/10.1137/S0097539796300921`](https://epubs.siam.org/doi/10.1137/S0097539796300921)

[2] Michael A. Nielsen and Isaac L. Chuang. (2011).
    Quantum Computation and Quantum Information: 10th Anniversary Edition (10th ed.). 
    Cambridge University Press, New York, NY, USA.

[3] Abraham Asfaw, Antonio Córcoles, Luciano Bello, Yael Ben-Haim, Mehdi Bozzo-Rey, Sergey Bravyi, Nicholas Bronn, Lauren Capelluto, Almudena Carrera Vazquez, Jack Ceroni, Richard Chen, Albert Frisch, Jay Gambetta, Shelly Garion, Leron Gil, Salvador De La Puente Gonzalez, Francis Harkins, Takashi Imamichi, Hwajung Kang, Amir h. Karamlou, Robert Loredo, David McKay, Antonio Mezzacapo, Zlatko Minev, Ramis Movassagh, Giacomo Nannicini, Paul Nation, Anna Phan, Marco Pistoia, Arthur Rattew, Joachim Schaefer, Javad Shabani, John Smolin, John Stenger, Kristan Temme, Madeleine Tod, Stephen Wood, and James Wootton. (2020).
    [`Bernstein-Vazirani Algorithm`](https://qiskit.org/textbook/ch-algorithms/bernstein-vazirani.html)

[4] Paul Nation and Blake Johnson. (2021).
    How to measure and reset a qubit in the middle of circuit execution.
    [`Qiskit Blog: Mid Circuit Measurements`](https://www.ibm.com/blogs/research/2021/02/quantum-mid-circuit-measurement/)