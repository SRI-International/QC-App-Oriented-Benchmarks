# Grover's Algorithm - Prototype Benchmark Program

Grover's algorithm [[1]](#references), also referred to as the quantum search algorithm, is one of the most well known quantum algorithms due to its large amount of applications as well as its quadratic runtime speedup from any known classical algorithm. It is a special case of the more general [Amplitude Estimation](../amplitude-estimation/) algorithm.

## Problem outline
This algorithm solves unstructured search problems, such as where we have a large, un-ordered list of items and have some "correct" item(s) we are looking for. We can check if each item is correct through an *oracle*, where for Grover's algorithm, the oracle gives the correct state a negative phase to distinguish it from other states. The methods we will describe here work for an arbitrary number of correct items, but in this benchmark we only have a single correct item we are looking for.

## Benchmarking
Grover's algorithm is benchmarked by running `max_circuits` circuits for different correct items `s_int`, chosen uniformly at random from <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\{0,1\}^N"> for <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}N"> qubits. Each circuit is repeated a number of times denoted by `num_shots`. We then run the algorithm circuit for numbers of qubits between `min_qubits` and `max_qubits`, inclusive. The test returns the averages of the circuit creation times, average execution times, fidelities, and circuit depths, like all of the other algorithms. For this algorithm's fidelity calculation, the algorithm can return a distribution when the number of qubits is small, so we compare against the analytical expected distribution using our [noise-normalized fidelity calculation](../_doc/POLARIZATION_FIDELITY.md).

## Classical Algorithm
For a classical algorithm, the only way to solve the problem is to test each potential solution one at a time. For a list of <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}N"> items (and a single solution), this would take <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}N/2"> steps on average, and <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}N"> steps in the worst case scenario. 

## Quantum Algorithm
However, for this quantum algorithm, we see that it will take us <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\sim\sqrt{N}"> rotations to reach a high likelihood of success. Reference [[2]](#references) has an excellent overview of the algorithm, but we will also reproduce the key points for the algorithm here.

The primary intuition behind this algorithm can come from the geometric picture of sequentially applying reflections across the incorrect solutions and across the uniform superposition state. 

---

<p align="center">
<img align="center" src="../_doc/images/grovers/grover_step1.jpg"  width="600" />
</p>

This image and the following two come from [[2]](#references). We initially start in a uniform superposition of all <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}N"> states,

<p align="center">
<img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|s\rangle\equiv\frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle=\sin{\theta}|w\rangle+\cos{\theta}|s'\rangle">,
</p>

where <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|w\rangle"> is the correct solution and <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|s'\rangle"> contains the rest of the states in <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|s\rangle">. Note: in this explanation, <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}N"> is the total number of states, not the total number of qubits. For example, for 3 qubits, we would have <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}N=2^3=8"> items. By doing some math with these definitions, we can see that <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta=\arcsin\langle{s}|w\rangle=\arcsin(1/\sqrt{N})">. 

---

<p align="center">
<img align="center" src="../_doc/images/grovers/grover_step2.jpg"  width="600" />
</p>

We then apply the oracle, which by defintion will add a phase of <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}-1"> to the state <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|w\rangle">, which is stated as a unitary operator as <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_{w}=1-2|w\rangle\langle{w}|">. This definition does what we expect, as any state that is orthogonal to <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|w\rangle"> will only have the identity operator applied to it, while the state <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|w\rangle"> itself will have a non-zero result from both terms, leading to <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_w|w\rangle=-|w\rangle">. Geometrically, this is a reflection around <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|s'\rangle">. 

---

<p align="center">
<img align="center" src="../_doc/images/grovers/grover_step3.jpg"  width="600" />
</p>

We will then apply <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_s=2|s\rangle\langle{s}|-1">, which will give a reflection across <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|s\rangle">. (Note: we will actually apply <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}-U_s">, but there is no difference as this negative sign is a global phase which is unobservable). We can then successively apply <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_sU_w"> to bring the solution as close to <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|w\rangle"> as possible. By using the definitions of these operators and some matrix manipulation, we see that <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}(U_sU_w)^t|s\rangle=\sin[(2t+1)\theta]|w\rangle+\cos[(2t+1)\theta]|s'\rangle"> for <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}t"> applications of the oracle and diffuser. From this, we can see that the probability of measuring <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|w\rangle"> is maximized when <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}(2t+1)\theta\approx\pi/2">, which will only take <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}r\sim\sqrt{N}"> applications of these operators. This is much better than the <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\sim{N}"> applications with the classical algorithm!

### General Quantum Circuit

<p align="center">
<img align="center" src="../_doc/images/grovers/grovers_circuit.png"  width="800" />
</p>

Circuit diagram for Grover's algorithm with 4 qubits.

### Algorithm Steps

The steps for Grover's algorithm are the following:

1. Initialize all the qubits to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle"/>.
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_0\rangle=|0\rangle^{\otimes{n}}"/>
   </p>
   
2. Hadamard all qubits to create an equal superposition state.
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle=H^{\otimes{n}}|\psi_0\rangle"/>
   </p>
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle=\frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle"/>
   </p>
   
3. Apply the two oracle and diffuser operators <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}r\sim\sqrt{N}"> times.
   
   <p align="center">
   <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=(U_sU_w)^t|\psi_1\rangle">
   </p>
   <p align="center">
   <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=\sin[(2t+1)\theta]|w\rangle+\cos[(2t+1)\theta]|s'\rangle">
   </p>
   
4. Measure qubits and the solution string <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|w\rangle"/> will appear with high likelihood, assuming an error-free quantum computer

## Gate Implementation

### Grover Oracle <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}U_{w}">

The following is the quantum circuit for the Grover oracle for 4 qubits with <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s=1100">. The oracle applies a phase just to this correct state. Note that the barrier is solely for better visualization. To create this oracle, we start by applying <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}X"> gates on just the qubits which are not <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}1"> in the solution <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|w\rangle">. This ensures that the correct bitstring goes to the state <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|11\ldots\rangle">. We then apply a multi-<img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}CZ"> on all qubits to add a phase of <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}-1"> to only <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|11\ldots\rangle">. We then apply the same <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}X"> gates to make sure that only <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|w\rangle"> has aquired the relative phase of <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}-1">.

<p align="center">
<img align="center" src="../_doc/images/grovers/oracle.png"  width="400" />
</p>

### Grover Diffuser <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}U_{s}">

The following is the quantum circuit for the Diffuser for 4 qubits. As implemented, this will only provide a phase to the state <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|s\rangle">. Note that the barrier is solely for better visualization. We start by applying a Hadamard gate all qubits to take <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|s\rangle\rightarrow|00\ldots\rangle">. We then add a phase of <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}-1"> to only <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|00\ldots\rangle"> by applying <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}X"> gates on every qubit, applying a multi-<img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}CZ">, and applying the <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}X"> gates again. We finally Hadamard all qubits again to take <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}-|00\ldots\rangle\rightarrow-|s\rangle">.

<p align="center">
<img align="center" src="../_doc/images/grovers/diffuser.png"  width="400" />
</p>

---

## References


[1] Lov K. Grover. (1996)
   A fast quantum mechanical algorithm for database search.
    [`arXiv:quant-ph/9605043`](https://arxiv.org/abs/quant-ph/9605043v3)

[2] Abraham Asfaw, Antonio Córcoles, Luciano Bello, Yael Ben-Haim, Mehdi Bozzo-Rey, Sergey Bravyi, Nicholas Bronn, Lauren Capelluto, Almudena Carrera Vazquez, Jack Ceroni, Richard Chen, Albert Frisch, Jay Gambetta, Shelly Garion, Leron Gil, Salvador De La Puente Gonzalez, Francis Harkins, Takashi Imamichi, Hwajung Kang, Amir h. Karamlou, Robert Loredo, David McKay, Antonio Mezzacapo, Zlatko Minev, Ramis Movassagh, Giacomo Nannicini, Paul Nation, Anna Phan, Marco Pistoia, Arthur Rattew, Joachim Schaefer, Javad Shabani, John Smolin, John Stenger, Kristan Temme, Madeleine Tod, Stephen Wood, and James Wootton. (2020).
    [`Grover's Algorithm`](https://qiskit.org/textbook/ch-algorithms/grover.html)

[//]: # (Link to location which has the entire qiskit-textbook bibtex file: https://github.com/qiskit-community/qiskit-textbook/blob/main/content/qiskit-textbook.bib)

[3] Michael A. Nielsen and Isaac L. Chuang. (2011).
    Quantum Computation and Quantum Information: 10th Anniversary Edition (10th ed.). 
    Cambridge University Press, New York, NY, USA.