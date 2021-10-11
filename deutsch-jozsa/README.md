# Deutsch-Josza Benchmarking Test

The Deutsch-Josza Algorithm [[1]](#references) was the first example of a quantum algorithm that 
performs better than any classical algorithm. Despite the problem solved having little practical 
interest, it demonstrates how quantum computers are capable of outperforming classical computers with 
an exponential speedup.

## Problem outline

The goal of the Deutsch-Jozsa Algorithm is to gain information about a specific class of functions.
Suppose we are given access to a black-box Boolean function <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f:\{0,1\}^n\to\{0,1\}">, 
with the promise that <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f"> is either constant or balanced.
A constant function returns either all 0s or 1s, implying <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f(x)=c"> for all <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}x">.
A balanced function returns 0 for exactly half of all inputs and 1 for the other half. The goal is to determine whether 
the given function is balanced or constant.

## Benchmarking
The Deutsch-Jozsa algorithm is benchmarked by running a max of 2 circuits: both the constant and balanced oracle functions. For the constant oracle, it is randomly chosen if the constant is 0 or 1. The balanced oracle always has the circuit generated the same way for a given number of qubits. Each circuit is repeated a number of times denoted by `num_shots`. We then run the algorithm for numbers of qubits between `min_qubits` and `max_qubits`, inclusive. The test returns the averages of the circuit creation times, average execution times, fidelities, and circuit depths, like all of the other algorithms. For this algorithm's fidelity calculation, as we always have a single correct state, we compare the returned measurements against the distribution that has the single state with 100% probability using our [noise-normalized fidelity calculation](../_doc/POLARIZATION_FIDELITY.md).

The Deutsch-Jozsa algorithm seems to a natural algorithmic benchmark, as it is one of the most simple and well-known algorithms. However, the constant oracle poses a significant problems if the primary noise model is amplitude damping. We will see that this algorithm will always return an all zero state for the constant oracle, which makes it impossible to determine if this state was returned because of correct behavior or because the qubits relaxed to the  all zero state due to noise.

## Classical algorithm
Classically, to solve this problem requires applying <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f"> 
to a sequence of bitstrings until either two bitstrings <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}x_1"> and <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}x_2"> 
are found such that <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f(x_1)\ne{f}(x_2)">
or until <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2^{n-1}+1"> bitstrings have been tested. 
If <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f"> is constant, then in the worst case scenario
exactly half of the inputs plus one must be checked to confirm with 100% confidence <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f"> is constant.
Thus this classical algorithm takes <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}O(2^n)"> time.

## Quantum algorithm
Using a quantum algorithm, the problem can be solved with 100% confidence with only one call to <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f">
implying a runtime of <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}O(1)">.
This requires <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f"> to be implemented as a quantum
oracle. The quantum oracle for the function <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f"> 
is a unitary operator <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_f"> that acts on
<img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> data qubits and 1 ancilla qubit such that

<p align="center">
<img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_f|x\rangle|y\rangle=|x\rangle|y\oplus{f}(x)\rangle">,
</p>

where <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\oplus"> is addition modulo 2.

### General Quantum Circuit
The following circuit is the general quantum circuit for the Deutsch-Jozsa algorithm with <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> data qubits
and 1 ancilla qubit.

<p align="center">
<img align=center src="../_doc/images/deutsch-jozsa/deutsch_alg_circuit.png"  width="600" />
</p>

<p align="center">

*Fig 1. Diagram of general quantum circuit for Deutsch Algorithm [[2]](#references)*

</p>

References [[2]](#references) and [[3]](#references) both have overviews of the mathematical details of the 
algorithm, but the key points will be reproduced here.

### Algorithm Steps

The steps in the Deutsch-Josza algorithm are the following:

1. Initialize two quantum registers where <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> data qubits are initialized to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle"/> for the first register and the 
   one ancilla qubit is initialized to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|1\rangle"/> for the second register.
  
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_0\rangle=|0\rangle^{\otimes{n}}|1\rangle"/>
   </p>
   
2. Apply the Hadamard gate to all <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> qubits, creating an equal superposition state in the first register and
   <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|-\rangle=\frac{1}{\sqrt{2}}\big(|0\rangle-1\rangle\big)"> in the second register.
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle=H^{\otimes{n+1}}|\psi_0\rangle"/>
   </p>
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle=\frac{1}{\sqrt{2^{n+1}}}\sum_{x=0}^{2^n-1}|x\rangle(|0\rangle{-}|1\rangle)"/>
   </p>

3. Apply the oracle <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_f"> to the data and ancilla qubits. Recall
   <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_f|x\rangle|y\rangle=|x\rangle|y\oplus{f}(x)\rangle"> thus
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=U_f|\psi_1\rangle"/>
   </p>
   
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
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_3\rangle=\frac{1}{2^n}\sum_{x=0}^{2^n-1}\sum_{z=0}^{2^n-1}(-1)^{f(x)+x\cdot{z}}|z\rangle\frac{(|0\rangle{-}|1\rangle)}{\sqrt{2}}"/>
   </p>
   
5. Measure the data qubits. Note the probability of measuring <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle^{\otimes{n}}:P(|0\rangle^{\otimes{n}})=|\frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}(-1)^{f(x)}|^2"/>
   Thus if the measurement result is the all zero bitstring 00...0, then <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f"> is constant. Otherwise, f is balanced.

## Gate Implementation

### Constant Oracle

The constant oracle will either leave the qubits unchanged or apply a single x gate on the data qubit. 

### Balanced Oracle

The balanced oracle applies CNOTs to the ancilla qubit controlled by all the data qubits, with some modification of which input states return 0 and which return 1 with the shifting with the X gates at the start and end of the oracle.

<p align="center">
<img align=center src="../_doc/images/deutsch-jozsa/balanced_oracle.png"  width="600" />
</p>

## References

[1] David Deutsch and Richard Jozsa. (1992).
    Rapid Solution of Problems by Quantum Computation
    [`doi.org/10.1098/rspa.1992.0167`](https://royalsocietypublishing.org/doi/10.1098/rspa.1992.0167)

[2] Michael A. Nielsen and Isaac L. Chuang. (2011).
    Quantum Computation and Quantum Information: 10th Anniversary Edition (10th ed.). 
    Cambridge University Press, New York, NY, USA.

[3] Abraham Asfaw, Antonio Córcoles, Luciano Bello, Yael Ben-Haim, Mehdi Bozzo-Rey, Sergey Bravyi, Nicholas Bronn, Lauren Capelluto, Almudena Carrera Vazquez, Jack Ceroni, Richard Chen, Albert Frisch, Jay Gambetta, Shelly Garion, Leron Gil, Salvador De La Puente Gonzalez, Francis Harkins, Takashi Imamichi, Hwajung Kang, Amir h. Karamlou, Robert Loredo, David McKay, Antonio Mezzacapo, Zlatko Minev, Ramis Movassagh, Giacomo Nannicini, Paul Nation, Anna Phan, Marco Pistoia, Arthur Rattew, Joachim Schaefer, Javad Shabani, John Smolin, John Stenger, Kristan Temme, Madeleine Tod, Stephen Wood, and James Wootton. (2020).
    [`Deutsch-Jozsa Algorithm`](https://qiskit.org/textbook/ch-algorithms/deutsch-jozsa.html)