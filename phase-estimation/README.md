# Quantum Phase Estimation - Prototype Benchmark Program

Phase estimation [[1]](#references) is one of the most important quantum subroutines, fundamental to most quantum 
algorithms such as [Shor's Order Finding](../shors/) and the HHL algorithm for solving linear systems of equations. 
The algorithm utilizes the [Quantum Fourier Transform](../quantum-fourier-transform/)
as a key procedure, which is described and implemented as a benchmark in this repository. 

## Problem outline
The goal of the Quantum Phase Estimation (QPE) algorithm is to estimate the eigenvalues of a unitary operator.
Specifically, QPE estimates a value <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta\">
for a given unitary operator <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathit{U}"/> defined as 

<p align="center">
<img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathit{U}|\psi\rangle=e^{2{\pi}{i}\theta}|\psi\rangle"/>.
</p>

where <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi\rangle"/> is an
eigenvector with <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}e^{2{\pi}i\theta}"/> 
as the corresponding eigenvalue for <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathit{U}"/>.

## Benchmarking

The Phase Estimation algorithm is benchmarked by running `max_circuits` circuits for random <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta"/> values of the form <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\frac{n}{2^k}"/> where <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}k"/> represents the total number of qubits in the counting register and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> is an integer between <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}0"/> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2^k-1"/>. Each circuit is repeated a number of times denoted by `num_shots`. We then run the algorithm circuit for numbers of qubits between `min_qubits` and `max_qubits`, inclusive. The test returns the averages of the circuit creation times, average execution times, fidelities, and circuit depths, like all of the other algorithms. For this algorithm's fidelity calculation, because we chose <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta"/> to never require more precision than the number of counting qubits can provide, we compare the returned measurements against the distribution that has the single state with 100% probability using our [noise-normalized fidelity calculation](../_doc/POLARIZATION_FIDELITY.md).

## Classical algorithm
Classically determining the eigenvalues of a <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n">-by-<img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> operator <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathit{U}"/> involves performing eigendecompostion which 
reduces down to basic matrix multiplication. For an arbitary operator, eigenvalue decomposition involves matrix 
multiplication which has <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathit{O}(n^3)"/> time.
However, using the Coppersmith and Winograd algorithm, eigenvalue decomposition for a unitary operator has 
<img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathit{O}(n^w)"/> time where 
<img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\omega=2.376"/> [[2]](#references). This
produces a symbolic determinant which is used to recover all the eigenvalues.
Then acquiring <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}m"> bits of the eigenvalue has complexity of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathit{O}(n^{w+1}{m})"/>  since
the operations take place over <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> degree polynomials with coefficients in <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}m"> bits [[2]](#references). With more 
complex methods, to approximate the eigenvalue within <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2^{-\epsilon}"/> has
<img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathit{O}(n^2\text{log}n+n^2\text{log}^2{n}\text{log}\epsilon)"/> time [[3]](#references).

## Quantum algorithm

The quantum algorithm has a runtime of about <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}O(m^2)"/> 
which is dependent solely on the amount of bits <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}m"> required to represent the eigenvalue. Since the complexity time
of the classical algorithm is dependent on both the order <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> of the unitary matrix and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}m">, determining whether the quantum
or classical algorithm performs better is dependent on these relevant variables. However, the real use of 
quantum phase estimate comes from the fact that many other interesting problems can be reduced to phase 
estimation, leading to more dramatic improvements in complexity time for other algorithms. The general
quantum phase estimation algorithm will be summarized below along with its implementation in this benchmark.


### General Quantum Circuit
The following circuit is the general quantum circuit for quantum phase estimation with <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}t"> qubits in the counting register and the bottom qubits representing the <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi\rangle"/> 
state as the data register.

<p align="center">
<img align=center src="../_doc/images/phase-estimation/qpe_tex_qz.png"  width="600" />
</p>

*Fig 1. Diagram of general quantum circuit for Phase Estimation Algorithm [[4]](#references)*


When <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta"/> can be represented by a <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}t"> bit binary expansion, QPE solves for <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta"/> exactly. 
In other cases, it can be shown the output measurement is an approximation of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta"/> accurate to 
<img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}t-[\text{log}(2+(2\epsilon)^{-1})]"/> bits with probability of success at least 
<img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}1-\epsilon"/>. Similarly stated,
to obtain <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta"/> accurate to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}m"> bits with probability of success at least <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}1-\epsilon"/>,
choose <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}t=m+\text{log}(2+(2\epsilon)^{-1})"/> counting qubits.

This quantum algorithm has a runtime of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}O(t^2)"/> 
operations and one call to the controlled-<img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U^j"/> 
operator that succeeds with probability of at least <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}1-\epsilon"/>. 
References [[4]](#references) and [[5]](#references) both have overviews of the mathematical details of the 
algorithm, but the key points will be reproduced here.

### Algorithm Steps
The steps for phase estimation are the following: 

1. The procedure uses two registers. We initialize <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}t"> qubits to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle"/> which represent the counting register. This register
   will store the value <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2^t\theta"/> by the end of the algorithm.
   The second register contains the state <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi\rangle"/> with as many qubits
   as necessary to store the state:
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_0\rangle=|0\rangle^{\otimes{t}}|\psi\rangle"/>
   </p>
   
2. Apply a <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}t">-bit Hadamard gate operation to the counting register:
    
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle=H^{\otimes{t}}|\psi_0\rangle"/>    
   </p>
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle=\frac{1}{2^{\frac{t}{2}}}(|0\rangle+|1\rangle)^{\otimes{t}}|\psi\rangle"/>
   </p>
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle=\frac{1}{2^{\frac{t}{2}}}\sum_{j=0}^{2^t-1}|j\rangle|\psi\rangle"/>
    </p>

   where <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}j"> denotes the integer representation of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}t">-bit binary numbers.

3. Next, the <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}t"> controlled unitary operators are applied on the data register as shown in the circuit above.
   Recall since <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U"> is a unitary operator with eigenvector <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi\rangle"/> such that
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U|\psi\rangle=e^{2\pi{i}\theta}|\psi\rangle"/>, then
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=\frac{1}{2^{\frac{t}{2}}}\sum_{j=0}^{2^t-1}|j\rangle{U}^j|\psi\rangle"/>
   </p>
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=\frac{1}{2^{\frac{t}{2}}}\sum_{j=0}^{2^t-1}{e}^{2\pi{i}{j}\theta}|j\rangle{|}\psi\rangle"/>
   </p>
   
4. The previous expression is the same derived expression for the quantum Fourier transform on 
   a <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}t">-qubit input state where <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}x"> is replaced by <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2^t\theta">. 
   Review the [Quantum Fourier Transform](../quantum-fourier-transform/) benchmark for the full derivation.
   Thus applying the inverse QFT to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle"> retrieves 
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2^t\theta"> in the counting register:
  
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_3\rangle=QFT^{-1}|\psi_2\rangle">   
   </p>
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_3\rangle=\frac{1}{2^t}\sum_{x=0}^{2^t-1}\sum_{j=0}^{2^t-1}e^{-\frac{2\pi{i}j}{2^t}(x-2^t\theta)}|x\rangle|\psi\rangle">
   </p>
   
5. Finally, measure the counting register in the computational basis. The above expression peaks 
   near <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}x=2^t\theta"> and when <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2^t\theta"> 
   is an integer then the measured output becomes the following which includes the phase exactly in the counting register:
   
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_4\rangle=|2^t\theta\rangle|\psi\rangle">
   </p>
   
   When <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2^t\theta"> is not an integer it can be shown the above expression peaks near <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2^t\theta">
   with probability better than <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\frac{4}{\pi^2}\approx40\%"> [[4]](#references).

## Gate Implementation
In this benchmark, we chose to only have a single qubit in the data register, using CPHASE gates as our C-U. For these gates, the state <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|1\rangle"> has eigenvalue <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}e^{2\pi{i}\theta}">, with <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta"> tunable.

This benchmark's gate implementation deviates slightly from way we've described the algorithm above in one important way: our C-U gates start with the most significant qubit in the counting register and work towards the least significant qubit. This can bee seen in the below image of the circuit we apply:

<p align="center">
<img align=center src="../_doc/images/phase-estimation/qpe_circuit.png"  width="600" />
</p>

This is because the benchmark uses this repository's internal inverse quantum Fourier transform gate. Like we mention in the [Quantum Fourier Transform benchmark](../quantum-fourier-transform/), we have removed the cannonical swaps to improve performance. This means that our qubits are instead ordered in the reverse way by the inverse QFT gate, which is why our applications of C-U work the other way around.

## References

[1] Hamed Mohammadbagherpoor, Young-Hyun Oh, Patrick Dreher, Anand Singh, Xianqing Yu, Andy J. Rindos. (2019).
    An Improved Implementation Approach for Quantum Phase Estimation on Quantum Computers.
    [`arXiv:1910.11696`](https://arxiv.org/abs/1910.11696v1)

[2] Josh Alman, Virginia Vassilevska Williams. (2020).
    A Refined Laser Method and Faster Matrix Multiplication.
    [`arXiv:2010.05846`](https://arxiv.org/abs/2010.05846)

[3] Victor Y. Pan, Zhao Q. Chen. (1999).
    The Complexity of the Matrix Eigenproblem.
    [`doi.org/10.1145/301250.301389`](https://dl.acm.org/doi/abs/10.1145/301250.301389)

[4] Michael A. Nielsen and Isaac L. Chuang. (2011).
    Quantum Computation and Quantum Information: 10th Anniversary Edition (10th ed.). 
    Cambridge University Press, New York, NY, USA.

[5] Abraham Asfaw, Antonio Córcoles, Luciano Bello, Yael Ben-Haim, Mehdi Bozzo-Rey, Sergey Bravyi, Nicholas Bronn, Lauren Capelluto, Almudena Carrera Vazquez, Jack Ceroni, Richard Chen, Albert Frisch, Jay Gambetta, Shelly Garion, Leron Gil, Salvador De La Puente Gonzalez, Francis Harkins, Takashi Imamichi, Hwajung Kang, Amir h. Karamlou, Robert Loredo, David McKay, Antonio Mezzacapo, Zlatko Minev, Ramis Movassagh, Giacomo Nannicini, Paul Nation, Anna Phan, Marco Pistoia, Arthur Rattew, Joachim Schaefer, Javad Shabani, John Smolin, John Stenger, Kristan Temme, Madeleine Tod, Stephen Wood, and James Wootton. (2020).
    [`Quantum Phase Estimation`](https://qiskit.org/textbook/ch-algorithms/quantum-phase-estimation.html#1.) 

