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
   <img align=center src="../_doc/images/hhl/hhl_circuit.png"  width="600" />
</p>

*Fig 1. Diagram of general quantum circuit for Bernstein-Vazirani Algorithm [[2]](#references)*

References [[2]](#references) and [[3]](#references) both have overviews of the mathematical details of the 
algorithm, but the key points will be reproduced here.


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

For the given bitstring with zeroed ancilla qubit, <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\tilde{\lambda_j}\rangle|0\rangle_{a}">, apply a controlled RY gate on the ancilla qubit to get the state, <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\tilde{\lambda_j}\rangle(\sqrt{1-\frac{C^2}{\tilde{\lambda_j^2}}}|0\rangle_{a}+\frac{C}{\tilde{\lambda_j}}|1\rangle_a)">.

This rotation process can be implemented by setting the angle of rotation: 

   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta=2\arcsin(\frac{C}{\tilde{\lambda_{j}}})"/>
   </p>

   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}RY(\theta)=\cos(\frac{\theta}{2})|0\rangle_{a}+\sin(\frac{\theta}{2})|1\rangle_{a}"/>
   </p>

Controlled Rotations can be implemented like this,
   <p align="center">
   <img align=center src="../_doc/images/hhl/old_ry.png" width="600"/>
   </p>

However, it can also be implemented using single CNOT and rotation gates by changing the angles as can be seen below
   <p align="center">
   <img align=center src="../_doc/images/hhl/new_ry.png" width="600"/>
   </p>

To do this, the user needs to apply the following matrix to the <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\alpha"> angles. 


   <p align="center">
   <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}\begin{bmatrix}\theta_{1}\\\theta_{2}\\\vdots\\{\theta_{2^k}}\end{bmatrix}&=M\begin{bmatrix}\alpha_{1}\\\alpha_{2}\\\vdots\\{\alpha_{m}}\end{bmatrix}\end{align*}\">
   </p>

   where <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}M_{ij}=2^{-k}(-1)^{{b_{j-1}}\cdot{g_{i-1}}}\end{align*}\">. <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}g_m\end{align*}\"> stands for the binary reflected gray code representation of the integer m. How the gray code works can be read here https://en.wikipedia.org/wiki/Gray_code.

   
- **e^{iAt}**:

The controlled rotation that's implemented in this benchmark is influenced by the Quantum Walk circuit from the paper, "Exponential algorithmic speedup by quantum walk." In this paper, a Hamiltonian is implemented that calculates the connected vertex in a graph given an input vertex and a colored edge.


   <p align="center">
   <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}H=V_{c}TV_{c}\\T|a,b,0\rangle=|b,a,0\rangle\\T|a,b,1\rangle=0\\V_c|a,b,r\rangle=|a,b\otimes{v_c{a}},r\otimes{f_c(a)}\rangle\end{align*}\">
   </p>

T is written in the following way:

   <p align="center">
   <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}T=(\bigotimes^{2n}_{l=1}S^{l,2n+1})\otimes|0\rangle\langle0|\\S|z_{1}z_{2}\rangle=|z_{2}z_{1}\rangle\end{align*}\">
   </p>
   
   One can note the eigenvalues of the swap operator as follows

   <p align="center">
   <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}S\frac{1}{\sqrt{2}}(|01\rangle+|10\rangle)=(+1)\frac{1}{\sqrt{2}}(|01\rangle+|10\rangle)\\S\frac{1}{\sqrt{2}}(|01\rangle-|10\rangle)=(-1)\frac{1}{\sqrt{2}}(|01\rangle-|10\rangle)\end{align*}\">
   </p>

It can be seen that the swap gate can be diagonalized using the properties of the eigenvectors and eigenvalues of the S gate by using the W operator which operates as follows. It's also worth noting that W is unitary.

   <p align="center">
   <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}W|00\rangle=|00\rangle\\W\frac{1}{\sqrt{2}}(|01\rangle+|10\rangle)=|01\rangle\\W\frac{1}{\sqrt{2}}(|01\rangle-|10\rangle)=|10\rangle\\W|11\rangle=|11\rangle\end{align*}\">
   </p>

As such, applying the W gate to the vectors allows one to use the diagonalized version of the SWAP gate for computations. The circuit <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}e^{-iTt}"> can be seen below.

Now one can observe the action of the Hamiltonian operator:

   <p align="center">
   <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}H|a,0,0\rangle=\sum_{c}V_{c}T|a,v_{c}(a),f_{c}(a)\rangle\\=\sum_{c}\delta_{0,f_{c}}(a)V_{c}|v_{c}(a),a,0\rangle\\=\sum_{c:v_{c}(a)\in{G}}|v_{c}(a),a\oplus{v_c(v_c(a))},f_c(v_c(a))\rangle\end{align*}\">
   </p>

   
<img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}v_cv_c(a)=a"> because it returns to the same vertex on the graph along the edge and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f_c(a)=0"> because a is a vertex on the graph. So


   <p align="center">
   <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}H|a,0,0\rangle=\sum_{c:v_{c}(a)\in{G}}|v_c(a),0,0\rangle\end{align*}\">
   </p>


So far, the discussion has been about quantum graphs but now is the time to see how it relates to the HHL algorithm.
We start of by defining the V gate, mainly the $v_c(a)$ function. The $v_c(i)$ returns the column index of the cth non-zero element on the ith column.


For example in the below matrix:

<p align="center">
   <img align=center src="../_doc/images/hhl/sparse_matrix.png"  width="600" />
</p>


We want to be able to generate the eigenvectors of a from b. From this 2 sparse matrix, one can note the following 2 eigenvectors.

   <p align="center">
   <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{align*}|{\lambda=\frac{1}{2}}\rangle=\begin{pmatrix}1\\0\\0\\0\\0\\1\\0\\0\end{pmatrix},\|{\lambda=1}\rangle=\begin{pmatrix}-1\\0\\0\\0\\0\\1\\0\\0\end{pmatrix}\end{align*}\">
   </p>

To get the first eigenvector all you have to do is apply the control phase gate before doing the rest of the Hamiltonian simulation with the following expression

   <p align="center">
   <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{verbatim}qc.cp(-t*(diag\_el+sign*off\_diag\_el),control,anc)\end{verbatim}\">
   </p>

In this case, it will be 0.75+(-0.25) = 0.5

And replacing the <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}e^{-iZt}"> expression in the quantum walk algorithm,


   <p align="center">
   <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\small\begin{verbatim}qc.cp((sign*2*t*off\_diag\_el),control,anc)\end{verbatim}\">
   </p>


This will be subtracted from to the original value so, 0.5-(2*-0.25) = 1.0. 

The sign is determined by if there are an even or odd amount of 1s are in the column index. For example index 5 has 2 1s. As such, the sign would be positive 1. This is to determine the sign of the eigenvalue that you would get from applying the Z gate.



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
    
