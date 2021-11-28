# Hidden Shift - Prototype Benchmark Program

The hidden shift algorithm [[1]](#references) is an algorithm that has previously 
been used to benchmark hardware [[2]](#references) [[3]](#references). 
While it is quite a contrived toy problem, it still provides an exponential speed up. It is a particularly interesting benchmark as it requires 
the same number of two qubit gates for a variety of hidden strings 

## Problem outline
Our problem is defined by first assuming we have some known Boolean function 
<img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}F(x)">. 
We also have some sort of black box, which implements the function with the input bits shifted by 
<img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s">, 
<img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}F(x\oplus{s})">. 
Then, the goal of the problem is to find <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s">. 
However, as the function is highly non-linear, finding the string <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s" >
is non-trivial.

## Benchmarking
The hidden shift algorithm is benchmarked by running `max_circuits` circuits for different hidden bitstrings `s_int`, chosen uniformly at random from <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\{0,1\}^N"> for <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}N"> qubits. Each circuit is repeated a number of times denoted by `num_shots`. We then run the algorithm circuit for numbers of qubits between `min_qubits` and `max_qubits`, inclusive. The test returns the averages of the circuit creation times, average execution times, fidelities, and circuit depths, like all of the other algorithms. For this algorithm's fidelity calculation, as we always have a single correct state, we compare the returned measurements against the distribution that has the single state with 100% probability using our [noise-normalized fidelity calculation](../_doc/POLARIZATION_FIDELITY.md).

## Classical algorithm
Classically, solving this problem would take <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\Theta(\sqrt{2^N})"> queries to identify the hidden shift [[2]](#references). Note that this is worse than the scaling from the Bernstein-Vazirani algorithm, as the function is non-linear.

## Quantum algorithm
We want the problem to be well formed for quantum hardware, so we choose a function which maps 
through a unitary as <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_f|x\rangle=f(x)|x\rangle">, 
such that <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f(x)\in\{-1,1\}">. 
This conversion from a Boolean function to a *sign function* is done as <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f(x)\equiv(-1)^{F(x)}">.

Our black box will implement <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}g(x)\equiv{f}(x\oplus{s})"> 
through the unitary <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_g">, 
which applies onto a state as <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_g|x\rangle=g(x)|x\rangle">

The algorithm also requires the ability to implement the dual bent function of 
<img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f(x)">, 
<img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\tilde{f}(x)">.
**We specifically choose a self-dual bent function of the Maiorana McFarland class [[4]](#references), 
such that** <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\tilde{f}(x)=f(x)">.

The function we choose is

<p align="center">
<img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}F(x)=\bigoplus_{i=0,2,\ldots}^{N-1}x_ix_{i+1}"> ,
</p>

remembering that <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f(x)=(-1)^{F(x)}">. This function is specifically chosen such that it can be implemented as

<p align="center">
<img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f(x)|x\rangle=\bigotimes_{i=0,2,\ldots}^{N-1}CZ_{i,i+1}|x\rangle">,
</p>

where <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}CZ_{i,i+1}"> 
indicates a controlled-Z gate on qubits <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}{i,i+1}">.

The quantum algorithm as described only requires a single query to the shifted function and a query to the original function. This means the number of queries is reduced to only <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}O(1)">, an exponential improvement.

### General Quantum Circuit 
<p align="center">
<img align=center src="../_doc/images/hidden-shift/HS_circ.png"  width="600" />
</p>

*Fig 1. Diagram of general efficient quantum circuit for Hidden Shift Algorithm [[2]](#references)*

### Algorithm Steps
The steps for the Hidden Shift algorithm are the following, reproduced from the original paper's [[2]](#references) author's response to a stack overflow question [[5]](#references):

1. Initiate all qubits to <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle^{\otimes{n}}">.
    <p align="center">
    <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_0\rangle=|0\rangle^{\otimes{n}}"/>
    </p>
2. Apply the Hadamard gate to all qubits.
   <p align="center">
   <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle=\frac{1}{\sqrt{2^{n}}}\sum_{x=0}^{2^n-1}|x\rangle"/>
   </p>
3. Apply the quantum oracle <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_g"> to all the qubits.
    <p align="center">
    <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=\frac{1}{\sqrt{2^{n}}}\sum_{x=0}^{2^n-1}g(x)|x\rangle"/>
    </p>
    <p align="center">
    <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=\frac{1}{\sqrt{2^{n}}}\sum_{x=0}^{2^n-1}f(x\oplus{s})|x\rangle"/>
    </p>
4. Apply the Hadamard gate to all qubits. The last step uses the fact that <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f(x)=\frac{1}{\sqrt{2^{n}}}\sum_{x=0}^{2^n-1}(-1)^{xy}f(x)"> because <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f(x)"> is a self-dual bent function.
    <p align="center">
    <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=\frac{1}{2^{n}}\sum_{y=0}^{2^n-1}\left(\sum_{x=0}^{2^n-1}(-1)^{xy}f(x\oplus{s})\right)|y\rangle"/>
    </p>
    <p align="center">
    <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=\frac{1}{2^{n}}\sum_{y=0}^{2^n-1}\left(\sum_{z=0}^{2^n-1}(-1)^{(z\oplus{s})y}f(z)\right)|y\rangle"/>
    </p>    
    <p align="center">
    <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=\frac{1}{2^{n}}\sum_{y=0}^{2^n-1}(-1)^{ys}\left(\sum_{z=0}^{2^n-1}(-1)^{zy}f(y)\right)|y\rangle"/>
    </p>
    <p align="center">
    <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_2\rangle=\frac{1}{\sqrt{2^{n}}}\sum_{y=0}^{2^n-1}(-1)^{ys}f(y)|y\rangle"/>
    </p>
5. Apply the quantum oracle <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_f"> to all the qubits.
    <p align="center">
    <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_3\rangle=\frac{1}{\sqrt{2^{n}}}\sum_{y=0}^{2^n-1}(-1)^{ys}f(y)f(y)|y\rangle"/>
    </p>
    <p align="center">
    <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_3\rangle=\frac{1}{\sqrt{2^{n}}}\sum_{y=0}^{2^n-1}(-1)^{ys}|y\rangle"/>
    </p>
6. Apply the Hadamard gate to all qubits
    <p align="center">
    <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_4\rangle=|s\rangle"/>
    </p>
7. Measure all the qubits to determine the hidden string <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s">

Review [[5]](#references) for a good explanation of how the math works, as this is an explanation from Martin Roetteler, the one who originally formulated the problem.

## Gate Implementation

### Implementation of <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}U_g">
The following are the subcircuits for a 6 qubit hidden shift algorithm with <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s=010110">.

To implement <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}g(x)\equiv{f}(x\oplus{s})"> 
as <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_g|x\rangle=g(x)|x\rangle">, we first apply <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}X"> gates on qubits in <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s"> provide the bitwise transformation. Then, the <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}CZ"> on neighboring pairs implements <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f(x)">. Finally, <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}X"> gates on qubits in <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}s"> provides the same bitwise transformation back to the input string
<p align="center">
<img align=center src="../_doc/images/hidden-shift/HS_Ug.png" height="400"/>
</p>

### Implementation of <img align="center" src="https://latex.codecogs.com/svg.latex?\pagecolor{white}U_f">
To implement <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\tilde{f}(x)">
as <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U_f|x\rangle=f(x)|x\rangle">, we apply <img align="center" src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}CZ"> on neighboring pairs.
<p align="center">
<img align=center src="../_doc/images/hidden-shift/HS_Uf.png" height="400"/>
</p>

## References

[1] Martin R̈otteler. (2008).
    Quantum algorithms for highly non-linear Boolean functions.
    [`arXiv:0811.3208`](https://arxiv.org/abs/0811.3208)

[2] N. M. Linke, D. Maslov, M. Roetteler, S. Debnath, C. Figgatt, K. A. Landsman, K. Wright, and C. Monroe. (2017).
    Experimental Comparison of Two Quantum Computing Architectures.
    [`arXiv:1702.01852`](https://arxiv.org/abs/1702.01852)

[3] K. Wright, K. M. Beck, S. Debnath, J. M. Amini, Y. Nam, N. Grzesiak, J. -S. Chen, N. C. Pisenti, M. Chmielewski, C. Collins, K. M. Hudek, J. Mizrahi, J. D. Wong-Campos, S. Allen, J. Apisdorf, P. Solomon, M. Williams, A. M. Ducore, A. Blinov, S. M. Kreikemeier, V. Chaplin, M. Keesan, C. Monroe, and J. Kim. (2019).
    Benchmarking an 11-qubit quantum computer.
    [`arXiv:1903.08181`](https://arxiv.org/abs/1903.08181)

[4] Claude Carlet, Lars Eirik Danielsen, Matthew G. Parker, and Patrick Sole. (2010)
    Self-dual bent functions
    [`10.1504/IJICOT.2010.032864`](https://www.inderscience.com/info/inarticle.php?artid=32864)

[5] Martin R̈otteler. (2019).
    Quantum Computing Stack Exchange Answer.
    [`Hidden shift problem as a benchmarking function`](https://quantumcomputing.stackexchange.com/a/5873)