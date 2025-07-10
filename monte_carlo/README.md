# Monte Carlo - Prototype Benchmark Program

The Quantum Monte Carlo Sampling Algorithm presented here is an application of the [Quantum Amplitude Estimation](../amplitude-estimation) routine we implemented as benchmark. We notice that this quantum algorithm performs quadratically better than any known classical algorithm. We follow the process outlined in [[1]](#references).

## Problem outline

This algorithm aims to estimate the expected value of some function of a random variable <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}E[f(X)]"> given quantum oracle access to the probability distribution of the variable <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,p(X)"> and the function of interest <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,f(\cdot)">. For a distribution <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,p(X)"> which is potentially not analytically known or otherwise difficult to calculate directly, it might not be feasible to get an exact result for the expectation value of the function applied to the distribution. Both the classical and quantum versions of the algorithm utilize methods to approximate this expectation value by utilizing calls to the distribution.

## Benchmarking

The Monte Carlo algorithm is benchmarked in two different ways: for method 1, we run `max_circuits` circuits for random Gaussian distributions with mean <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mu"/> values of the form <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\frac{n}{2^k}"/> where <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}k"/> represents the total number of qubits in register 1 and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> is an integer between <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}0"/> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2^k-1"/> and for method 2, we run only a single circuit with the specific distribution. Each circuit is repeated a number of times denoted by `num_shots`. We then run the algorithm circuit for numbers of qubits between `min_qubits` and `max_qubits`, inclusive. The test returns the averages of the circuit creation times, average execution times, fidelities, and circuit depths, like all of the other algorithms. For this algorithm's fidelity calculation, the algorithm can return a distribution when the number of qubits in register 1 is finite, so we compare against the analytical expected distribution using our [noise-normalized fidelity calculation](../_doc/POLARIZATION_FIDELITY.md).

## Classical algorithm

The classical Monte Carlo process, in general, defines a random process which gives values according to the distribution, and then takes an average over the function evaluated at these values. This can be used to calculate empirical estimates for quantities of interest; for example, repeatedly drawing five cards at random from a standard deck of playing cards to approximate the theoretical probabilities of specific hands in poker. Such sampling methods are often referred to as Monte Carlo, and can also play a key role in non-deterministic algorithms. 

The process is usually such that the sample average is an unbiased estimator of the expected value of the underlying distribution. Then, due to the Central Limit Theorem, the variance of the mean of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,M"> samples collected in this way will scale as <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,O(M^{-1/2})\,">. We use the scaling of the variance of the mean as a gauge for comparing the classical and quantum algorithms, as a better scaling would require less samples to achieve the same uncertainty in the estimate.

The scaling with the size of the sample space <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,N"> can depend on the choice of sampling implementation, but it often scales as <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}O(N)">.

## Quantum algorithm

With the quantum algorithm, our probability distribution and function are instead coded into calls to a quantum oracle. We can then use the ideas of Quantum Amplitude Estimation to return an estimate of the exectation value which has variance decreasing quadratically faster than ordinary classical sampling.

Using <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,O(M)"> calls to the oracles, the variance scales in the best case as <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,O(M^{-1})">, and in the most limited case with bounded error as <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,O(M^{-2/3})"> [[1]](#references). Even this worst case improves convergence compared to classical Monte Carlo sampling estimation, which scales as <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,O(M^{-1/2})"> for <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,M"> samples. 

To encode information about our distribution and function we define the following operators which will be called as oracles; <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,\mathcal{R},\mathcal{F}"> defined on <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,n+1"> qubits such that:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}\large\mathcal{R}|0\rangle_{n}|0\rangle=\sum_i\sqrt{p(X=i)}|i\rangle_n|0\rangle\end{align*}\">
</p>
   
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}\large\mathcal{F}|i\rangle_n|0\rangle=|i\rangle_n\big(\sqrt{1-f(i)}|0\rangle+\sqrt{f(i)}|1\rangle\big)\end{align*}">
</p>

In this context, <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,\mathcal{R}"> prepares the zero state into the all real-valued superposition according to the desired measurement probabilities, while <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,\mathcal{F}"> encodes the value of the function <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,f(i)"> of the state of the <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,n"> qubit register <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,|i\rangle"> into the objective qubit at the end. The product of these two is:

<p align="center">
<img
src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}\large\mathcal{F}\mathcal{R}|0\rangle_{n}|0\rangle=\sum_i|i\rangle_n\big(\sqrt{p(i)}\sqrt{1-f(i)}|0\rangle+\sqrt{p(i)}\sqrt{f(i)}|1\rangle\big)\end{align*}\">
</p>

If we remember that the expection value we are looking for is defined as 

<p align="center">
<img
src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}\large{E}[f(X)]\equiv\sum_ip(i)f(i)\end{align*}\">
</p>

we can see that if there is an efficient way of estimating the amplitude on the objective qubit in the <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,|1\rangle"> state, there is an efficient way to estimate the expected value of the function on the random variable <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\,E[f(X)]\,."> As in quantum amplitude estimation, we aim to estimate <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a"> from the application

<p align="center">
<img
src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}\large\mathcal{A}|0\rangle_{n+1}=\sqrt{1-a}|\psi_0\rangle|0\rangle+\sqrt{a}|\psi_1\rangle|1\rangle\end{align*}\">
</p>

we can clearly see that using <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}=\mathcal{F}\mathcal{R}"> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a=E[f(X)]\">, Amplitude Estimation will allow us to find the value <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a">. For information about this subroutine, please refer to the [Amplitude Estimation benchmark](../amplitude-estimation/), as our implementation of QAE in this benchmark is identical.

### Correct Distribution

The fidelity calculation we use in this benchmark is significantly more involved than in some of the other benchmarks. The amplitude estimation algorithm has granularity in the number of answers it can return as a result of the finite number of counting qubits (the number of qubits in the first register in phase estimation). This means that unless we have an infinite number of counting qubits, our probability of getting back the closest estimate will always be less than 1, though it will be peaked at the closest estimate [[2]](#references).

To generate the correct distribution to compare to, we use that the probability of measuring the raw bitstring <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}b"> from the counting qubits is given by equation (5.25) in [[2]](#references) as

<p align=center>
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}|\alpha_b|^2\equiv\left|\frac{1}{2^t}\left(\frac{1-e^{2\pi{i}(2^t\varphi-b)}}{1-e^{2\pi{i}(\varphi-b/2^t)}}\right)\right|^2">,
</p>

with <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}t"> being the number of counting qubits, and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\varphi"> being the correct phase in the state with eigenvalue <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}e^{2\pi{i}\varphi}">. Note, in this explanation, we are being a bit vague about the transformation between amplitude estimation, phase estimation, and our shifted functions for the sake of brevity.

### General Quantum Circuit
<p align=center>
<img src="../_doc/images/monte-carlo/qae_circuit.png" width = 800>
</p>

*Fig 1. Diagram of general quantum circuit for [Quantum Amplitude Estimation](../amplitude-estimation/)*

### Algorithm Steps

1. Generate the amplitude generator <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}"> from the function <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{F}"> and distribution <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{R}">.

2. Use [Amplitude Estimation](../amplitude-estimation/) to find the amplitudes.

3. If using method 1, shift the amplitudes according to the inverse shift outlined in the section on Implementing F (method 1).


## Gate Implementation

To generate our amplitude amplification operator <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}"> used in the amplitude estimation routine, we need to have a gate implementation of our probability distribution and function. In this benchmark, we implement two methods for generating these two elements of a Monte-Carlo simulation:

- **Method 1** implements circuits for <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{R}"> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{F}"> which allows arbitrary distributions and functions. This generally results in much deeper circuits. Our default choices for <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{R}"> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{F}"> are a truncated Gaussian distribution with variable mean and the function <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f(x)=x^2">.
- **Method 2** implements circuits for <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{R}"> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{F}"> which are extremely simple distributions and functions. This allows the algorithm to run with much less deep circuits. For <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{R}">, we use a uniform distribution, and for <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{F}">, we use a function similar to the balanced oracle used in the [Deutsch-Jozsa algorithm](../deutsch-jozsa/).

### Implementing R (method 1)

To generate <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{R}"> for an arbitrary probability distribution we use methods introduced in [[3]](#references). We use the intuition that a distribution can be broken into successive bisectings of regions. This property can be visualized in the below image, where we are estimating the probability of being in a specific region of a Gaussian distribution. We have each sector labeled by the qubit state which would have this probability, with the height being the corresponding probability.

<p align="center">
<img src="../_doc/images/monte-carlo/distribution.png">
</p>

We start off with our first bisection (in red), where for this Gaussian distribution, we have an even chance of being in the left or right region. We then apply this halving again for both sections. We then bisect each region again (orange), calculating the probability of being in the left or right side given we are know we are in one of the sides of the distribution. For example, we multiply the probability of being in the left side of the left region to find the total probability of being in the first of the now four regions.

If we iteratively apply this algorithm, we see that by just dividing up the regions, we can fully generate any probability distribution with however many qubits we want.

In order to implement this process, we use controlled Ry gates, as when an Ry gate acts only on a single qubit in the 0 state, the rotation is applied as:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\large{RY}(\theta)|0\rangle=\cos\frac{\theta}{2}|0\rangle+\sin\frac{\theta}{2}|1\rangle">
</p>

This will clearly split any region in half with arbitrary probabilities for the two states determined by the angle theta. The controlled aspect applies the part of the algorithm where we are given that we are in a particular region. This can be noticed on the above image by noticing that any two states below a region have the same first qubits, with the last, new qubit being either a 0 or 1.

Below is an example circuit which uses this bisecting method to create a uniform superposition. With the angle <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}-\pi/2">, we are evenly distributing between the two halves of the region. Then, by using controlled operations based on if the controlled qubit is 1 (0), shown by the closed (open) circle, we select which region be are bisecting. Note we do not modify the last qubit, as when applying <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\mathcal{R}">, we leave the last qubit for defining the correct states in <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\mathcal{F}">.

<p align=center>
<img src="../_doc/images/monte-carlo/R_method_1.png">
</p>

To see more in-depth explanation of how this process works, the math is explaned in a more detail in [[3]](#references).

---

### Implementing F (method 1)

In implementing our <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{F}">, we follow the process outlined in [[1]](#references). This uses the idea that if we have an operation of the form: <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|x\rangle_n|0\rangle\rightarrow|x\rangle_n(\cos(\zeta(x))|0\rangle+\sin(\zeta(x))|1\rangle)"> with <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\zeta(x)"> as a polynomial, then this operator can be efficently constructed with multi-controlled Y-rotations. So, we have two steps to implementing <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{F}">: transforming our function such that <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\sin^2(\zeta(x))=f(x)"> and then impelementing this polynomial via controlled rotations.

To generate our <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\zeta(x)">, instead of just setting this equal to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\sin^{-1}(\sqrt{f(x)})">, we instead find a Taylor approximation of 

<p align=center>
<img src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\zeta(x)=\frac{1}{c}\left(\sin^{-1}\left(\sqrt{c^\star\left(f(x)-\frac{1}{2}\right)+\frac{1}{2}}\right)\right)">
</p>

with <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}c^\star=(2\epsilon)^{1/u+1}"> where <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\epsilon"> is the desired error in the approximation, and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}u"> is the degree of the Taylor approximation for <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\zeta(x)">. The default values are <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\epsilon=0.05"> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}u=2">, which have shown to perform well without introducing additional errors. This shift in <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\zeta(x)"> leads to a better approximation; the mathematical reasonings for choosing this shift is explained in depth in [[1]](#references).

Now that we have <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\zeta(x)">, we can describe how to implement this polynomial via controlled rotations. To do this, we find a way of representing the polynomial in terms of sums of products of the states on each qubit. For example, for <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\zeta(x)=ax^2+bx+c"> and 2 data qubits, we can use that <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}x=2q_1+q_0">, with <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}q_i"> representing the state of qubit <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}i">. Then, utilizing <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}q_i^2=q_i"> allows us to write our polynomial as <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\zeta(x)=(4a+2b)q_1+4aq_0q_1+(a+b)q_0+c">. The mapping <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|x\rangle_n|0\rangle\rightarrow|x\rangle_n(\cos(\zeta(x))|0\rangle+\sin(\zeta(x))|1\rangle)"> is then applied by the circuit below, which can clearly be seen to apply the equation with a little thought.

<p align=center>
<img src="../_doc/images/monte-carlo/poly_exp.png">
</p>

Finally, we give an example of what this circuit looks like when <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}f(x)=x^2"> with the default parameters and 3 data qubits:

<p align=center>
<img src="../_doc/images/monte-carlo/F_method_1.png">
</p>

---

### Implementing R (method 2)

For our <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{R}"> method 2, our distribution is just a uniform distribution on all states, which can be easily implemented with Hadamards on all qubits. Again, since we use the last qubit for distinguishing states when applying <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{F}">, we do not modify it when preparing <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{R}">  .

<p align=center>
<img src="../_doc/images/monte-carlo/R_method_2.png">
</p>

---

### Implementing F (method 2)

For our <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{F}"> in method 2, we describe a simple circuit where in <img align= center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}\mathcal{F}|i\rangle_n|0\rangle=|i\rangle_n\big(\sqrt{1-f(i)}|0\rangle+\sqrt{f(i)}|1\rangle\big)\end{align*}">, <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}f(i)=1"> if <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}i"> has an odd number of 1's in its binary expansion, and is 0 otherwise. This is clearly implemented by the below circuit of successive controlled NOT operations. We also note that because this function does not use the shifts described in <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{F}"> method 1, we no longer need to do any shifting or inverse shifting of the measured amplitudes.

<p align=center>
<img  src="../_doc/images/monte-carlo/F_method_2.png">
</p>

## References

[1] Woerner, S., Egger, D.J. Quantum risk analysis. npj Quantum Inf 5, 15 (2019). 
    [`doi:10.1038`](https://doi.org/10.1038/s41534-019-0130-6)

[2] Michael A. Nielsen and Isaac L. Chuang. (2011).
    Quantum Computation and Quantum Information: 10th Anniversary Edition (10th ed.). 
    Cambridge University Press, New York, NY, USA.

[3] Lov Grover, Terry Rudolph. (2002)
    Creating superpositions that correspond to efficiently integrable probability distributions.
    [`arXiv:quant-ph/0208112`](https://arxiv.org/abs/quant-ph/0208112)
