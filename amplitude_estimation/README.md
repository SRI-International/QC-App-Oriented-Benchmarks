# Quantum Amplitude Estimation - Prototype Benchmark Program

Quantum Amplitude Estimation (QAE) [[1]](#references) is an extremely useful algorithm which provides a quadratic speedup over classical computers for wide classes of problems which would typically be solved by classical Monte Carlo simulations. This algorithm uses two ubiquitous elements of quantum algorithms: Quantum Amplitude Amplification (QAA), described in this README, and [Phase Estimation (PE)](../phase-estimation/). 

## Problem outline

The Quantum Amplitude Estimation Algorithm was first proposed by Brassard et al. as an extension of [Grover's search algorithm](../grovers/). The goal of this algorithm is to estimate a value <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a"> given a unitary operation <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}\"> defined as:

<p align="center">
    
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}|\psi\rangle\equiv\mathcal{A}|0\rangle_{n+1}=\sqrt{1-a}|\psi_0\rangle|0\rangle+\sqrt{a}|\psi_1\rangle|1\rangle">

</p>

Where <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_0\rangle,|\psi_1\rangle"> are any two states on <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> qubits, not necessarily orthogonal. In this context, we can think of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}\"> as distinguishing between the "bad state" <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_0\rangle"> and "good state" <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle"> using the value of an additional qubit. Therefore, <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a"> is the probability of measuring a good state from the prepared state <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}|0\rangle">. 

## Benchmarking
The Amplitude Estimation algorithm is benchmarked by running `max_circuits` circuits for random <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a"/> values of the form <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\sin^2\left(\frac{n}{2^k}\right)"/> where <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}k"/> represents the total number of qubits in register 1 and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> is an integer between <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}0"/> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2^k-1"/>. Each circuit is repeated a number of times denoted by `num_shots`. We then run the algorithm circuit for numbers of qubits between `min_qubits` and `max_qubits`, inclusive. The test returns the averages of the circuit creation times, average execution times, fidelities, and circuit depths, like all of the other algorithms. For this algorithm's fidelity calculation, because we chose <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta"/> to never require more precision than the qubits in register 1 can provide, we compare the returned measurements against the distribution that has the single state with 100% probability using our [noise-normalized fidelity calculation](../_doc/POLARIZATION_FIDELITY.md).

## Classical algorithm

In order to estimate <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a"> using a classical computer, we see we have a quadratic runtime penalty to get the same uncertainty in the measurement. We go more in-depth in the classical process used and the exact scaling in the [Monte Carlo benchmark](../monte-carlo/). This benchmark's README is also where we describe the scaling of the quantum algorithm.

## Quantum algorithm

To find an estimation of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a">, we first use QAA to generate a unitary operator <img src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}Q"> which encodes <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a"> into a phase as <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta_a=\sin^{-1}{\sqrt{a}}\">. We can do this through calls to our oracle <img src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}A"> and its inverse; we do not require knowledge of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a"> to generate <img src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}Q">. By using PE, we can determine this phase. We then measure out the counting qubits to obtain an integer <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}y"> and associated <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\tilde{\theta}=2\pi\frac{y}{2^m}\">
We finally get an estimation of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a"> by evaluating <img src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\tilde{a}=\sin^2{\tilde{\theta}}\">.

### Quantum Amplitude Amplfication (QAA)

QAA is a process by which the amplitude of good states can be increased, in such a way that only roughly <img align=center src="https://latex.codecogs.com/svg.latex?\tiny\pagecolor{white}\,\frac{1}{\sqrt{a}}\"> applications of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}\"> is required to achieve a constant probability of measuring the good state for known values of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a">. [Grover's search algorithm](../grovers/) is based on a special case of QAA where <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a"> is known to be <img src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}1/N">, <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle\equiv|w\rangle"> is a single marked basis state, and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_0\rangle\equiv|s'\rangle"> is a uniform superposition of all basis states not including <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|w\rangle">. The visualization provided in the Grover's algorithm README can provide some intuition for the amplification operator. 

The amplification operator in QAE looks quite similar to Grover's algorithm. The idea here is to increase the amplitude of good states by reflecting amplitudes according to the objective qubit state, then performing a reflection about the full original <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n">+1 qubit state. In alignment with [[1]](#references), we call a reflection about objective qubit state <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle"> the operator <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}S_{\chi}\"> and call a reflection around our initial state <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi\rangle"> the operator <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}S_{|\psi\rangle}\">:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\mathcal{Q}=S_{|\psi\rangle}S_{\chi}\">
</p>

It is shown in [[2]](#references) that this has the desired effect of increasing the amplitude of the good states with the following relationship. Using <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta_a=\sin^{-1}{\sqrt{a}}\">:


<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\mathcal{Q}^k\mathcal{A}|0\rangle=\cos\big((2k+1)\theta_a\big)|\psi_0\rangle|0\rangle+\sin\big((2k+1)\theta_a\big)|\psi_1\rangle|1\rangle">
</p>

This has the amplitude of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle|1\rangle"> increasing for integer <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}k"> iterations with <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}1\leq\,k\leq\,\frac{1}{2}\big[\frac{\pi}{2\theta_a}-1\big]">.

For implementation purposes, it is necessary to define <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}S_{|\psi\rangle}\"> in terms of oracle access to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}\">. Since this is placing a minus sign on all amplitudes of states orthogonal to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi\rangle">, it is easier to implement the <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}-S_{|\psi\rangle}\">: placing a minus sign on the original state. In this case, we can implement this operator by transforming from the <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}\"> basis, which maps <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi\rangle"> to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle_{n+1},"> perform a reflection on <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle_{n+1}\"> and return to the <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}\"> basis. That is:


<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}-S_{|\psi\rangle}=\mathcal{A}S_{|0\rangle_{n+1}}\mathcal{A}^{-1}\">
</p>

This is convenient because <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}S_{|0\rangle_{n+1}}\"> can be easily implemented using <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}X,H,\"> and Multi-controlled NOT gates.

So, all together the QAA operator is given as:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\mathcal{Q}=-\mathcal{A}S_{|0\rangle_{n+1}}\mathcal{A}^{-1}S_{\chi}">
</p>

With some investigation, we can see the corresponence with Grover's algorithm. This is made more clear in [Gate Implementation](#gate-implementation) section.

### General Quantum Circuit 
<p align="center">
<img src="../_doc/images/amplitude-estimation/qae_circuit.png"  width="800" />
</p>

*Fig 1. Circuit diagram for QAE using PE as presented by Grinko et al. [[3]](#references)*

### Algorithm Steps

1. Generate <img src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{Q}"> from the amplitude generator <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}">. 

2. Use [Phase Estimation](../phase-estimation/) to find the eigenvalue(s) of <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}|\psi\rangle=\mathcal{A}|0\rangle_{n+1}"> with respect to <img src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{Q}">. It is shown in [[1]](#references) that the two eigenstates of <img src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}Q"> have eigenvalue <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}{e}^{\pm2\pi\,i\,\theta_a}\">. While <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}|\psi\rangle"> is not actually an eigenstate of <img src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{Q}">, PE guarentees that the returned bitstring will always correspond to one of the two eigenvalues.

2. Transform eigenvalue estimates into amplitude estimates. At this stage, the bitstrings corresponding to the two phases will lead to an estimator of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\tilde{\theta}=2\pi\frac{y}{2^m}\">and therefore <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\tilde{a}=\sin^2{\tilde{\theta_a}}\">. It is clear that both of the eigenvalues will lead to the same value of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}a"> due to the even-ness of the squared sine function. It is proven in [[1]](#references) that this follows the variance:
    <p align="center">
    <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}|a-\tilde{a}|\leq\frac{\pi}{2^m}+\frac{\pi^2}{2^{2m}}\">
    </p>
    with probability at least <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\frac{8}{\pi^2}\">.

## Gate Implementation

To implement this algorithm, all that is necessary is explicit definitions for the operators <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A},\mathcal{A}^{-1},S_{|0\rangle_{n+1}},-S_{\chi}\">. Note that we implement the negative sign, originally from <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}S_{|\psi\rangle}">, in our implementation of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}S_{\chi}\">.

<img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A},\mathcal{A}^{-1}\"> are given to the algorithm as the oracle. <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}-S_{\chi}\"> performs a phase flip only if the objective qubit is in the <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle"> state. This is equivalent to the operations <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}XZX"> on the objective qubit. Lastly, <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}S_{|0\rangle_{n+1}}\">, the phase flip only when all state registers are <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|0\rangle">, is equivalent to performing <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}X\"> on all qubits in the register except the objective qubit, then a multi-controlled-phase gate between the <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}n"> state qubits and objective qubit, and another set of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}X\"> gates. Equivalent to multi-controlled-phase is a <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}H\"> on the objective qubit, multi-controlled-NOT, and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}H\"> again on the objective qubit.

<p align="center">
<img src="../_doc/images/amplitude-estimation/ae_circuit.png" />
</p>

If we then copy the circuits for the oracle and diffuser from Grover's algorithm:

<p align="center">
<img align="center" src="../_doc/images/grovers/oracle.png"  width="400" />
</p>

<p align="center">
<img align="center" src="../_doc/images/grovers/diffuser.png"  width="400" />
</p>

We can see the correspondence that <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}S_{\chi}=U_{f}"> and <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\mathcal{A}S_{|0\rangle_{n+1}}\mathcal{A}^{-1}=U_s">. The circuit used for QAA (<img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}S_{\chi}">) is simpler than the Grover oracle (<img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}U_{f}">) as the "correct" state we are marking is only dependent on the objective qubit. The second statement can be explicitely seen by taking <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}"> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{A}^{-1}"> to be Hadamards on all qubits. Additionally, we note that while the negative sign in Grover's algorithm didn't matter as it was global, because <img src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\mathcal{Q}"> is not applied on all qubits, the negative sign matters.

### Default implementation of <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\mathcal{A}">

In our default implementation of A, we set <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_0\rangle=|00\ldots\rangle"> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi_1\rangle=|11\ldots\rangle">. To implement arbitrary angles, we use a y-rotation and CNOTs to engtangle all of the states. As an example, with two state qubits and the single "correct-ness" qubit, if we start in the initial state of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|000\rangle">, after applying an <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\textrm{Ry}(2\theta_a)"> gate, we will be in the state <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|00\rangle(\cos(\theta_a)|0\rangle+\sin(\theta_a)|1\rangle)">. By then applying CNOT gates controlled by the last qubits, we entangle the first two qubits with the state of the last: <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\cos(\theta_a)|000\rangle+\sin(\theta_a)|111\rangle">. By remembering that <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\theta_a=\sin^{-1}{\sqrt{a}}\">, we have now generated the state we are looking for.  

<p align="center">
<img src="../_doc/images/amplitude-estimation/A.png" />
</p>

## References

[1] Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
    Quantum Amplitude Amplification and Estimation.
    [`arXiv:quant-ph/0005055`](http://arxiv.org/abs/quant-ph/0005055)

[2] Rao, P., Yu, K., Lim, H., Jin, D., Choi, D.,  (2020).
    Quantum amplitude estimation algorithms on IBM quantum devices.
    [`arXiv:2008.02102`](https://arxiv.org/abs/2008.02102v1)
     
[3] Grinko, D., Gacon, J., Zoufal, C., & Woerner, S. (2019).
    Iterative Quantum Amplitude Estimation.
    [`arXiv:1912.05559`](https://arxiv.org/abs/1912.05559)