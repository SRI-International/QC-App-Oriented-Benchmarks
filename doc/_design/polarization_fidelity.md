# Practical Quantum Algorithm Benchmarking
In developing a suite of benchmarks intended by to evaluate quantum computers by their ability to provide an end user with a usable, productive result, a practical view must be taken in defining metrics of success of any test quantum algorithms. In particular, quantum state tomography, while able to provide insight into the path through Hilbert space that a quantum state takes throughout a quantum algorithm, requires an exponentially large number of measurements to be made to determine the accuracy with which a quantum computer performs the prescribed algorithm. *Citations on state tomography.*

Other ways of defining the empirical difference between outputs of a quantum computer can take into account only what the user is able to see; that is, the measurement probabilities over the computational basis states. In this case, it is standard to borrow measures of divergence from probability theory, such as the Hellinger Distance.

The Hellinger Distance is a similarity measure for two probability distributions <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}P,Q"> of the following form:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}H^2(P,Q)=1-\sum_{i}\sqrt{p_iq_i}\end{align*}" />
</p>


The squared Hellinger Fidelity, defined using the Hellinger Distance above and used in quantum computing applications like Qiskit, is defined as
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}F_H(P,Q)\equiv\left(1-H^2(P,Q)\right)^2=\left(\sum_{i=1}^N\sqrt{p_i&space;q_i}\right)^2\end{align*}"/>
</p>

A type of *f-divergence*, it is easy to show that <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}H(P,Q)"> will be equal to zero if <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}P"> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}Q"> are identical, and equal to one if <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}P"> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}Q"> have disjoint supports. *Citation on Hellinger distance*. As a consequence, the squared Hellinger Fidelity will always be nonzero if there is any shared support between <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}P"> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}Q">, which leads to a problem when considering the expected behavior of highly noisy quantum devices.

## Using the Squared Hellinger Fidelity

It is tempting to prescribe using the squared Hellinger Fidelity as the measure of success for the output of a quantum computer. Say that a quantum algorithm, when run on a noiseless device acting on some initial state, should produce the state
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}|\psi\rangle=\sum_{i=1}^Ne^{-i\theta_i}\sqrt{p_i}|i\rangle.\end{align*}" />
</p>

When tested however, a real device instead returns the (possibly mixed) state
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}\textbf{diag}(\rho_{\phi})=\{q_i\}\end{align*}" />
</p>

Then the benchmarker is left only with the resulting set of estimators <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\hat{Q}=\{\hat{q_i}\}"> in the form of the normalized measurement results of the noisy algorithm. In constructing a benchmark suite, it is easy enough to choose algorithm instances wherein the values <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\{p_i\}"> are known analytically. Therefore, for a given algorithm and empirical results, the squared Hellinger Fidelity measuring the quality of the results given the theoretical noiseless results is calculated as
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}F_H(P(|\psi\rangle),\hat{Q})=\left(\sum_{i=1}^N\sqrt{p_i\hat{q_i}}\right)^2\end{align*}" />
</p>

In practice, this yields a quantity bounded on <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}F_H\in\,[0,1]"> which offers a scale on which to compare two quantum devices running the same algorithm; a higher value of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}F_H"> means a result closer to the theoretical result and therefore the device with the higher <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}F_H"> has performed the test algorithm better.

## Maximally Noisy Output

Consider the output of any quantum algorithm running on a quantum computer with maximal noise; that is, near-zero coherence time and uncorrelated errors between each one or two qubit gate. On average, the result of a quantum algorithm of any significant length with be completely thermalized by accumulated noise and error, leading to a uniformly random output
<img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}U=\{u_i=\frac{1}{N}\}">.

It is impossible to obtain information from such a device - the output is independent of the input - and yet the squared Hellinger Fidelity between the output distribution and the theoretically correct state will always be nonzero. Namely, this limiting quantity is
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}F_H(P(|\psi\rangle),U)=\frac{1}{N}\left(\sum_{i=1}^N\sqrt{p_i}\right)^2\end{align*}" />
</p>

# High Noise Behavior

The limiting behavior of the squared Hellinger Fidelity measured on a maximally noisy device is essentially a measure of the spread of the noiseless distribution. In the case of quantum algorithms where the expected distribution is highly centralized <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\{p_i\approx\delta_{ik}\}"> for a single state <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|k\rangle"> (most of the probability mass is on a single marked state), a maximally noisy quantum device will see a squared Hellinger Fidelity fall off exponentially with the number of qubits:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}F_H(\{p_i=\delta_{ik}\},U)=\frac{1}{N}=\frac{1}{2^n}\end{align*}" />
</p>

This result already poses a problem in the low qubit count limit. If it is understood that theoretically, the squared Hellinger Fidelity is capable of returning values less than this parameter, a user may believe that the device is achieving more than it really is. In this case, despite the fact that the scale theoretically goes as low as <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}0">, a completely useless quantum device would achieve <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}F_H=0.25"> by sheer random guessing for a <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}2">-qubit instance.

It is also of interest to measure the quality of results for quantum algorithms with some significant variance in the theoretical result distribution. In this case, the skew can be much worse. For the sake of simplicity, consider the case where the noiseless state is an equal superposition over <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}K"> states:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(|\psi\rangle)\rightarrow\,p_i=\pagecolor{white}\begin{cases}\frac{1}{K}&1\leq\,i\leq\,K\\0&i\,>K\\\end{cases}" />
</p>

Then the squared Hellinger Fidelity returns
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}F_H(P(|\psi\rangle),U)=\frac{K}{N}\end{align*}" />
</p>

When trying to measure the performance of quantum algorithms with significant spread, the high noise limiting behavior of the squared Hellinger Fidelity may lead to artificially high results, especially with low qubit counts. This directly impedes the usefulness of the measure in a well-rounded applications focused benchmarking suite, in which a sliding scale is intended to provide the user confidence that high performance means the algorithm is far from noise-limited.

## Normalizing to High Noise Behavior
There are several ways to amend the squared Hellinger Fidelity as a metric to account for the shortcomings described so far. The authors argue that due to the dependence on spread of noiseless distribution, a simple rescaling to account for the fidelity inflation due to low qubit counts would not be enough. Instead, the best approach is to take full advantage of the fact that the analytic noiseless quantum state results <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}|\psi\rangle"> are known, and normalize the measure against the performance of full thermalizing noise for each algorithm instance\footnotemark . That is, define the Noise-Normalized Squared Hellinger Fidelity as a linear transformation which ensures that a uniform result distribution is always evaluated to be equal to <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}F_H=\frac{1}{N}">, leaving the measure consistent in the case that the noiseless distribution is fully centralized. The authors believe that the exponential decay with respect to qubit count is sufficient to distinguish incredibly noisy devices, so long as it is understood that this method will have artificially high minimum measured fidelities at low qubit counts. In algorithms where the effective spread <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\tilde{K}"> is a nontrivial function of <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}N">, the scaling can be arbitrary; this situation makes it that much more difficult to distinguish the behavior of accurate quantum computers producing non-localized results, and high noise machines matching some amount of spread anyway.

The proposed linear transformation takes the form:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}\widetilde{F_H}(P(|\psi\rangle),\hat{Q})&=s\left[F_H(P(|\psi\rangle),\hat{Q})-1\right]&plus;1\\[10pt]s&=\frac{N-1}{N\left[1-F_H(P(|\psi\rangle),U)\right]}\end{align*}" />
</p>

which guarantees that a fidelity equal to one is unchanged and the uniform distribution will decay exponentially; <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\widetilde{F_H}(P(|\psi\rangle),P(|\psi\rangle))=1"> and <img align=center src="https://latex.codecogs.com/svg.latex?\small\pagecolor{white}\widetilde{F_H}(P(|\psi\rangle),U)=\frac{1}{N}">.


**In the case we want to use polarization fidelity, replace the above with:**

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\begin{align*}\widetilde{F_H}(P(|\psi\rangle),\hat{Q})&=s\left[F_H(P(|\psi\rangle),\hat{Q})-1\right]&plus;1\\[10pt]s&=\frac{1}{1-F_H(P(|\psi\rangle),U)}\end{align*}" />
</p>
