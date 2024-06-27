# Hamiltonian Simulation

Simulation of quantum systems is one of the most promising applications for quantum computers [1](#references). In the current version of this benchmark, we have three strategies for calculating fidelities. 

In the first strategy, we compare the quantum simulation against a classical circuit simultion in order to report our fidelity. This works well for small circuit sizes, but is not scalable past a certain number of qubits. 

In the second strategy, we compare the quantum simulation against a classical simultion of the exact Hamiltonian dynamics in order to report our fidelity. Again, this is not scalable.

In the third strategy, we use the mirror circuits method developed by Sandia Labratories [2](#references). This is scalable to all qubit sizes. 

## Problem outline

This benchmark is written as a simulation of non-trivial Hamiltonians. It is constrained to model a linear chain of interacting bodies with an open boundary condition. Our Hamiltonians of interest are the **Heisenberg model with disordered fields** and the **Transverse Field Ising Model (TFIM)**, both with open boundary conditions.

**For the Heisenberg Hamiltonian**, we start the system in an easily preparable classical state $|\psi(0)\rangle\equiv|010101\ldots\rangle$. **For the Transverse Field Ising Model (TFIM)**, we start the system in a GHZ state  $|\psi(0)\rangle = \left| \text{GHZ} \right\rangle = \frac{1}{\sqrt{2}} \left( |0\rangle^{\otimes n} + |1\rangle^{\otimes n} \right)$, where $n$ is equivalently the number of spins or qubits. In either case, we aim to evolve the system for $t$ time according to the solution to the Schrödinger equation with $H$ constant,

$$
|\psi(t)\rangle=e^{-i{H}t}|\psi(0)\rangle
$$


where we set <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\hbar=1"> here and elsewhere.

### Heisenberg Model

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}H=J\sum_{i=0}^{N-2}(\sigma^x_i\sigma^x_{i+1}+\sigma^y_i\sigma^y_{i+1}+\sigma^z_i\sigma^z_{i+1})+w\sum_{i=0}^{N-1}(h_{x,i}\sigma^x_i+h_{z,i}\sigma^z_i)"/>
</p>

Where <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}J"/> is the strength of the interaction, <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}w"/> is the strength of the disordered fields, <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}h_{x,i}"/> and <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}h_{z,i}"/> give the strength of the x and z disorded fields at site <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}i"/>, and <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\sigma^{\{x,y,z\}}_i"/> are the usual Pauli operators acting on site <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}i"/>. We will use the notation <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\{X_i,Y_i,Z_i\}\equiv\sigma^{\{x,y,z\}}_i"/> interchangably throughout this explanation.

The first sum represents the **interaction terms**, wheras the second sum represents the **disordered field terms** (see Fig 1.)

In our benchmarks, currently both $J=1$ and $w=1$.

### Transverse Field Ising Model (TFIM)

The TFIM Hamiltonian is given by:

$$
H= J\sum_{i=0}^{N-2}\sigma^z_i\sigma^z_{i+1} + h\sum_{i=0}^{N-1}\sigma^x_i
$$

Where <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}J"/> is the coupling strength between neighboring spins, and <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}h"/> is the strength of the transverse field.

In our benchmarks, currently both $J=1$ and $h=1$.

## Benchmarking
The Hamiltonian Simulation algorithm is benchmarked by running **just a single circuit**. This circuit is repeated a number of times denoted by `num_shots`. We then run the algorithm circuit for numbers of qubits between `min_qubits` and `max_qubits`, inclusive. The test returns the averages of the circuit creation times, average execution times, fidelities, and circuit depths, like all of the other algorithms. 

There are currently three methods for how to produce the fidelity metric. All three methods evolve a state, and create a metric based on how well the state evolved. 

The first two methods evolve an initial state a time $t$, and compare the final state against a precalculated distribution. Method = 1 creates the precalculated distribution from a noiseless simulation of the Hamiltonian Simulation quantum circuit. Method = 2 uses a classical matrix technique to simulate the evolution of the Hamiltonian directly. Wheras method = 1 only tests the performance of the hardware, method = 2 also tests the accuracy of the Hamiltonian simulation itself. 

We calculate these precalculated distributions in the jupyter notebook `precalculated_data.ipynb`, which stores the results for up to 20 qubits in the `precalculated_data.json` data file. The python code then imports the distributions from the `json` file. This is a less than ideal fidelity calculation as it does not scale to any size of qubits. It requires the classical simulation of matrix products, which requires resources exponential in number of qubits. 

In the `precalculated_data.ipnyb`, we set the trotterization steps (k) to 5 and the time to .2. For the Heisenberg Hamiltonian, $w$ is set to 1 but $J$ is hard-coded to 1. For TFIM, the Hamiltonian variables are both hard-coded to $J=1$ and $h=1$ respectively. 

Method = 3 uses a mirror circuit based off of the Hamiltonian Simulation circuit, designed so that the target distribution is trivial. It evolves an initial state forwards a time $t$, then backwards in time $t$, so that the final state should be the (trivial) initial state. Because this doesn't utilize classical resources to generate a comparison metric, this scales to any size of qubits. 


In all cases, we compare the resultant distribution using our [noise-normalized fidelity calculation](../_doc/POLARIZATION_FIDELITY.md).

In the run() method for the benchmark, there are a number of optional arguments that can be specified. Some of the key arguments are as follows: 

```
Parameters 
---- 

 min_qubits (int): Minimum number of qubits (smallest circuit is 2 qubits).
        max_qubits (int): Maximum number of qubits.
        max_circuits (int): Maximum number of circuits to execute per group.
        num_shots (int): Number of shots for each circuit execution.
        hamiltonian (str): Which hamiltonian to run. "heisenberg" by default but can also choose "TFIM". 
        method (int): Method for fidelity checking (1 for noiseless trotterized quantum, 2 for exact classical, 3 for mirror circuit.)
        use_XX_YY_ZZ_gates (bool): Flag to use unoptimized XX, YY, ZZ gates.
        random_pauli_flag (bool): Flag to activate more sophisticated mirror circuit formation that utilizes a layer of random paulis seperating the mirror circuits in addition to using a quasi-inverse rather than an inverse. 
        init_state (str): The desired initial state. Choices are "checkerboard" or "ghz". 
```

## Classical algorithm

Much effort has been done in the field of many-body physics to understand the approximate behaviors of Hamiltonians like the ones we have here. However, to calculate the evolution of an excited state through exact diagonalization scales approximately as <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}O(2^{3n})"> for <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}n"> qubits, quite poor scaling [[2]](#references). This quickly becomes intractible even utilizing extremely powerful classical supercomputers.

## Quantum algorithm

To run this algorithm on our quantum computer, we need to find a way to apply the unitary <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}U(t)\equiv{e}^{-i{H}t}"/> through a combination of quantum gates. In order to approximate this operator, we use Trotterization [[3]](#references), where we note that Lie product formula gives

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{-i{\sum_j{H}_j}t}=\lim_{k\rightarrow\infty}\left(\prod_j{e}^{-iH_j{t}/k}\right)^k"/>.
</p>

If we take <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}k"/> to be finite, this is called Trotterization. This has a gate complexity of <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}O(n^5)"/>, which is an exponential speedup. We can then apply successive layers of by exponentiating the individual terms in the Hamiltonian to approximate the evolution of any state. This makes the simulation easier, as it is much easier to calculate the gates which apply <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{i\theta\sigma^x_0\sigma^x_1}"/> and <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{i\theta\sigma^x_1\sigma^x_2}"/> than to find the gates which apply <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{i\theta(\sigma^x_0\sigma^x_1+\sigma^x_1\sigma^x_2)}"/>. This process can be visualized in the circuit diagram below for the **Heisenberg Hamiltonian** with a single step.

### General Quantum Circuit

<p align="center">
<img align=center src="../_doc/images/hamiltonian-simulation/ham_sim_circuit.png"  width="700" />
</p>

*Fig 1. Example of circuit with 1 Trotter step. We can see that our <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\sigma^x_i"/> and <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}\sigma^y_i"/> turned into Rx and Ry gates, while the two qubit interactions turned into the gates that result from exponentiating these terms in the Hamiltonian. Note that this circuit application is less efficient than applying the XX, YY, and ZZ operations all at once.*

<p align="center">
<img align=center src="../_doc/images/hamiltonian-simulation/ham_sim_full_circuit.png"  width="700" />
</p>

*Fig 2. Circuit with 2 Trotter steps and the optimal XXYYZZ operator.*

### Algorithm Steps

1. Initialize qubits in alternating state <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}|\psi(0)\rangle\equiv|010101\ldots\rangle"/>.

2. Build the Trotter step layer.

3. Apply the Trotter step layer for as many Trotter steps were chosen.

4. Measure out all of the qubits

## Gate Implementation

There are two options of circuit creation for this simulation:

- **Default:** Optimal implementation of <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{it(XX+YY+ZZ)}"/>, used as the default. See [[4]](#references) for reasoning for why this is the optimal application of gates.

- **use_XX_YY_ZZ_gates:** Simple implementation of <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{it(XX)}"/>, <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{it(YY)}"/>, and <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{it(ZZ)}"/>, provided for reference purposes and validation of the optimal implementation. In essence, we initially generate <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{it(ZZ)}"/> using two CNOT gates and an RZ gate. We then apply the XX and YY versions of this gate by providing a basis change from Z to X and from Z to Y, using Hadamard gates for the X transformation and using S and Hadamard gates for the Y transformation respectively. These circuits are below. It is possible to use this type of gates by passing `use_XX_YY_ZZ_gates=True` to the `run()` function.

<p align="center">
<img align=center src="../_doc/images/hamiltonian-simulation/XXYYZZ_gate.png"  />
</p>

*Fig 3. Optimal gate which applies <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{it(XX+YY+ZZ)}"/>.*

---

<p align="center">
<img align=center src="../_doc/images/hamiltonian-simulation/ZZ_gate.png"  />
</p>

*Fig 4. Naive <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{it(ZZ)}"/> gate.*

---

<p align="center">
<img align=center src="../_doc/images/hamiltonian-simulation/XX_gate.png"  />
</p>

*Fig 5. Naive <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{it(XX)}"/> gate.*

---

<p align="center">
<img align=center src="../_doc/images/hamiltonian-simulation/YY_gate.png"  />
</p>

*Fig 6. Naive <img align=center src="https://latex.codecogs.com/svg.latex?\pagecolor{white}e^{it(YY)}"/> gate.*
 
## References

[1] Feynman, RP. (1982) Simulating physics with computers. Int J Theor Phys 21:467–488.

[2] Proctor, T., Rudinger, K., Young, K. et al. Measuring the capabilities of quantum computers. Nat. Phys. 18, 75–79 (2022). https://doi.org/10.1038/s41567-021-01409-7 

[2] Andrew M. Childs, Dmitri Maslov, Yunseong Nam, Neil J. Ross, Yuan Su. (2017).
    Toward the first quantum simulation with quantum speedup.
    [`arXiv:1711.10980`](https://arxiv.org/pdf/1711.10980.pdf)

[3] Naomichi Hatano, Masuo Suzuki. (2005).
    Finding Exponential Product Formulas of Higher Orders
    [`arXiv:math-ph/0506007`](https://arxiv.org/abs/math-ph/0506007v1)

[4] Farrokh Vatan, Colin Williams. (2004).
    Optimal Quantum Circuits for General Two-Qubit Gates.
    [`arXiv:quant-ph/0308006`](https://arxiv.org/abs/quant-ph/0308006)

[5] D. Zhu, S. Johri, N. H. Nguyen, C. Huerta Alderete, K. A. Landsman, N. M. Linke, C. Monroe, A. Y. Matsuura. (2021).
    Probing many-body localization on a noisy quantum computer.
    [`arXiv:2006.12355`](https://arxiv.org/abs/2006.12355)

[//]: # (Below are some thoughts that went into the choice of the type of hamiltonian simulation to be used for this benchmark.)

[//]: # (Nearest-neighbor 1D, 2D and 3D cases are all physically motivated, corresponding to say, a nanowire, a thin film, and a cubic crystal. In this case, 1D is a suitable benchmark for current quantum computers since it has the least number of gates. As quantum computers improve, the simulation benchmarks could be designed to include higher dimensions.)

[//]: # (If interactions between all pairs of spins were to be added, it would correspond to a limit of the long-range Heisenberg model. There are studies on this, but it may not show the localization behavior being captured in the accuracy metric in the benchmark. For this reason, the benchmark was written for the 1D case only.)

[//]: # (The bodies being simulated can be represented as a linear chain of bodies, i.e. a 'open boundary condition' or in a circle, i.e. a 'periodic boundary condition'. In the limit of large system size, the boundary will not affect the bulk physics represented here. For small system sizes, periodic boundaries have some advantage in that they are less susceptible to 'edge effects', and the system can be Fourier transformed and studied in momentum space. Their drawback is that some numerical approximation techniques work poorly. In other simulation studies such as those of topological phenomena, there is important physics corresponding to a 'bulk-boundary correspondence', and the open edge is purposefully introduced.)

[//]: # (Therefore, the choice of boundary condition depends on the system being studied and the computational technique being employed. Here it does not matter too much.)
