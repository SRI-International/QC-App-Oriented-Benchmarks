# MaxCut Algorithm - Benchmark Program

This benchmnark uses the MaxCut algorithm [[1]](#references) as an example of a quantum application that solves a combinatorial optimization problem.
With the MaxCut algorithm, the goal is to find the maximum cut size of an undirected graph.
It is representative of a class of optimization problems that are easy to specify but difficult to solve efficiently, NP-HARD problems.
These often arise from mapping practical applications to computing hardware and can appear as subroutines in composite algorithms.
The Max-Cut problem offers a simple early-stage target for evaluating the effectiveness of quantum computing solutions that are hybrid in nature, i.e. combining both classical and quantum computation.

This benchmark measures the performance characteristics of quantum computing systems when executing a combinatorial optimization application that uses either the QA and QAOA algorithms and can evaluate the gate model and annealing styles of quantum computing side-by-side.
It is designed to provide insight into unique aspects of quantum computing while maintaining a presentation recognizable to practitioners in the optimization field and uniformly captures, analyzes, and presents metrics associated with the execution of both models of quantum computing to support comparisons across architectures.

The remainder of this README is Work-in-Progress.

## Problem outline


## Benchmarking


## Classical algorithm


## Quantum algorithm


### General Quantum Circuit


### Algorithmic Visualization



### Algorithm Steps

  

## Gate Implementation



## Circuit Methods


## References

[Solving combinatorial optimization problems using QAOA (Qiskit Tutorial)](https://qiskit.org/textbook/ch-applications/qaoa.html)

[Unique Games hardness of Quantum Max-Cut,
and a vector-valued Borell’s inequality](https://arxiv.org/pdf/2111.01254.pdf)

[Max-Cut and Traveling Salesman Problem (Qiskit Tutorial)](https://qiskit.org/documentation/optimization/tutorials/06_examples_max_cut_and_tsp.html)

[Almost optimal classical approximation algorithms
for a quantum generalization of Max-Cut](https://arxiv.org/pdf/1909.08846.pdf)

[Quantum Approximate Optimization Algorithm for MaxCut: A Fermionic View](https://arxiv.org/pdf/1706.02998.pdf)
