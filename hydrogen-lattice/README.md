# Hydrogen Lattice - Benchmark Program

This benchmark uses the VQE algorithm [[1]](#references) as an example of a quantum application that can simulate the dynamics of a lattice arrangement of hydrogen atoms and determine its lowest energy state.

With the VQE algorithm applied to this simulation, the goal is to find ..

This benchmark measures the performance characteristics of quantum computing systems when executing a simulation application that uses the VQE algorithm to compute properties of the simulation.

The remainder of this README offers a brief summary of the benchmark and how to run it.  For more detail, please see the aforementioned paper.

## Problem outline

Describe ...

## Benchmarking

In the run() method for the benchmark, there are a number of optional arguments that can be specified. All of the arguments can be examined in the source code, but the key arguments that would typically be modifed from defaults are the following:

(note these need to be modifed for the VQE ...)

```
    method : int, optional
        If 1, then do standard metrics, if 2, implement iterative algo metrics. The default is 1.
    rounds : int, optional
        number of QAOA rounds. The default is 1.
    degree : int, optional
        degree of graph. The default is 3. Can be -3 also.
    thetas_array : list, optional
        list or ndarray of beta and gamma values. The default is None, which uses [1,1,...].
    use_fixed_angles : bool, optional
        use betas and gammas obtained from a 'fixed angles' table, specific to degree and rounds
    parameterized : bool, optional
        Whether to use parametrized circuits or not. The default is False.
    max_iter : int, optional
        Number of iterations for the minimizer routine. The default is 30.
    score_metric : list or string, optional
        Which metrics are to be plotted in area metrics plots. The default is 'fidelity'. For method 2 s/b 'approx_ratio'.
    x_metric : list or string, optional
        Horizontal axis for area plots. The default is 'cumulative_exec_time' Can be 'cumulative_elapsed_time' also.
```

## Classical algorithm

Describe ...

## Quantum algorithm

Describe ...

### General Quantum Circuit

Describe ...

### Algorithmic Visualization

Describe ...

### Algorithm Steps

Describe ...

## Gate Implementation

Describe ...

## Circuit Methods

Describe ...

## References

Modify for the VQE and hydrogen lattice problem ...

[Solving combinatorial optimization problems using QAOA (Qiskit Tutorial)](https://qiskit.org/textbook/ch-applications/qaoa.html)

