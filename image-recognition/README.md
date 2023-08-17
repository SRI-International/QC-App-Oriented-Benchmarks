# Image Recognition - Benchmark Program

This benchmark uses Quantum neural networks & Quantum Machine Learning  [[1]](#references) as an example of a quantum application that can classify images from mnist data set using classical optimizer

With the Image recognition applied to this simulation, the goal is to find ..

This benchmark measures the performance characteristics of quantum computing systems for image recognition

The remainder of this README offers a brief summary of the benchmark and how to run it.  


## Current flow of implementation

    1. Fetch the MNIST dataset
    2. Access the data and target  
        i) filter the data with 7 and 9 ( for initial development we are doing binary classification )
    3. Split the data into train and test data ( 80% for training and 20% for testing )
    4. pca for dimensionality reduction (x_scaled_data is the scaled data)
    5. batching the data ( we are not using batching for now)
    6. custom varational circuit is defined for now instead of QCNN (var_circuit)
        i)   it will be updated once we have the QCNN
    7. Input data is encoded into quantum state and then passed through the custom variational circuit(qcnn_model)
    8. loss function is defined (loss) as our objective is to minimize the loss ( Pending )
        i)   Function to predict the label ( 7 or 9 i.e, 0,1) from circuit output is pending
        ii)  loss function is pending and will be updated once the above function is done
        iii) need to check if to use classical neural network to extract labels from the circuit output
    9. Optimizer is defined (optimizer) 
    10. Testing the model ( Pending )

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

Modify for the Image Recognition ...


