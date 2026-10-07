# Image Recognition - Benchmark Program

**NOTE:** This benchmark program will not function with Qiskit 1.0, as the opflow package has been completely removed.  However, it will execute with Qiskit <= 0.46. This is an active issue and the benchmark will be updated soon.

NOTE: This entire README is a WORK-IN-PROGRESS

This benchmark uses Quantum neural networks & Quantum Machine Learning  [[1]](#references) as an example of a quantum application that can classify images from mnist data set using classical optimizer

This benchmark measures the performance characteristics of quantum computing systems using an image recognition algorithm as the basis for bencmkaring.

The remainder of this README offers a brief summary of the benchmark and how to run it.  

## Requirements

This benchmark program requires that the Scikit-learn python package is installed in your environment prior to execution. The Scikit-learn package provides a number of unsupervised and supervised learning algorithms, as well as tools for loading various test databases.

To install the Scikit-learn package, execute the following commands:

    pip install scikit-learn noisyopt


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

(this list needs to be made complete)

```
    method : int, optional
        If 1, test standard metrics on ansatz, if 2, execute iterative training pass, if 3, execute testing pass. Default is 1.
    thetas_array : list, optional
        list or ndarray of beta and gamma values. The default is None, which uses [1,1,...].
    parameterized : bool, optional
        Whether to use parametrized circuits or not. The default is False.

    train_size : int, optional
        Size of training dataset. The default is 200.
    test_size : int, optional
        Size of testing dataset. The default is 50.
    batch_size : int, optional
        Size of a batch of images. The default is 50.
    test_pass_count : int, optional
        Number of passes to make during testing over the history of training results. The default is 50.
    max_iter : int, optional
        Number of iterations for the minimizer routine. The default is 30.

    score_metric : list or string, optional
        Which metrics are to be plotted in area metrics plots. The default is 'accuracy' for methods 2 and 3.
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


