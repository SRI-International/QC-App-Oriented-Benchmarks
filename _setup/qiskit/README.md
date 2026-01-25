# Quantum Computing - Toolkit Information

# Qiskit Version

This directory describes the requirements and operational conventions for using Qiskit as the programming environment for running the benchmark programs contained in the QC-App-Oriented-Benchmarks repository.
In particular, this document explains how to set up the tools needed to run the Qiskit implementation of these benchmarks.

**NOTE** This repository now requires Qiskit version 1.0 or greater.  A branch named **__master-240322-works-0.46** has been preserved for use with earlier versions of Qiskit. 

## Configure a Qiskit Environment
If you are using Anaconda environments, create an environment named "qiskit" and then "activate" it using the following commands:

    conda create -n qiskit python=3

    conda activate qiskit

The conda environment is now ready for you to install the Qiskit package.

## Install Qiskit

Enter the following commands to install the latest version of Qiskit and the other required packages.

    pip install numpy matplotlib qiskit qiskit-ibm-runtime qiskit-aer notebook

You are now ready to run the benchmark programs.

## Configuring Quantum Hardware

The `qiskit` package allows quantum circuits to be executed on real quantum hardware hosted by [IBM Quantum](https://quantum-computing.ibm.com/) and many other vendors.

To use a hardware backend from a computer system provider other than IBM, you will need to install the "Qiskit Provider" module that is specific to that manufacturer. Please see the documentation provided by that provider.

To execute the benchmarks on IBM systems, go your account page, 
[create an account](https://quantum-computing.ibm.com/docs/manage/account/) in IBM Quantum and save the account token in your local machine using instructions [here](https://quantum-computing.ibm.com/docs/manage/account/ibmq).

## Configuring IBM Runtime primitives

The `qiskit-ibm-runtime` package offers the Sampler primitive and the Estimator primitive to execute quantum circuits. The Sampler primitive performs tasks of drawing samples from the output state of the quantum circuits in the computation basis, and the Estimator primitive performs tasks of estimating expectation values of observables with respect to the output state of the quantum circuits. 

To configure the options for the Sampler and the Estimator, simply specify with keyword arguments in `exec_options` in `benchmarks-qiskit.ipynb`. Below is an example of configuring the Sampler options:
```
    exec_options = {
        "use_ibm_quantum_platform": False,
        "use_sessions": True,
        "sampler_options": {
            "dynamical_decoupling": {
                "enable": True,
            }
        }
    }
```

For more options to run the Sampler and the Estimator primitives, please go to related pages on the [IBM Quantum Documentation](https://docs.quantum.ibm.com/guides/runtime-options-overview).

## Run the benchmark programs in a Jupyter Notebook

Many Python users prefer to execute these benchmark programs in a Jupyter notebook.
Execute the following commands in the top-level directory of the benchmark repository and invoke the Jupyter notebook server.

    jupyter-notebook
    
This will invoke the Jupyter notebook in a new browser tab. There you can select the benchmarks-qiskit.ipynb notebook and execute most of the benchmarks.

The first code cell of the notebook is configured by default to execute all the benchmarks on the Qiskit Aer simulator, running locally. Simply un-commenting the lines in one of the hardware sections will execute the benchmarks on a different backend system.
    
Once configured, you can do a Run All command to execute all the top-level benchmarks at once.

Important note: there may be costs associated with execution on some hardware systems. You may consider lowering the value of the num_shots, max_circuits, and max_qubits settings in the first code cell, during your initial testing to avoid unexpected charges.


## Run the benchmark programs in a command window.

For example, in an Anaconda command window, you can enter the following commands to change the directory to the Qiskit Bernstein-Vazirani directory and run the benchmark program:

    cd [your github home directory]/QC-App-Oriented-Benchmarks/bernstein_vazirani

  
    python bv_benchmark.py
    
This will execute the benchmark program and report the benchmark metrics to the console.

The other benchmarks follow a similar format and structure and are executed in the same way (using the appropriate benchmark program filename).



## Tested Versions

The repository has been validated on Linux using the following versions as minimums:

    Miniconda Version: 4.10.3
    Python Version: 3.9 or later
    Qiskit Version: 1.0 or later

Earlier (or later) versions of the software might work without issues, but the benchmark has been specifically validated on these versions. If you have any issues installing, please raise an bug report in the issues tab of the repository.
