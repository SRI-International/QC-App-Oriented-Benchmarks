# Quantum Computing - Toolkit Information

# Qiskit Version

This directory describes the requirements and operational conventions for using Qiskit as the programming environment for running the prototype benchmark programs contained in the QC-Proto-Benchmarks repository.
In particular, this document explains how to set up the tools needed to run the Qiskit implementation of these benchmarks.

## Configure a Qiskit Environment
Create an environment named "qiskit" and then "activate" it using the following commands:

    conda create -n qiskit python=3

    conda activate qiskit

The conda environment is now ready for you to install the Qiskit package.

## Install Qiskit

After activating the conda environment, to ensure you are using the correct installation of `pip`, run the following command:

    pip show pip

If everything is working correctly, the `Location` field should have your newly created environment's name present. For example:

    Location: c:\users\[user]\miniconda\envs\qiskit\lib\site-packages

Enter the following commands to install the latest version of Qiskit and the other required packages.

    pip install numpy matplotlib qiskit "qiskit[visualization]" notebook

You are now ready to run the benchmark programs.

## Configuring Quantum Hardware

The `qiskit` package allows quantum circuits to be run in a real quantum hardware hosted by [IBM Q Experience](https://quantum-computing.ibm.com/). To use a hardware backend, 
[create an account](https://quantum-computing.ibm.com/docs/manage/account/) in IBM Q Experience and save the account token in your local machine using instructions [here](https://quantum-computing.ibm.com/docs/manage/account/ibmq).
 

## Run the benchmark programs in an Anaconda command window.

For example, in an Anaconda command window, you can enter the following commands to change directory to the Qiskit Bernstein-Vazirani directory and run the benchmark program:

    cd [your github home directory]\QC-Proto-Benchmarks\bernstein-vazirani\qiskit
  
    python bv_benchmark.py
    
This will execute the benchmark program and report the benchmark metrics to the console.

The other benchmarks follow a similar format and structure and are executed in the same way (using the appropriate benchmark pgrogram filename).

## Run the benchmark programs in a Jupyter Notebook

Many Python users prefer to execute their Python programs in a Jupyter notebook, which is automatically available with your Anaconda installation.
Execute the following commands to change directory to one that contains a Jupyter notebook and execute and invoke Jupyter notebook server.

    cd to directory containing jupyter notebook
    jupyter-notebook
    
This will then invoke the Jupyter notebook in a new browser tab. There you can copy and paste any of the benchmark program code and execute the programs interactively.
    
Note; In some Windows environments, it is necessary to install one additional package (if running a Jupyter notebook results in a Windows "kernel error"):

    conda install pywin32

Once installed, you should be able to successfully start your Jupyter notebook.

## Tested Versions

The repository has been validated on Linux using the following versions as minimums:

    Miniconda Version: 4.10.3
    Python Versions: 3.8.5 and 3.9.7
    Qiskit-Terra Version: 0.18.3

Earlier (or later) versions of the software might work without issues, but the benchmark has been specifically validated on these versions. If you have any issues installing, please raise an bug report in the issues tab of the repository.
