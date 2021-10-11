# Quantum Computing - Toolkit Information

# Cirq Version

This directory describes the requirements and operational conventions for using Cirq as the programming environment for running the prototype benchmark programs contained in the QC-Proto-Benchmarks repository.
In particular, this document explains how to set up the tools needed to run the Cirq implementation of these benchmarks.

Note: the instructions contained here describe configuring a Windows environment to run the benchmark programs. Similar procedures will be used in a Linux environment using appropriate syntax of course.

## Configure a Python Environment

The Cirq version of the prototype benchmark programs require that you have available Python version 3.6 or later, and have installed the necessary Python packages.

If you have a proper Python environment available, skip this section and go directly the the *'Install Cirq'* section below.

If you do not already have Python available, a convenient way to set one up is to download a minimum version of the Anaconda package (called Miniconda). Go to the URL below and follow the instructions to set up the "Miniconda" package.

    https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html

Once you have installed the Miniconda package, from the Windows Start menu launch an Anaconda prompt in which you will run the programs. It is recommended that you create a conda "environment" to hold the specific set of Python packages you will install to run the benchmark programs. Create an environment named "cirq" and then "activate" it using the following commands:

    conda create -n cirq python=3

    conda activate cirq

The conda environment is now ready for you to install the Cirq package.

## Install Cirq

Enter the following commands to install the latest version of Cirq and other required packages.

    conda install scipy

    pip install cirq notebook

You are now ready to run the benchmark programs.

## Configuring Quantum Hardware

More to come on this later.


## Run the benchmark programs in an Anaconda command window.

For example, in an Anaconda command window, you can enter the following commands to change directory to the Cirq Bernstein-Vazirani directory and run the benchmark program:

    cd [your github home directory]\QC-Proto-Benchmarks\bernstein-vazirani\cirq
  
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

The repository has been validated on Linux using the following minimum package versions:

    Miniconda version: 4.10.3
    Python Version: 3.9.7
    Cirq Version: 0.12.0

Earlier (or later) versions of the software might work without issues, but the benchmark has been specifically validated on these versions. If you have any issues installing, please raise an bug report in the issues tab of the repository.
