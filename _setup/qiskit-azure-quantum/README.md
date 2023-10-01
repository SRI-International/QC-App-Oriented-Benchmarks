# Quantum Computing - Toolkit Information

# Qiskit Version for Azure Quantum

This directory describes the requirements and operational conventions for using Qiskit as the programming environment for running the benchmark programs contained in the QC-App-Oriented-Benchmarks repository.
In particular, this document explains how to set up the tools needed to run the Qiskit implementation of these benchmarks using the Azure Quantum SDK.

## Configure a Qiskit Environment
If you are using Anaconda environments, create an environment named "qiskit" and then "activate" it using the following commands:

    conda create -n qiskit python=3

    conda activate qiskit

The conda environment is now ready for you to install the Qiskit package.

## Install Qiskit

Enter the following commands to install the latest version of the Azure Quantum Qiskit SDK and the other required packages.

    pip install -U azure-quantum
    pip install -U azure-quantum[qiskit]

You are now ready to run the benchmark programs.

## Configure Azure Quantum for Access to Quantum Computing Systems

The Azure Quantum `qiskit` package allows quantum circuits to be executed on real quantum hardware or simulators provided by a variety of Azure Quantum partners.
Please see the Azure Quantum documentation for information about how to configure Azure Quantun to enable access to any of these backend systems.

To configure the benchmark notebooks to select a specific Azure Quantum target for execution, the run() method of each benchmark would require the following arguments be set. 

    hub = "azure-quantum";
    backend_id = "<YOUR_BACKEND_NAME_HERE>"

Using the Jupyter notebook described below makes this easy.

Before running in either a Jupyter notebook or from the command line, you will need to set the Azure Quantum resource_id and location information in the following environment variables:
 
    set AZURE_QUANTUM_RESOUCE_ID="<YOUR_RESOURCE_ID>"
    set AZURE_QUANTUM_LOCATION="<YOUR_LOCATION>"

You can find this information in the Azure portal under your workspace > Overview > "Resource ID" and "Location". To create an Azure Quantum workspace, you can read documentation at
[`https://aka.ms/AQ/Docs/CreateWorkspace`](https://aka.ms/AQ/Docs/CreateWorkspace)

## Run the benchmark programs in a Jupyter Notebook

Many Python users prefer to execute these benchmark programs in a Jupyter notebook.
Execute the following command in the top-level directory of the benchmark repository and invoke the Jupyter notebook server.

    jupyter-notebook
    
This will invoke the Jupyter notebook in a new browser tab. There you can select the benchmarks-qiskit.ipynb notebook and execute most of the benchmarks.

The first code cell of the notebook is configured by default to execute all the benchmarks on the Qiskit Aer simulator, running locally. Simply un-commenting these lines in the Azure Quantum section will select an Azure Quantum backend for the execution.

    hub = "azure-quantum"; backend_id = "<YOUR_BACKEND_NAME_HERE>"
    
Once configured, you can do a Run All command to execute all the top-level benchmarks at once.

    Important note: there may be costs associated with execution on some hardware systems.
    You may consider lowering the value of the num_shots, max_circuits, and max_quibts settings in the first code cell
    during your initial testing to avoid unexpected charges.

Another note: executing the benchmarks from an unauthenticated command window will result in Azure Quantum requesting authentication upon execution. To avoid this, you can download and install the Azure Quantum Command Line Tool (CLI). In a command window, execute the login command:

    az login

Once you are authenticated, executing the benchmarks will not request further authentication.

## Run the benchmark programs in a command window.

For example, in an Anaconda command window, you can enter the following commands to change directory to the Qiskit Bernstein-Vazirani directory and run the benchmark program:

    cd [your github home directory]\QC-App-Oriented-Benchmarks\bernstein-vazirani\qiskit
  
    python bv_benchmark.py
    
This will execute the benchmark program and report the benchmark metrics to the console.

The other benchmarks follow a similar format and structure and are executed in the same way (using the appropriate benchmark program filename).

## Tested Versions

The repository has been validated on Linux using the following versions as minimums:

    Miniconda Version: 4.10.3
    Python Versions: 3.8.5 and 3.9.7
    Qiskit-Terra Version: 0.18.3

Earlier (or later) versions of the software might work without issues, but the benchmark has been specifically validated on these versions. If you have any issues installing, please raise a bug report in the issues tab of the repository.
