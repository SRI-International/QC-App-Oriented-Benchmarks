# Preparing to Run Benchmarks

You can run the benchmarks in Qiskit, Cirq, Braket, or Q#. For each of the programming environments supported by this project, there is a subdirectory containing detailed information about requirements and operational conventions. To view this information, along with instructions on how to to configure a unique environment for the API of your choice, follow the links in [Links to API Specific Setup](#links-to-api-specific-setup) below.

All versions of the benchmark programs require that you have available the Python interpreter (**version 3.6 or later**), and have installed the necessary Python packages in a virtual environment. If you have a proper Python environment available, you may go directly to your preferred API directory by following one of the links under [Links to API Specific Setup](#links-to-api-specific-setup). For instructions on how to download Python and set up a base virtual environment, see [General Environment Setup](#general-environment-setup) below.

## Links to API Specific Setup
* [Qiskit](qiskit/README.md)
* [Qiskit in Azure Quantum](qiskit-azure-quantum/README.md)
* [Cirq](cirq/README.md)
* [Braket](braket/README.md)
* [Ocean](ocean/README.md)
* [Q# (Comming soon)](README.md)

# General Environment Setup

**Note**: All instructions contained here describe configuring a Windows environment to run the benchmark programs. Similar procedures will be used in a Linux environment using appropriate syntax of course.

If you do not already have Python available, a convenient way to set one up is to download a minimum version of the Anaconda package (called Miniconda). Go to the URL below and follow the instructions to set up the "Miniconda" package.

    https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html

Once you have installed the Miniconda package, from the Windows Start menu launch an Anaconda prompt in which you will run the programs. It is recommended that you create a conda "environment" to hold the specific set of Python packages you will install to run the benchmark programs. We recommend you create a separate conda environment for each API. Details for creating environments is specific to each API and can be found under the *'Configure a \<API\> Environment'* by following one of the links under [Links to API Specific Setup](#links-to-api-specific-setup).

See the link below for additional resources on conda environments.

    https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
