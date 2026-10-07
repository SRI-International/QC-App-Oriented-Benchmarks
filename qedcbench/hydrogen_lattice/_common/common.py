# hydrogen_lattice/_common
#
# This file contains code that can be shared by all API instances of this benchmark,
# e.g. loading of problem instances and expected solution

from __future__ import annotations

import os
import json
import numpy as np
import glob

INSTANCE_DIR = "instances"


# Utility functions for processing Max-Cut data files
# If _instances is None, read from data file.  If a dict, extract from a named field
# (second form used for Qiskit Runtime and similar systems)

# DEVNOTE: Python 3.10 will support the following argument syntax in all the methods below. 
#          However, for backwards compatibility with 3.8 and 3.9, we reduce the type checking (for now)
#
#   def read_paired_instance(
#       file_path: str, _instances: dict | None = None
#   ) -> tuple[list[str], list[float]] | tuple[None, None]:


def read_vqe_instance(file_path) -> dict:
    """Generate a dictionary containing JSON problem instance information."""

    with open(file_path, "r") as file:
        instance = json.load(file)
    return instance


def read_paired_instance(file_path: str, _instances: dict = None) -> tuple:
    """Return the paired hamiltonian operators and their coefficients."""
    if isinstance(_instances, dict):
        inst = os.path.splitext(os.path.basename(file_path))[0]
        return _instances.get(inst, {}).get(
            "instance",
            (
                None,
                None,
            ),
        )

    if os.path.exists(file_path) and os.path.isfile(file_path):
        # read .json file
        instance = read_vqe_instance(file_path)

        # get paired hamiltonian ops and coefficient lists
        paired_hamiltonian = instance["paired_hamiltonian"]
        paired_hamiltonian_ops = list(paired_hamiltonian.keys())
        paired_hamiltonian_coeffs = list(paired_hamiltonian.values())

        return (
            paired_hamiltonian_ops,
            paired_hamiltonian_coeffs,
        )
    else:
        return None, None


def read_jw_instance(file_path: str, _instances: dict = None) -> tuple:
    """Return the Jordon Wigner hamiltonian operators and their coefficients."""
    if isinstance(_instances, dict):
        inst = os.path.splitext(os.path.basename(file_path))[0]
        return _instances.get(inst, {}).get(
            "instance",
            (
                None,
                None,
            ),
        )

    if os.path.exists(file_path) and os.path.isfile(file_path):
        # read .json file
        instance = read_vqe_instance(file_path)

        # get jordan wigner ops and coefficient lists
        jordan_wigner_hamiltonian = instance["jordan_wigner_hamiltonian"]
        jordan_wigner_ops = list(jordan_wigner_hamiltonian.keys())
        jordan_wigner_coeffs = list(jordan_wigner_hamiltonian.values())

        return (
            jordan_wigner_ops,
            jordan_wigner_coeffs,
        )
    else:
        return (
            None,
            None,
        )


def read_geometry_instance(file_path: str, _instances: dict = None) -> tuple:
    """Return geometry information from a file path. The xyz information is returned as a (n, 3) array."""
    if isinstance(_instances, dict):
        inst = os.path.splitext(os.path.basename(file_path))[0]
        return _instances.get(inst, {}).get(
            "instance",
            (
                None,
                None,
            ),
        )

    if os.path.exists(file_path) and os.path.isfile(file_path):
        # read .json file
        instance = read_vqe_instance(file_path)

        # create a (n,3) array containing atomic lattice xyz positions
        atoms = instance["atoms"]

        # form a (n,3) array. use the length of x pos. to get n
        xyz = np.zeros((len(instance["x"]), 3))
        xyz[:, 0] = instance["x"]
        xyz[:, 1] = instance["y"]
        xyz[:, 2] = instance["z"]

        return (xyz, atoms)
    else:
        return (
            None,
            None,
        )



def read_puccd_solution(file_path: str, _instances: dict = None) -> tuple:
    """Return solution information from a file path. Information includes the method used to generate
    the solution and also the numerical value of the solution itself."""

    if isinstance(_instances, dict):
        inst = os.path.splitext(os.path.basename(file_path))[0]
        return _instances.get(inst, {}).get("sol", (None))

    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, "r") as file:
            solution_method_names = []
            solution_values = []

            # go through file now and insert them into the list
            for index, line in enumerate(file):
                line = (
                    line.strip()
                )  # remove leading/trailing whitespace and newline characters
                if line:
                    name, number = line.split(":")
                    solution_method_names.append(str(name.strip()))
                    solution_values.append(number)

        return (
            solution_method_names,
            solution_values,
        )  # for now this is the doci and fci energies

    else:
        return None, None

# get a list of instance file paths for num_qubits, or single file path if radius given
def get_instance_filepaths(num_qubits=2, radius=None):

    # return a single file path
    if radius is not None:
        sradius = f"{radius:06.2f}".replace(".","_")
        instance_filepath = os.path.join(os.path.dirname(__file__), \
            INSTANCE_DIR, f"h{num_qubits:03}_chain_{sradius}.json")
        if not os.path.isfile(instance_filepath):  
            instance_filepath = None
        return instance_filepath
    
    # or list of multiple filepaths
    instance_filepath_list = [file \
        for file in glob.glob(os.path.join(os.path.dirname(__file__), \
        INSTANCE_DIR, f"h{num_qubits:03}_*_*_*.json"))]

    return instance_filepath_list
    

# return the random energies dictionary
def get_random_energies_dict():
    # declare the random energy file path relative to the current file path
    random_energy_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_sampler/precomputed_random_energies.json") 

    with open(random_energy_file_path) as f:
        random_energies_dict = json.load(f)

    return random_energies_dict
