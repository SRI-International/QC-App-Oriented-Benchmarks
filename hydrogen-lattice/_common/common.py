#
# hydrogen_lattice/_common
#
# This file contains code that can be shared by all API instances of this benchmark,
# e.g. loading of problem instances and expected solution

import os
import json
import numpy as np

INSTANCE_DIR = "instances"

# Utility functions for processing Max-Cut data files
# If _instances is None, read from data file.  If a dict, extract from a named field
# (second form used for Qiskit Runtime and similar systems)

# DEVNOTE: change these as needed for VQE and hydrogen lattice

def read_vqe_instance(file_path):
    with open(file_path, "r") as file:
        instance = json.load(file)
    return instance

def read_puccd_instance(file_path, _instances=None):
    if isinstance(_instances, dict):
        inst = os.path.splitext(os.path.basename(file_path))[0]
        return _instances.get(inst, {}).get("instance", (None, None, None))

    if os.path.exists(file_path) and os.path.isfile(file_path):
        #read .json file 
        instance = read_vqe_instance(file_path)

        #get paired hamiltonian ops and coefficient lists
        paired_hamiltonian = instance["paired_hamiltonian"]
        paired_hamiltonian_ops = list(paired_hamiltonian.keys())
        paired_hamiltonian_coeffs = list(paired_hamiltonian.values())

        #get jordan wigner ops and coefficient lists
        jordan_wigner_hamiltonian = instance["jordan_wigner_hamiltonian"]
        jordan_wigner_ops = list(jordan_wigner_hamiltonian.keys())
        jordan_wigner_coeffs = list(jordan_wigner_hamiltonian.values())

        #create a (n,3) array containing atomic lattice xyz positions
        atoms = instance["geometry"]["atoms"]
        xyz = np.zeros((len(instance["geometry"]["x"]),3))
        xyz[:,0] = instance["geometry"]["x"]
        xyz[:,1] = instance["geometry"]["y"]
        xyz[:,2] = instance["geometry"]["z"]

        return xyz, atoms, jordan_wigner_ops, jordan_wigner_coeffs, paired_hamiltonian_ops, paired_hamiltonian_coeffs
    else:
        return None, None, None, None, None, None


def read_puccd_solution(file_path, _instances=None):
    if isinstance(_instances, dict):
        inst = os.path.splitext(os.path.basename(file_path))[0]
        return _instances.get(inst, {}).get("sol", (None))

    if os.path.exists(file_path) and os.path.isfile(file_path):

        with open(file_path, 'r') as file:

            #create an arbitrary length list to store solution data
            #num_lines = len(file.readlines())
            num_lines = 2 #for now just hardcode it to 2
            solutions = np.zeros(num_lines)

            #go through file now and insert them into the list
            for index, line in enumerate(file):
                line = line.strip()  # remove leading/trailing whitespace and newline characters
                if line:
                    name, number = line.split(':')
                    solutions[index] = float(number.strip())

        return solutions.tolist() # for now this is the doci and fci energies, length 2 list

    else:
        return None

#debugging line
if __name__ == '__main__':
    file_path = "instances/h2_chain_0.75.json"
    print(read_puccd_instance(file_path, ))
    file_path = "instances/h2_chain_0.75.sol"
    print(read_puccd_solution(file_path,))

