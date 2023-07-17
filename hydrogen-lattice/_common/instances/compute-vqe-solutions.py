#!/usr/bin/env python3

import os
import sys

import pathlib
import numpy as np
from qiskit.quantum_info import SparsePauliOp

# import from common.py
# to avoid working directory issues, go via folder path with pathlib
common_folder = pathlib.Path(__file__).parent / ".."
common_folder = common_folder.resolve()
sys.path.append(str(common_folder))

from common import read_vqe_instance

instance_files = []
for root, dirs, files in os.walk("."):
    for name in files:
        if name.endswith(".json"):
            instance_files.append(os.path.join(root, name))

instance_files.sort()

print("instances found: {}".format(len(instance_files)))

for path in instance_files:
    print("working on: {}".format(path))
    instance = read_vqe_instance(path)

    num_atoms = len(instance["atoms"])

    if num_atoms > 8:
        continue  # too many qubits to solve below

    paired_hamiltonian = instance["paired_hamiltonian"]
    paired_matrix = SparsePauliOp(
        list(paired_hamiltonian.keys()), coeffs=list(paired_hamiltonian.values())
    ).to_matrix()

    jordan_wigner_hamiltonian = instance["jordan_wigner_hamiltonian"]
    jordan_wigner_matrix = SparsePauliOp(
        list(jordan_wigner_hamiltonian.keys()),
        coeffs=list(jordan_wigner_hamiltonian.values()),
    ).to_matrix()

    solution_path = path.replace(".json", ".sol")

    with open(solution_path, "w") as file:
        file.write(f"doci_energy: {np.linalg.eigvalsh(paired_matrix)[0]}\n")
        file.write(f"fci_energy: {np.linalg.eigvalsh(jordan_wigner_matrix)[0]}\n")
        file.write(f"hf_energy: {instance['hf_energy']}\n")
