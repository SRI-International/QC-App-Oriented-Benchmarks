#
# hydrogen_lattice/_common
#
# This file contains code that can be shared by all API instances of this benchmark,
# e.g. loading of problem instances and expected solution

import os
import json

INSTANCE_DIR = "instances"

# Utility functions for processing Max-Cut data files
# If _instances is None, read from data file.  If a dict, extract from a named field
# (second form used for Qiskit Runtime and similar systems)

# DEVNOTE: change these as needed for VQE and hydrogen lattice


def read_puccd_instance(file_path, _instances=None):
    if isinstance(_instances, dict):
        inst = os.path.splitext(os.path.basename(file_path))[0]
        return _instances.get(inst, {}).get("instance", (None, None))

    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, "r") as file:
            nodes = int(file.readline())
            edges = []
            for line in file.readlines():
                parts = line.split()
                edges.append((int(parts[0]), int(parts[1])))

        return nodes, edges

    else:
        return None, None


def read_puccd_solution(file_path, _instances=None):
    if isinstance(_instances, dict):
        inst = os.path.splitext(os.path.basename(file_path))[0]
        return _instances.get(inst, {}).get("sol", (None, None))

    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, "r") as file:
            objective = int(file.readline())
            solution = [int(v) for v in file.readline().split()]

        return objective, solution

    else:
        return None, None
