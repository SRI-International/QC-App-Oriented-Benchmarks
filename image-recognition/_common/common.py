#
# hydrogen_lattice/_common
#
# This file contains code that can be shared by all API instances of this benchmark,
# e.g. loading of problem instances and expected solution

import json
import numpy as np

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


data = [-0.07397172, 1.35735454, -0.3169999, 2.74608058, 2.57309511, 3.09098808,
        1.51373312, -0.01892117, 0.08219012, 0.22553648, 0.48439645, 0.97254863,
        0.95120939, 0.84353814, 0.36972275, 0.39970566, 0.22521085, -0.19281133,
       -0.36072423, 1.09389512, 0.8480923, 0.52705624, -0.21508983, 0.8841463]


json_data = json.dumps(data)

with open('data.json', 'w') as json_file:
    json_file.write(json_data)
    
    
    
def read_parameters(file_path) -> dict:
    """Generate a dictionary containing JSON problem instance information."""

    with open(file_path, "r") as json_file:
        json_data = json_file.read()
    loaded_data = json.loads(json_data)
    final_array = np.array(loaded_data)
    return final_array

