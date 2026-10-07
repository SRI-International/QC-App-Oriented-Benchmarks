import os
import json

INSTANCE_DIR = 'instances'

# Utility functions for processing Max-Cut data files
# If _instances is None, read from data file.  If a dict, extract from a named field
# (second form used for Qiskit Runtime and similar systems)

def read_maxcut_instance(file_path, _instances=None):

    if isinstance(_instances, dict):
        inst = os.path.splitext(os.path.basename(file_path))[0]
        return _instances.get(inst, {}).get("instance", (None, None))
        
    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            nodes = int(file.readline())
            edges = []
            for line in file.readlines():
                parts = line.split()
                edges.append((int(parts[0]), int(parts[1])))

        return nodes, edges
        
    else:
        return None, None

def read_maxcut_solution(file_path, _instances=None):

    if isinstance(_instances, dict):
        inst = os.path.splitext(os.path.basename(file_path))[0]
        return _instances.get(inst, {}).get("sol", (None, None))
        
    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            objective = int(file.readline())
            solution = [int(v) for v in file.readline().split()]

        return objective, solution
    
    else:
        return None, None

def eval_cut(nodes, edges, solution, reverseStep = 1):
    assert(nodes == len(solution))
    solution = solution[::reverseStep] # If reverseStep is -1, the solution string is reversed before evaluating the cut size
    cut_size = 0
    for i,j in edges:
        if solution[i] != solution[j]:
            cut_size += 1

    return cut_size

# Load from given filename and return a list of lists
# containing fixed angles (betas, gammas) for multiple degrees and rounds
def read_fixed_angles(file_path, _instances=None):

    if isinstance(_instances, dict):
        #inst = os.path.splitext(os.path.basename(file_path))[0]
        #return _instances.get(inst, {}).get("fixed_angles", (None, None))
        return _instances.get("fixed_angles", (None, None))

    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, 'r') as json_file:
        
            # 'thetas_array', 'approx_ratio_list', 'num_qubits_list
            fixed_angles = json.load(json_file)

        return fixed_angles
    
    else:
        return None

# return the thetas array containing betas and gammas for specific degree and rounds     
def get_fixed_angles_for(fixed_angles, degree, rounds):

    if str(degree) in fixed_angles and str(rounds) in fixed_angles[str(degree)]:
        param_pairs = fixed_angles[str(degree)][str(rounds)]
        return [param_pairs['beta'] + param_pairs['gamma']]
    else:
        return None 
        