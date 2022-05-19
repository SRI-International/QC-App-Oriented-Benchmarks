
import os

INSTANCE_DIR = 'instances'
RESULTS_DIR = 'results'

def read_maxcut_instance(file_path):
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

def read_maxcut_solution(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            objective = int(file.readline())
            solution = [int(v) for v in file.readline().split()]

        return objective, solution
    
    else:
        return None, None

def eval_cut(nodes, edges, solution):
    assert(nodes == len(solution))

    cut_size = 0
    for i,j in edges:
        if solution[i] != solution[j]:
            cut_size += 1

    return cut_size
