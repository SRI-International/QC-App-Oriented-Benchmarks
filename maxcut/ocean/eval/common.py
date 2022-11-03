INSTANCE_DIR = 'instance'
INSTANCE_FOCUS_DIR = 'instance-focus'
INSTANCE_FINAL_DIR = 'instance-final'

RESULTS_DIR = 'results'

def read_maxcut_instance(file_path):
    with open(file_path, 'r') as file:
        nodes = int(file.readline())
        edges = []
        for line in file.readlines():
            parts = line.split()
            edges.append((int(parts[0]), int(parts[1])))

    return nodes, edges

def read_maxcut_solution(file_path):
    with open(file_path, 'r') as file:
        objective = int(file.readline())
        solution = [int(v) for v in file.readline().split()]

    return objective, solution

def eval_cut(nodes, edges, solution):
    assert(nodes == len(solution))

    cut_size = 0
    for i,j in edges:
        if solution[i] != solution[j]:
            cut_size += 1

    return cut_size


def optimal_maxcut(input_file_path):
    solution_file = input_file_path.replace(".txt",".sol")
    objective, solution = read_maxcut_solution(solution_file)

    return objective
