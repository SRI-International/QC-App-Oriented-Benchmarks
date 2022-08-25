#!/usr/bin/env python3

import os

from gurobipy import Model, GRB

def read_maxcut_instance(file_path):
    with open(file_path, 'r') as file:
        nodes = int(file.readline())
        edges = []
        for line in file.readlines():
            parts = line.split()
            edges.append((int(parts[0]), int(parts[1])))

    return nodes, edges

instance_files = []
for root, dirs, files in os.walk('.'):
    for name in files:
        if name.endswith('.txt'):
            instance_files.append(os.path.join(root, name))

instance_files.sort()

print('instances found: {}'.format(len(instance_files)))

for path in instance_files:
    #print()
    print('working on: {}'.format(path))
    nodes, edges = read_maxcut_instance(path)

    m = Model()
    variables = [m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x_'+str(i)) for i in range(nodes)]
    m.update()

    obj = 0
    for i,j in edges:
        obj += -2*variables[i]*variables[j] + variables[i] + variables[j]

    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    objective = int(m.ObjVal)
    solution = [0 if variables[i].X <= 0.5 else 1 for i in range(nodes)]

    solution_path = path.replace('.txt', '.sol')
    with open(solution_path, 'w') as file:
        file.write('{}\n'.format(objective))
        for v in solution:
            file.write('{} '.format(v))
        file.write('\n')
