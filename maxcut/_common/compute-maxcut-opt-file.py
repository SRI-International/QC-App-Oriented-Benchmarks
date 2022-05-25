#!/usr/bin/env python3

import os

from gurobipy import Model, GRB

import common

instance_files = []
for root, dirs, files in os.walk(common.INSTANCE_DIR):
    for name in files:
        if name.endswith('.txt'):
            instance_files.append(os.path.join(root, name))

instance_files.sort()

print('instances found: {}'.format(len(instance_files)))

for path in instance_files:
    #print()
    print('working on: {}'.format(path))
    nodes, edges = common.read_maxcut_instance(path)

    m = Model()
    variables = [m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x_'+str(i)) for i in range(nodes)]
    m.update()

    obj = 0
    for i,j in edges:
        obj += -2*variables[i]*variables[j] + variables[i] + variables[j]

    m.setObjective(obj, GRB.MAXIMIZE)

    m.update()

    m.optimize()

    objective = int(m.ObjVal)
    solution = [0 if variables[i].X <= 0.5 else 1 for i in range(nodes)]
   
    #print(m.display())
    #print(nodes) 
    #print(edges)
    #print(objective)
    #print(solution)

    solution_path = path.replace('.txt', '.sol')
    with open(solution_path, 'w') as file:
        file.write('{}\n'.format(objective))
        for v in solution:
            file.write('{} '.format(v))
        file.write('\n')
    #break