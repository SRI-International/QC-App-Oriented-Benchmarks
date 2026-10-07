#!/usr/bin/env python3

import os
import networkx

INSTANCE_DIR = '.'
SEEDS = 1
N_MIN = 4
N_MAX = 26
R = 3
R_minus = 3

for n in range(N_MIN, N_MAX+1, 2):
    for s in range(0,SEEDS):
        print('instance config: {} {} {}'.format(n, R, s))

        graph = networkx.random_regular_graph(R, n, seed=s)

        file_name = 'mc_{:03d}_{:03d}_{:03d}.txt'.format(n, R, s)
        file_path = os.path.join(INSTANCE_DIR, file_name)
        with open(file_path, 'w') as file:
            file.write('{}\n'.format(len(graph.nodes)))

            for i,j in graph.edges:
                file.write('{} {}\n'.format(i, j))

for n in range(N_MIN, N_MAX+1, 2):
    for s in range(0,SEEDS):
        regularity = n-R_minus
        if regularity <= 3:
            continue
        print('instance config: {} {} {}'.format(n, regularity, s))

        graph = networkx.random_regular_graph(regularity, n, seed=s)

        file_name = 'mc_{:03d}_{:03d}_{:03d}.txt'.format(n, regularity, s)
        file_path = os.path.join(INSTANCE_DIR, file_name)
        with open(file_path, 'w') as file:
            file.write('{}\n'.format(len(graph.nodes)))

            for i,j in graph.edges:
                file.write('{} {}\n'.format(i, j))

for n in [40, 80, 160, 320]:
    for s in range(0,SEEDS):
        print('instance config: {} {} {}'.format(n, R, s))

        graph = networkx.random_regular_graph(R, n, seed=s)

        file_name = 'mc_{:03d}_{:03d}_{:03d}.txt'.format(n, R, s)
        file_path = os.path.join(INSTANCE_DIR, file_name)
        with open(file_path, 'w') as file:
            file.write('{}\n'.format(len(graph.nodes)))

            for i,j in graph.edges:
                file.write('{} {}\n'.format(i, j))
