#!/usr/bin/env python3

import os

import networkx

import common

SEED = 0

N_MIN = 4
N_MAX = 24

for n in range(N_MIN, N_MAX+1, 2):
    for d in range(3, n, 1):
        print('instance config: {} {} {}'.format(n, d, SEED))

        graph = networkx.random_regular_graph(d, n, seed=SEED)

        file_name = 'mc_{:03d}_{:03d}_{:03d}.txt'.format(n, d, SEED)
        file_path = os.path.join(common.INSTANCE_DIR, file_name)
        with open(file_path, 'w') as file:
            file.write('{}\n'.format(len(graph.nodes)))

            for i,j in graph.edges:
                file.write('{} {}\n'.format(i, j))