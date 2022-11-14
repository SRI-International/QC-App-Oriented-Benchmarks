#!/usr/bin/env python3

import os, statistics, math, time
import common
import neal
import dimod
from datetime import datetime
import time

from dwave.system.samplers import DWaveSampler
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from neal import SimulatedAnnealingSampler


def maxcut_sa(nodes, edges, shots):

    results = {}
    results["stats"] = {}
    results["timing"] = {}
    results["stats"]["max"] = 0
    results["stats"]["mean"] = 0
    results["stats"]["sd"] = 0
    results["stats"]["wallClockStart"] = 0
    results["stats"]["wallClockEnd"] = 0

    sampler = SimulatedAnnealingSampler()

    h = {}
    J = {}
    for e in edges:
        if e in J:
            J[e] += 1
        else:
            J[e] = 1

    results["stats"]["wallClockStart"] = time.time()
    response = sampler.sample_ising(h, J, num_reads=shots)
    results["stats"]["wallClockEnd"] = time.time()
    results["solution"] = response.first.sample
    results["timing"] = response.info

    solution_values = []
    for sample, counts, energy in response.data(fields=['sample', 'num_occurrences', 'energy']):
        cut_size = (len(J) - energy)/2
        
        sol = [0 if sample[i] <= 0 else 1 for i in range(nodes)]
        sol_eval = common.eval_cut(nodes, edges, sol)
        if not math.isclose(sol_eval, cut_size):
            results["warning"] = "WARNING: objective values do not match"

        for i in range(counts):
            solution_values.append(sol_eval)

    results["stats"]["max"] = max(solution_values)
    results["stats"]["mean"] = statistics.mean(solution_values)
    results["stats"]["sd"] = statistics.stdev(solution_values)

    return (results)


def maxcut_qa(nodes, edges, shots, token, solverTime, embedding=None):
    ''' solverTime: in microseconds
    '''

    results = {}
    results["stats"] = {}
    results["timing"] = {}
    results["stats"]["max"] = 0
    results["stats"]["mean"] = 0
    results["stats"]["sd"] = 0
    results["warning"] = ""
    results["stats"]["wallClockStart"] = 0
    results["stats"]["wallClockEnd"] = 0

    h = {}
    J = {}
    for e in edges:
        if e in J:
            J[e] += 1
        else:
            J[e] = 1

    results["stats"]["wallClockStart"] = time.time()

    qpu = DWaveSampler(token=token, solver={'topology__type': 'pegasus'})
    
    if (embedding==None):
        sampler = EmbeddingComposite(qpu)
    else:
        sampler = FixedEmbeddingComposite(qpu, embedding=embedding)

    sampleset = sampler.sample_ising(h, J, num_reads=shots, annealing_time=solverTime)

    results["stats"]["wallClockEnd"] = time.time()

    results["solution"] = sampleset.first.sample
    results["timing"] = sampleset.info
    
    if (embedding == None):
        embedding = sampleset.info["embedding_context"]["embedding"]

    solution_values = []
    for sample, counts, energy in sampleset.data(fields=['sample', 'num_occurrences', 'energy']):
        cut_size = (len(J) - energy)/2

        sol = [0 if sample[i] <= 0 else 1 for i in range(nodes)]
        sol_eval = common.eval_cut(nodes, edges, sol)

        if not math.isclose(sol_eval, cut_size):
            results["warning"] = "WARNING: objective values do not match"

        for i in range(counts):
            solution_values.append(sol_eval)

    results["stats"]["max"] = max(solution_values)
    results["stats"]["mean"] = statistics.mean(solution_values)
    results["stats"]["sd"] = statistics.stdev(solution_values)

    return (sampleset, results, embedding)