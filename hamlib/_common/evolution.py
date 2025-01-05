'''
Evolution Exact Functions
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

"""
This module provides functions to classically compute the value of observables during Hamiltonian evolution,
serving as a reliable reference for benchmarking estimates obtained from quantum simulations.
"""

import copy
from math import sin, cos, pi
import time

import numpy as np
import scipy as sc

from qiskit.quantum_info import Operator, Pauli
from qiskit.quantum_info import Statevector

# Set numpy print options to format floating point numbers
np.set_printoptions(precision=3, suppress=True)

verbose = False
   
"""
Compute theoretical energies from Hamiltonian and initial state.
This version is returning an array of classically computed exact energies, one for each step of evolution over time.
"""
def compute_theoretical_energies(initial_state, H, time, step_size):

    if H is None:
        return [None]
        
    # Create the Hamiltonian matrix (array form)
    H_array = H.to_matrix()

    # need to convert to Statevector so the evolve() function can be used
    initial_state = Statevector(initial_state)
    
    # use this if string is passed for initialization
    #initial_state = Statevector.from_label("001100")

    # We define a slightly denser time mesh
    exact_times = np.arange(0, time+step_size, step_size)
    
    # We compute the exact evolution using the exp
    exact_evolution = [initial_state]
    exp_H = sc.linalg.expm(-1j * step_size * H_array)
    for time in exact_times[1:]:
        print('.', end="")
        exact_evolution.append(exact_evolution[-1].evolve(exp_H))

    # Having the exact state vectors, we compute the exact evolution of our operators’ expectation values.
    exact_energy = np.real([sv.expectation_value(H) for sv in exact_evolution])
    
    return exact_energy, exact_times
    
