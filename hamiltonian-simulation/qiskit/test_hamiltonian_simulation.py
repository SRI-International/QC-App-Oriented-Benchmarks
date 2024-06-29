import hamiltonian_simulation_kernel as kernel
import hamiltonian_simulation_benchmark as benchmark 
import numpy as np 
import os 
import json 
import sys

from qiskit.primitives import Sampler

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit"]
import execute as ex
import metrics as metrics

import pytest

# Import precalculated data to compare against
filename = os.path.join(os.path.dirname(__file__), os.path.pardir, "_common", "precalculated_data.json")
with open(filename, 'r') as file:
    data = file.read()
precalculated_data = json.loads(data)

@pytest.mark.parametrize("n_spins,hamiltonian", [(spin, hamiltonian) for spin in range(2,13) for hamiltonian in ["Heisenberg","TFIM"]])
def test_high_num_shots_heisenberg_circuit(n_spins, hamiltonian):
    """
    check high n_shots matches what's stored in precalculated_distribution by seeing if polarization fidelity is close to 1
    """
  
    w = 1
    t = .2 

    np.random.seed(26)
    hx = list(2 * np.random.random(20) - 1) # random numbers between [-1, 1]
    np.random.seed(75)
    hz = list(2 * np.random.random(20) - 1) # random numbers between [-1, 1]

    K = 5 

    ham_circ = kernel.HamiltonianSimulation(n_spins, K, t, hamiltonian, w, hx, hz, use_XX_YY_ZZ_gates= False) 

    shots = 10000

    results = Sampler().run(circuits=ham_circ, n_shots = shots).result().quasi_dists[0].binary_probabilities()
    correct_dist = precalculated_data[f"{hamiltonian} - Qubits{n_spins}"]

    polar_fid = metrics.polarization_fidelity(results, correct_dist)["hf_fidelity"]

    assert np.isclose(1, polar_fid, atol = .01)
