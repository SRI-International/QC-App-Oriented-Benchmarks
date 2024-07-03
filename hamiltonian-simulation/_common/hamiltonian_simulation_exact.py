import sys

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit"]
sys.path[1:1] = ["../qiskit"]
from qiskit import QuantumCircuit 
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import TimeEvolutionProblem, SciPyRealEvolver
import numpy as np
np.random.seed(0)
import execute as ex
import metrics as metrics

def HamiltonianSimulationExact(hamiltonian: SparsePauliOp, t: float, init_state: QuantumCircuit):
    """
    Perform exact Hamiltonian simulation using classical matrix evolution.

    Args:
        hamiltonian (SparsePauliOp): Qiskit hamiltonian to run the hamiltonian simulation algorithm with.  
        t (float): Duration of simulation.
        init_state (QuantumCircuit): The chosen initial state. 

    Returns:
        dict: The distribution of the evolved state.
    """
    time_problem = TimeEvolutionProblem(hamiltonian, t, initial_state=init_state)
    result = SciPyRealEvolver(num_timesteps=1).evolve(time_problem)
    return result.evolved_state.probabilities_dict()
