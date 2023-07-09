#
# Define a custom noise model to be used during execution of Qiskit Aer simulator
#

# Note: this custom definition is the same as the default noise model provided
# The code is provided here for users to copy to their own file and customize

from qiskit.providers.aer.noise import NoiseModel, ReadoutError
from qiskit.providers.aer.noise import depolarizing_error, reset_error

def my_noise_model():
    
    noise = NoiseModel()
    
    # Add depolarizing error to all single qubit gates with error rate 0.3%
    #                    and to all two qubit gates with error rate 3.0%
    depol_one_qb_error = 0.003
    depol_two_qb_error = 0.03
    noise.add_all_qubit_quantum_error(depolarizing_error(depol_one_qb_error, 1), ['rx', 'ry', 'rz'])
    noise.add_all_qubit_quantum_error(depolarizing_error(depol_two_qb_error, 2), ['cx'])

    # Add amplitude damping error to all single qubit gates with error rate 0.0%
    #                         and to all two qubit gates with error rate 0.0%
    amp_damp_one_qb_error = 0.0
    amp_damp_two_qb_error = 0.0
    noise.add_all_qubit_quantum_error(depolarizing_error(amp_damp_one_qb_error, 1), ['rx', 'ry', 'rz'])
    noise.add_all_qubit_quantum_error(depolarizing_error(amp_damp_two_qb_error, 2), ['cx'])

    # Add reset noise to all single qubit resets
    reset_to_zero_error = 0.005
    reset_to_one_error = 0.005
    noise.add_all_qubit_quantum_error(reset_error(reset_to_zero_error, reset_to_one_error),["reset"])

    # Add readout error
    p0given1_error = 0.000
    p1given0_error = 0.000
    error_meas = ReadoutError([[1 - p1given0_error, p1given0_error], [p0given1_error, 1 - p0given1_error]])
    noise.add_all_qubit_readout_error(error_meas)
    
    # assign a quantum volume (measured using the values below)
    noise.QV = 32
    
    #print("... using custom noise model")
    return noise
