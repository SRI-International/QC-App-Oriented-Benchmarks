import random
from hamlib._common.new_functions import create_measurement_circuts, do_execute, compute_energy
from hamlib._common.new_functions import compute_exact_value
from qiskit_aer import Aer
from _common.qiskit import execute
from hamlib import hamlib_simulation_benchmark
from hamlib._common import observables
from hamlib.qiskit import hamlib_simulation_kernel
from hamlib._common.new_functions import create_measurement_circuts, do_execute, compute_energy
from qiskit.circuit.library import EfficientSU2
import numpy as np
import time


def get_ansatz(num_qubits, params):
    ansatz = EfficientSU2(num_qubits).decompose()
    ansatz.barrier()

    params_dict = {p: params[i] for i, p in enumerate(ansatz.parameters)}

    # Use assign_parameters instead of bind_parameters
    ansatz = ansatz.assign_parameters(params_dict)
    
    return ansatz


def generate_random_int_and_bitstring(num_qubits):
    """
    Generates a random bitstring representing an integer from 0 to N-1,
    given a number of qubits exponentiated where N is a power of 2.

    Args:
        num_qubits (int): number of qubits for which to generate random int state.

    Returns:
        str: A random bitstring representation of an integer in range [0, N-1].
    """

    N = 2 ** num_qubits
    
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError("N must be a power of 2.")

    num_bits = N.bit_length() - 1  # Determine the number of bits needed
    random_number = random.randint(0, N - 1)  # Pick a random number in range [0, N-1]
    
    return random_number, format(random_number, f'0{num_bits}b')  # Convert to bitstring with leading zeros


def estimate_expectation_value_top(num_qubits, sparse_pauli_terms, group_method, num_shots, K, t,
                                   num_k, use_diag_method, init_state, initial_circuit, metrics_object, distribute_shots, wsd_method,
                                   backend_id):
    ts = time.time()

    ######### Use Qiskit Estimator

    if group_method == "estimator":

        # create Trotterized evolution circuit for HamLib Hamiltonian
        ts = time.time()
        qc, _ = hamlib_simulation_kernel.HamiltonianSimulation(
            num_qubits=num_qubits, 
            ham_op=sparse_pauli_terms,
            K = K, t = t,
            init_state = init_state,
            append_measurements = False,
            method = 1, 
            initial_circuit = initial_circuit
        )
    
        create_time = round(time.time()-ts, 3)
        metrics_object["create_base_time"] = create_time
        #print(f"\n... finished creating base circuit, total creation time = {create_time} sec.\n")
    
        #print(qc)
            
        # Ensure that the pauli_terms are in 'full' format, not 'sparse' - convert if necessary
        pauli_terms = observables.ensure_pauli_terms(sparse_pauli_terms, num_qubits=num_qubits)
        pauli_terms = observables.swap_pauli_list(pauli_terms)
        
        estimator_energy = observables.estimate_expectation_with_estimator(execute.backend, qc, pauli_terms,
                                                            num_shots=num_shots, noise_model=execute.noise)

        obs_time = round(time.time()-ts, 3)
        metrics_object["observable_compute_time"] = obs_time
        
        metrics_object["grouping_time"] = 0.0
        #metrics_object["create_base_time"] = 0.0
        metrics_object["append_measurements_time"] = 0.0
        metrics_object["execute_circuits_time"] = 0.0
    
        return estimator_energy

    ######### Group Commuting Terms

    # Arrange the Pauli terms into commuting groups based on group_method
    
    pauli_term_groups, pauli_str_list = hamlib_simulation_benchmark.find_pauli_groups(num_qubits, sparse_pauli_terms, group_method, num_k)

    grouping_time = round(time.time()-ts, 3)
    metrics_object["grouping_time"] = grouping_time
    #print(f"\n... finished grouping of terms, total grouping time = {grouping_time} sec.\n")
    
    ######### Create Base Circuit
    
    # create Trotterized evolution circuit for HamLib Hamiltonian
    ts = time.time()
    qc, _ = hamlib_simulation_kernel.HamiltonianSimulation(
        num_qubits=num_qubits, 
        ham_op=sparse_pauli_terms,
        K = K, t = t,
        init_state = init_state,
        append_measurements = False,
        method = 1,
        initial_circuit = initial_circuit
    )

    create_time = round(time.time()-ts, 3)
    metrics_object["create_base_time"] = create_time
    #print(f"\n... finished creating base circuit, total creation time = {create_time} sec.\n")

    #print(qc)

    ######### Append Measurement Circuits
    
    # Append measurement circuits for each term group and return an array of circuits
    circuits = create_measurement_circuts(qc, num_qubits, pauli_term_groups, pauli_str_list, use_diag_method, num_k)


    ######### Execution
    
    results, pauli_term_groups = do_execute(backend_id, circuits, num_shots, pauli_term_groups, distribute_shots, ds_method = wsd_method)

    ######### Compute Energy
    
    energy = compute_energy(num_qubits, results, pauli_term_groups, group_method, num_k)
    energy = np.real(energy)

    return energy


def prepare_random_initial_state(num_qubits):
    ansatz = EfficientSU2(num_qubits)  
    num_params = ansatz.num_parameters

    # Generate initial parameters with correct length
    initial_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
    initial_qc = get_ansatz(num_qubits, initial_params)

    # we create an initial state vector for use in computing exact value
    backend = Aer.get_backend('statevector_simulator')
    init_state = backend.run(initial_qc).result().get_statevector().data
    return init_state, initial_qc


def do_random_state_loop(num_qubits, sparse_pauli_terms, group_method, num_shots, K, t, 
                         num_k, use_diag_method, iterations, initial_states, initial_qcs, distribute_shots, 
                         backend_id, wsd_method='max'):
         
    # create arrays to hold metrics
    init_values = []
    exact_energies = []
    computed_energies = []
    metrics_array = []
    from hamlib._common.new_functions import initialize_metrics_row

    print("")
    
    # For each iteration, generate random input, compute observable, and store data
    for it in range(iterations):
        init_state = initial_states[it]
        initial_qc = initial_qcs[it]
        #reset the metrics object
        metrics_object = initialize_metrics_row()
        
        ######### Compute Exact Energy 
        
        exact_energy = compute_exact_value(num_qubits, init_state, sparse_pauli_terms)
        exact_energies.append(exact_energy)
    
        ######### Estimate Energy using Quantum Algorithm
        
        # estimate expectation using the benchmark functions
        energy = estimate_expectation_value_top(num_qubits, sparse_pauli_terms, group_method, num_shots, K, t,
                                                num_k, use_diag_method, init_state, initial_qc, metrics_object, distribute_shots, 
                                                wsd_method, backend_id)   
        computed_energies.append(energy)
    
        # Append metrics from this iteration to the array of metrics
        metrics_array.append(metrics_object)

        print(".", end='')

    print("")

    return init_values, exact_energies, computed_energies, metrics_array




