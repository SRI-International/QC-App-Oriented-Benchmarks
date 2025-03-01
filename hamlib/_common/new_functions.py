'''
Hamiltonian Simulation Benchmark Program - Qiskit
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''

# These functions will all be moved to appropriate files later.
# This is simply staging ground enableing moveing them out of notebooks.

import numpy as np
import time

import hamlib_simulation_benchmark
import evolution_exact
import observables


metrics_object = {}

def initialize_metrics_row():
    global metrics_object
    #print("... init new metrics row")
    metrics_object = {}
    return metrics_object

#################################################

# generate an array of circuits, one for each pauli_string in list
# Create Measurement Circuits from Base Circuit and Pauli Terms
# Here, we append basis rotation gates for each Pauli Term group to the base evolution circuit
# to create an array of circuits for execution.

def create_measurement_circuts(qc, num_qubits, pauli_term_groups, pauli_str_list, use_diag_method, num_k):
    ts = time.time()

    # use normal basis rotation and measurement circuits
    if not use_diag_method:
        circuits = hamlib_simulation_kernel.create_circuits_for_pauli_terms(qc, num_qubits, pauli_str_list)
    
    # option to use new method for producing diagonalized measurement circuts with N-qubit groups
    else:
        #print(f"... ****** using diagonalization method for measurement circuits")
        
        # generate an array of circuits, one for each pauli_string in list
        from generate_measurement_circuits import create_circuits_for_pauli_terms_k_commute    
        circuits = [create_circuits_for_pauli_terms_k_commute(qc, ops, num_k) for ops in pauli_term_groups]
    
    """
    print(f"... Appended {len(circuits)} circuits, one for each group:")               
    for circuit, group in list(zip(circuits, pauli_term_groups)):
        print(group)
        #print(circuit)
    """
    
    append_time = round(time.time()-ts, 3)
    metrics_object["append_measurements_time"] = append_time
    ###print(f"\n... finished appending {len(circuits)} measurement circuits, total creating time = {append_time} sec.\n")

    return circuits


#################################################

def do_execute(backend_id: str, circuits: list, num_shots: int):
    
    # Initialize simulator backend
    
    #from qiskit_aer import Aer
    #####backend = Aer.get_backend('qasm_simulator')
    #backend = Aer.get_backend('statevector_simulator')     # doesn't work, only returns 1 shot
    
    ### print(f"... begin executing {len(circuits)} circuits ...")
    ts = time.time()
    
    # Execute all of the circuits to obtain array of result objects
    ###### results = backend.run(circuits, num_shots=num_shots, noise_model=execute.noise).result()

    results = hamlib_simulation_benchmark.execute_circuits(
                                    backend_id = backend_id,
                                    circuits = circuits,
                                    num_shots = int(num_shots / len(circuits))
                                    )

    #for ca in results.get_counts():
    #    print(ca)
    
    exec_time = round(time.time()-ts, 3)
    metrics_object["execute_circuits_time"] = exec_time
    ###print(f"... finished executing {len(circuits)} circuits, total execution time = {exec_time} sec.\n")

    return results


#################################################

# Compute the total energy for the Hamiltonian
def compute_energy(num_qubits, results, pauli_term_groups):

    ###print(f"... begin computing observable value ...")
    ts = time.time()
    
    total_energy, term_contributions = observables.calculate_expectation_from_measurements(
                                                num_qubits, results, pauli_term_groups)
    obs_time = round(time.time()-ts, 3)
    metrics_object["observable_compute_time"] = obs_time
    #print(f"... finished computing observable value, computation time = {obs_time} sec.\n")
    
    #print(f"    Total Energy: {round(np.real(total_energy), 4)}")
    ### print(f"    Term Contributions: {term_contributions}\n")

    create_time = 0
    group_time = 0
    append_time = 0
    exec_time = 0
    
    total_time = group_time + create_time + append_time + exec_time + obs_time
    total_time = round(total_time, 3)   
    metrics_object["total_time"] = total_time
    ###print(f"\n... total observable computation time = {total_time} sec.\n")
    
    return total_energy


#################################################

# Compute exact value
def compute_exact_value(num_qubits, init_state, sparse_pauli_terms):
    
    #print(f"... begin classical computation of expectation value ...")                 
    ts = time.time()

    if num_qubits <= hamlib_simulation_benchmark.max_qubits_exact:
        
        correct_exp, correct_dist = evolution_exact.compute_expectation_exact(
                init_state,
                observables.ensure_pauli_terms(sparse_pauli_terms, num_qubits),
                0.0            # time
                )
    else:
        correct_exp = 0.001
        correct_dist = None
            
    exact_time = round(time.time()-ts, 3)
    metrics_object["exact_time"] = exact_time
    """
    print(f"... exact computation time = {exact_time} sec")
    
    print(f"\nExact expectation value, computed classically: {round(np.real(correct_exp), 4)}")
    print(f"Estimated expectation value, computed using quantum algorithm: {round(np.real(total_energy), 4)}\n")
    """
    #simulation_quality = round(np.real(total_energy) / np.real(correct_exp), 3)
    #print(f"    ==> Simulation Quality: {np.real(simulation_quality)}\n")

    return correct_exp 
    
    
#################################################

def compute_timing_stats(metrics_list):
    """
    Computes the average and standard deviation of selected timing metrics.

    Args:
        metrics_list (list): A list of dictionaries containing timing metrics.

    Returns:
        dict: A dictionary containing both the average and standard deviation for selected fields.
    """
    if not metrics_list:
        return {}

    # Define fields that should have both avg and stddev 
    selected_fields = ['create_base_time', 'append_measurements_time', 'execute_circuits_time', 'observable_compute_time']

    # Initialize result dictionary
    stats = {}

    # Compute averages and stddevs for selected fields
    for key in selected_fields:
        values = [d[key] for d in metrics_list if key in d]  # Only use existing keys
        if values:  # Ensure we don't compute on an empty list
            stats[key] = np.mean(values)
            stats[f"{key}_stddev"] = np.std(values, ddof=1)

    # Compute averages for all fields (without stddev) including 'exact_time'
    all_keys = set().union(*metrics_list)  # Get all unique keys from all dictionaries
    for key in all_keys:
        if key not in stats:  # Only compute if not already processed
            values = [d[key] for d in metrics_list if key in d]  # Filter missing keys
            if values:
                stats[key] = np.mean(values)

    return stats

