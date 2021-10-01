import sys
sys.path[1:1] = [ "_common", "../../../_common" ]
import qsharp
from deutsch_jozsa import ConstantZero_Test
from deutsch_jozsa import ConstantOne_Test
from deutsch_jozsa import OddNumberOfOnes_Test
from deutsch_jozsa import NthQubitParity_Test
import time
import math
import numpy as np
import metrics

print("Deutsch-Jozsa Benchmark Program - Q#")

verbose = False

############### Execution and Data Analysis

# Run the DJ algorithm with the provided settings
def run_deutsch_jozsa(backend, number_of_qubits, oracle_type, num_shots):

    # Get the oracle
    oracles = [ ConstantZero_Test, ConstantOne_Test, OddNumberOfOnes_Test, NthQubitParity_Test ]
    expected_results = [ True, True, False, False ]
    oracle = oracles[oracle_type]

    # Estimate the resources required to run the circuit
    args = {"NumberOfQubits": number_of_qubits, "Validate": False}
    resource_estimates = oracle.estimate_resources(**args)

    # Run DJ on the correct backend
    correct_results = 0
    args = {"NumberOfQubits": number_of_qubits, "Validate": True}
    if backend == "simulator":
        for i in range(0, num_shots):
            is_constant = oracle.simulate(**args)
            print(is_constant)
            if is_constant == expected_results[oracle_type]:
                correct_results += 1

    else:
        raise f"Unknown backend {backend}"
    
    # Print the gate count for the circuit
    print(resource_estimates)

    # Record the fidelity
    fidelity = correct_results / num_shots
    if verbose: print(f"expected={expected_results[oracle_type]} count={correct_results} shots={num_shots} fidelity={fidelity}")
    return fidelity


################ Benchmark Loop

# Execute program with default parameters
def run (min_qubits=3, max_qubits=16, max_circuits=4, num_shots=5,
         backend="simulator"):

    # validate parameters (smallest circuit is 3 qubits)
    max_qubits = max(3, max_qubits)
    min_qubits = min(max(3, min_qubits), max_qubits)
    
    # Initialize metrics module
    metrics.init_metrics()
    metrics.set_plot_subtitle(f"Device = {backend}")

    # Execute Benchmark Program N times for multiple circuit sizes
    for num_qubits in range(min_qubits, max_qubits + 1):
    
        input_size = num_qubits - 1
        
        # determine number of circuits to execute for this group
        num_circuits = min(4, max_circuits)
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        
        # Loop over the circuits
        for type in range( num_circuits ):
            
            # Store an empty creation time metric since circuit creation isn't separated from execution in Q#
            ts = time.time()
            metrics.store_metric(num_qubits, type, 'create_time', 0)

            # Run the circuit on the provided backend
            fidelity = run_deutsch_jozsa(backend, num_qubits, type, num_shots)
            metrics.store_metric(num_qubits, type, 'fidelity', fidelity)
            metrics.store_metric(num_qubits, type, 'exec_time', time.time()-ts)
            metrics.store_metric(num_qubits, type, 'elapsed_time', time.time()-ts)
        
        metrics.finalize_group(str(num_qubits))

    # Plot metrics for all circuit sizes
    metrics.plot_metrics("Benchmark Results - Deutsch-Jozsa - Q#")


# if main, execute method
if __name__ == '__main__': run()
