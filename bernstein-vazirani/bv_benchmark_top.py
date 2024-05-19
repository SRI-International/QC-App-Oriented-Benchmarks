"""
Bernstein-Vazirani Benchmark Program - Qiskit
"""

import sys
import time
import random

import numpy as np

'''
sys.path[1:1] = [ "_common", "_common/qiskit" ]
#sys.path[1:1] = [ "../../_common", "../../_common/qiskit" ]
sys.path[1:1] = [ "../_common", "../_common/qiskit" ]
'''

sys.path[1:1] = [ "cudaq" ]
sys.path[1:1] = [ "_common", "_common/cudaq" ]
sys.path[1:1] = [ "../_common", "../_common/cudaq" ]


import execute as ex
import metrics as metrics

from bv_kernel import BersteinVazirani

# Benchmark Name
benchmark_name = "Bernstein-Vazirani Top"

np.random.seed(0)

verbose = True

# Variable for number of resets to perform after mid circuit measurements
num_resets = 1

# saved circuits for display
QC_ = None
Uf_ = None

# Routine to generate random oracle bitstring for execution
def random_bits(length: int):
    bitset = []
    for _ in range(length):
        bitset.append(random.randint(0, 1))
    return bitset
    
# Routine to generate random oracle bitstring for execution
def str_to_ivec(s_int: str):
    bitset = []
    for i, c in s_int:
        print(f"... {i} = {c}")
        bitset.append(random.randint(0, 1))
    return bitset

############### Result Data Analysis

# Analyze and print measured results
# Expected result is always the secret_int, so fidelity calc is simple
def analyze_and_print_result (qc, result, num_qubits, secret_int, num_shots):
    
    # size of input is one less than available qubits
    input_size = num_qubits - 1
    
    # obtain counts from the result object
    counts = result.get_counts(qc)
    if verbose: print(f"For secret int {secret_int} measured: {counts}")
    
    # create the key that is expected to have all the measurements (for this circuit)
    key = format(secret_int, f"0{input_size}b")
    
    # correct distribution is measuring the key 100% of the time
    correct_dist = {key: 1.0}
    
    print(key)

    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)
    
    print(fidelity)
    
    return counts, fidelity

################ Benchmark Loop

# Execute program with default parameters
def run (min_qubits=3, max_qubits=4, skip_qubits=1, max_circuits=3, num_shots=100,
        backend_id='qasm_simulator', method=1, input_value=None,
        provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None):

    print(f"{benchmark_name} ({method}) Benchmark Program - Qiskit")

    # validate parameters (smallest circuit is 3 qubits)
    max_qubits = max(3, max_qubits)
    min_qubits = min(max(3, min_qubits), max_qubits)
    skip_qubits = max(1, skip_qubits)
    #print(f"min, max qubits = {min_qubits} {max_qubits}")

    # create context identifier
    if context is None: context = f"{benchmark_name} ({method}) Benchmark"
    
    ##########
    
    # Variable for new qubit group ordering if using mid_circuit measurements
    mid_circuit_qubit_group = []

    # If using mid_circuit measurements, set transform qubit group to true
    transform_qubit_group = True if method ==2 else False
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler (qc, result, num_qubits, s_int, num_shots):  
     
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
        metrics.store_metric(num_qubits, s_int, 'fidelity', fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)

    # for noiseless simulation, set noise model to be None
    # ex.set_noise_model(None)

    ##########
    
    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):
    
        input_size = num_qubits - 1
        
        # determine number of circuits to execute for this group
        num_circuits = min(2**(input_size), max_circuits)
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        
        # determine range of secret strings to loop over
        if 2**(input_size) <= max_circuits:
            s_range = list(range(num_circuits))
        else:
            s_range = np.random.choice(2**(input_size), num_circuits, False)

        # loop over limited # of secret strings for this
        for s_int in s_range:
        
            # if user specifies input_value, use it instead
            # DEVNOTE: if max_circuits used, this will generate multiple bars per width
            if input_value is not None:
                s_int = input_value
                
            # If mid circuit, then add 2 to new qubit group since the circuit only uses 2 qubits
            if method == 2:
                mid_circuit_qubit_group.append(2)

            # create the circuit for given qubit size and secret string, store time metric
            ts = time.time()
            
            print(s_int)
            # perform CX for each qubit that matches a bit in secret string
            s = ('{0:0' + str(input_size) + 'b}').format(s_int)
            bitset = []
            print(s)
            for i in range(input_size):
                ###print(f"... {i} = {s[i]}")
                if s[input_size - 1 - i] == '1':
                    bitset.append(1)
                else:
                    bitset.append(0)
            
            print(bitset)
            #str_to_ivec(s_int)
            
            #qc = BersteinVazirani(num_qubits, bitset, method)
            qc = [BersteinVazirani, num_qubits, bitset, method]
            '''
            if qc:
                print(f"got QC")
            else:
                print("no QC")
            '''
            # save smaller circuit example for display
            global QC_
            if QC_ == None or num_qubits <= 6:
                if num_qubits < 9: QC_ = qc
        
            metrics.store_metric(num_qubits, s_int, 'create_time', time.time()-ts)

            # collapse the sub-circuit levels used in this benchmark (for qiskit)
            '''
            qc2 = qc.decompose()
            '''
            qc2 = qc
            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc2, num_qubits, s_int, shots=num_shots)
         
        # execute all circuits for this group, aggregate and report metrics when complete
        '''
        ADDED THIS
        '''
        print("about to execute quantum circuit")
        ex.execute_circuits()
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)  
        
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)
    
    
    ##########
    
    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    if method == 1: print("\nQuantum Oracle 'Uf' ="); print(Uf_ if Uf_ != None else " ... too large!")

    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - {benchmark_name} ({method}) - Qiskit",
                         transform_qubit_group = transform_qubit_group, new_qubit_group = mid_circuit_qubit_group)

# if main, execute method
if __name__ == '__main__': run(min_qubits=6, max_qubits=6, max_circuits=1, num_shots = 100)
   
