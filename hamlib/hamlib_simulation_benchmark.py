'''
Hamiltonian Simulation Benchmark Program - Qiskit
(C) Quantum Economic Development Consortium (QED-C) 2024.

This program benchmarks Hamiltonian simulation using Qiskit. 
The central function is the `run()` method, which orchestrates the entire benchmarking process.

HamiltonianSimulation forms the trotterized circuit used in the benchmark.

HamiltonianSimulationExact runs a classical calculation that 
perfectly simulates hamiltonian evolution, although it does not scale well. 
'''

import json
import os
import sys
import time
import math
import numpy as np
from typing import Dict, Optional   # for backwards compat <= py 3.10
from typing import Union, List, Tuple

sys.path[1:1] = ["_common"]

import evolution_exact
import metric_plots

############### Configure API
#
## DEVNOTE: This functiion may be more complicated than is needed; simplify if possible
## It is basically used to perform imports relative to the existing path and using subdirectories
## based on the "api" on which the benchmark executes.

# Configure the QED-C Benchmark package for use with the given API

api_ = "qiskit" 

def qedc_benchmarks_init(api: str = "qiskit"):

    global api_
    if api == None: api = "qiskit"
    api_ = api

    current_dir = os.path.dirname(os.path.abspath(__file__))
    down_dir = os.path.abspath(os.path.join(current_dir, f"{api}"))
    sys.path = [down_dir] + [p for p in sys.path if p != down_dir]

    up_dir = os.path.abspath(os.path.join(current_dir, ".."))
    common_dir = os.path.abspath(os.path.join(up_dir, "_common"))
    sys.path = [common_dir] + [p for p in sys.path if p != common_dir]
    
    api_dir = os.path.abspath(os.path.join(common_dir, f"{api}"))
    sys.path = [api_dir] + [p for p in sys.path if p != api_dir]

    import qcb_mpi as mpi
    globals()["mpi"] = mpi
    mpi.init()

    import execute as ex
    globals()["ex"] = ex

    import metrics as metrics
    globals()["metrics"] = metrics

    import hamlib_utils as hamlib_utils
    globals()["hamlib_utils"] = hamlib_utils
    
    import observables as observables
    globals()["observables"] = observables
    
    import hamlib_simulation_kernel as hamlib_simulation_kernel
    globals()["hamlib_simulation_kernel"] = hamlib_simulation_kernel

    # there must be a better way to do this
    from hamlib_simulation_kernel import HamiltonianSimulation, kernel_draw
    globals()["HamiltonianSimulation"] = HamiltonianSimulation
    globals()["kernel_draw"] = kernel_draw
    
    # this is only needed while testing the old exact evolution functions; remove soon (250125)
    if api == 'qiskit':
        from hamlib_simulation_kernel import initial_state
        globals()["initial_state"] = initial_state
    
    return HamiltonianSimulation, kernel_draw
    

# Benchmark Name
benchmark_name = "Hamiltonian Simulation"

# If set, write each dataset to a json file
save_dataset_file = False

# Maximum # of qubits for which to perform classical exact computation
max_qubits_exact = 16

# Data suffix appended to backend_id when saving data files
data_suffix = ""

np.random.seed(0)

verbose = False

# save sparse_pauli_terms after creation, for use by method 2 comparison
sparse_pauli_terms = None

# contains the correct bitstring for a random pauli circuit
bitstring_dict = {}

# Creates a key for distribution of initial state for method = 3.
def key_from_initial_state(num_qubits, num_shots, init_state, random_pauli_flag):
    """
    Generates a dictionary representing the correct distribution of quantum states based on the initial state configuration.

    This function supports generating specific patterns or distributions for different initial quantum states like
    'checkerboard' and 'ghz'. Depending on the initial state configuration, it may also factor in the effect of random
    Pauli operations applied across the qubits.

    Args:
        num_qubits (int): The number of qubits in the quantum system.
        num_shots (int): The number of measurements or shots to simulate.
        init_state (str): The type of initial state to configure. Supported values are 'checkerboard' and 'ghz'.
        random_pauli_flag (bool): Flag to indicate if random Pauli operations are considered.

    Returns:
        dict: A dictionary where keys are bit strings representing quantum states, and values are the counts
              (or probabilities) of these states occurring.
    """
    def generate_pattern(starting_bit):
        # Generate a bit pattern that alternates, starting from the 'starting_bit'
        pattern = ''.join([str((i + starting_bit) % 2) for i in range(num_qubits)])
        return pattern

    correct_dist = {}

    if init_state == "checkerboard":
        if random_pauli_flag:
            starting_bit = 0 if num_qubits % 2 != 0 else 1
        else:
            starting_bit = 1 if num_qubits % 2 != 0 else 0

        correct_dist[generate_pattern(starting_bit)] = num_shots
    elif init_state == "ghz":
        correct_dist = {
            '0' * num_qubits: num_shots/2,
            '1' * num_qubits: num_shots/2
        }

    return correct_dist


############### Result Data Analysis

#def analyze_and_print_result(qc: QuantumCircuit, result, num_qubits: int,
def analyze_and_print_result(
            qc,
            result,
            num_qubits: int,
            type: str,
            num_shots: int,
            hamiltonian: str,
            method: int,
            t: float,
            random_pauli_flag: bool,
            do_sqrt_fidelity: bool,
            init_state: str
        ) -> tuple:
    """
    Analyze and print the measured results. Compute the quality of the result based on operator expectation for each state.

    Args:
        qc (QuantumCircuit): The quantum circuit.
        result: The result from the execution.
        num_qubits (int): Number of qubits.
        type (str): Type of the simulation (circuit identifier).
        num_shots (int): Number of shots.
        hamiltonian (str): Which hamiltonian to run.
        method (int): Method for fidelity checking (1 for noiseless trotterized quantum, 2 for exact classical), 3 for mirror circuit.

    Returns:
        tuple: Counts and fidelity.
    """
    
    counts = result.get_counts(qc)

    if verbose:
        print_top_measurements(f"For type {type} measured counts = ", counts, 100)

    hamiltonian = hamiltonian.strip().lower()

    # calculate correct distribution on the fly
    
    # for method 1, compute expected dist using ideal quantum simulation of the circuit provided
    if method == 1:
    
        from hamiltonian_simulation_exact import HamiltonianSimulation_Noiseless
        
        if verbose:
            print(f"... begin noiseless simulation for expected distribution for id={type} ...")
            
        ts = time.time()
        correct_dist = HamiltonianSimulation_Noiseless(qc, num_qubits, circuit_id=type, num_shots=num_shots)
        
        if verbose:
            print(f"... noiseless simulation for expected distribution time = {round((time.time() - ts), 3)} sec")
            
    # for method 2, use exact evolution of the Hamiltonian, starting with the initial state circuit
    elif method == 2:
        if verbose:
            print(f"... begin exact computation for id={type} ...")
        
        ts = time.time()
        
        ################ Using the previous evolution_exact code:
        # the plan is to remove this code and use the new once it is validated.
        """
        # create quantum circuit with initial state
        qc_initial = initial_state(n_spins=num_qubits, init_state=init_state)
        
        # apply the Hamiltonian operator to initial state to get expectation/distribution
        correct_exp, correct_dist = evolution_exact.compute_expectation_exact_spo_scipy(
                init_state, 
                qc_initial,
                num_qubits,
                hamlib_simulation_kernel.ensure_sparse_pauli_op(sparse_pauli_terms, num_qubits),
                t        # time (hardocded to match default benchmark)
                )
        """
        ################ Test of the newer evolution_exact code:
        
        correct_exp, correct_dist = evolution_exact.compute_expectation_exact(
                init_state,
                observables.ensure_pauli_terms(sparse_pauli_terms, num_qubits),
                t        # time
                )
        
        # report details if verbose mode
        if verbose:
            print("")
            print(f"... exact computation time = {round((time.time() - ts), 3)} sec")
            print(f"Correct expectation = {correct_exp}")
            #print_top_measurements(f"Correct dist = ", correct_dist, 100)
            print("")
            

    # for method 3, compute expected distribution from the initial state
    elif method == 3: 

        # check simple distribution if not inserting random Paulis 
        if not random_pauli_flag: 
            correct_dist = key_from_initial_state(
                num_qubits, num_shots, init_state, random_pauli_flag
            )

        # if using random paulis, a potentially random bitstring is collected from circuit generation
        else: 
            global bitstring_dict
            correct_bitstring = bitstring_dict[qc.name]
            correct_dist = {correct_bitstring: num_shots}
        
    else:
        raise ValueError("Method is not 1 or 2 or 3, or hamiltonian is not valid.")
        
    if verbose:
        print_top_measurements(f"Correct dist = ", correct_dist, 100)

    # Use polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)
    
    # return the square root of the fidelities if indicated
    if do_sqrt_fidelity:
        fidelity["fidelity"] = math.sqrt(fidelity["fidelity"])
        fidelity["hf_fidelity"] = math.sqrt(fidelity["hf_fidelity"])
    
    if verbose:
        print(f"... fidelity = {fidelity}")
    
    return counts, fidelity
    
def print_top_measurements(label, counts, top_n):
    """
    Prints the top N measurements from a Qiskit measurement counts dictionary
    in a dictionary-like format. If there are more measurements not printed,
    indicates the count.
    
    Args:
        counts (dict): The measurement counts dictionary from Qiskit.
        top_n (int): The number of top measurements to print.
    """
    
    if label is not None: print(label)
    
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    
    total_measurements = len(sorted_counts)
    
    if top_n >= total_measurements:
        top_counts = sorted_counts
        more_counts = []
    else:
        top_counts = sorted_counts[:top_n]
        more_counts = sorted_counts[top_n:]
    
    print("{", end=" ")
    for i, (measurement, count) in enumerate(top_counts):
        print(f"'{measurement}': {np.round(count,6)}", end="")
        if i < len(top_counts) - 1:
            print(",", end=" ")
    
    if more_counts:
        num_more = len(more_counts)
        print(f", ... and {num_more} more.")
    else:
        print(" }")


############### Benchmark Loop

def run(min_qubits: int = 2, 
        max_qubits: int = 8, 
        max_circuits: int = 1,
        skip_qubits: int = 1, 
        num_shots: int = 100,
        hamiltonian: str = "TFIM", 
        hamiltonian_params: dict = None,
        pauli_terms: list = None,
        method: int = 1,
        do_observables = False,
        group_method = None,
        random_pauli_flag: bool = False, 
        random_init_flag: bool = False, 
        use_inverse_flag: bool = False,
        do_sqrt_fidelity: bool = False,
        distribute_shots: bool = False,
        init_state: str = None,
        K: int = None, t: float = None,
        draw_circuits: bool = True,
        plot_results: bool = True,
        backend_id: str = None,
        provider_backend = None,
        hub: str = "", group: str = "", project: str = "",
        exec_options = None,
        context = None,
        api = None):
    """
    Execute program with default parameters.

    Args:
        min_qubits (int): Minimum number of qubits for the simulation. 
                          The smallest circuit is 2 qubits.
        max_qubits (int): Maximum number of qubits for the simulation.
        max_circuits (int): Maximum number of circuits to execute per group.
        skip_qubits (int): Increment of number of qubits between simulations.
        num_shots (int): Number of measurement shots for each circuit execution.
        hamiltonian (str): The type of Hamiltonian to simulate. Default is "tfim".
                           Options include:
                           - "tfim": Transverse Field Ising Model.
                           - "heis": Heisenberg model.
                           - "random_max3sat-hams": Random Max 3-SAT Hamiltonians for binary optimization problems.
                           - "FH_D-1": Fermi-Hubbard model in 1D
                           - "BH_D-1_d-4": Bose-Hubbard model in 1D
        hamiltonian_params (dict): A dictionary of parameters for the given Hamiltonian name.
        method (int): Method for fidelity checking. 
                      Options include:
                      - 1: Noiseless Trotterized Quantum Simulation.
                      - 2: Exact Classical Simulation.
                      - 3: Mirror Circuit Simulation using Sandia Labs' method.
        random_pauli_flag (bool): If True and method is 3, activates random Pauli gates in the circuit.
        random_init_flag (bool): If True, initializes random quantum states. 
                                 Only active if random_pauli_flag is True and method is 3.
        use_inverse_flag (bool): If True, uses the inverse of the quantum circuit rather than the original circuit.
        do_sqrt_fidelity (bool): If True, computes the square root of the fidelity for measurement results.
        init_state (str): Specifies the initial state for the quantum circuit. 
                          If None, a default state is used.
        do_observables (bool): compute observable value from the Hamiltonian  
        group_method (str): Method for generating commuting groups for observable computation. 
                      Options include:
                      - None: no commuting groups used
                      - "simple": simple qubit-wise commuting groups
                      - "N": where N is the "k" in k-communiting groups  
        distribute_shots (bool): with "N" group method, distribute shots weighted by group coefficients                      
        K (int): Number of Trotter steps for the simulation. 
                 This is a crucial parameter for the precision of the Trotterized simulation.
        t (float): Total simulation time. This parameter is used to determine the evolution time for the Hamiltonian.
        draw_circuits : bool, optional
            Draw circuit diagrams only if True. The default is True.
        plot_results : bool, optional
            Plot results only if True. The default is True.
        backend_id (str): Backend identifier for execution on a quantum processor.
        provider_backend: Provider backend instance for advanced execution settings.
        hub (str): IBM Quantum hub identifier. Default is "ibm-q".
        group (str): IBM Quantum group identifier. Default is "open".
        project (str): IBM Quantum project identifier. Default is "main".
        exec_options: Additional execution options, such as optimization levels or custom settings.
        context: Execution context for running the simulation, such as cloud or local settings.
        api: API settings or credentials for accessing quantum computing resources.

    Returns:
        None
    """
    
    # configure the QED-C Benchmark package for use with the given API
    HamilatonianSimulation, kernel_draw = qedc_benchmarks_init(api)
    
    print(f"{benchmark_name} Benchmark Program - Qiskit")
    
    # Create context identifier
    if context is None: context = f"{benchmark_name} Benchmark"
    
    # Validate parameters (smallest circuit is 2 qubits)
    max_qubits = max(2, max_qubits)
    min_qubits = min(max(2, min_qubits), max_qubits)
    #if min_qubits % 2 == 1: min_qubits += 1  # min_qubits must be even (DEVNOTE: is this True? - NO!)
    skip_qubits = max(1, skip_qubits)
    
    hamiltonian_name = hamiltonian
    
    if verbose:
        print(f"... hamiltonian and params = {hamiltonian_name}, {hamiltonian_params}")    
        print(f"... group_method = {group_method}")
    
    # load the HamLib file for the given hamiltonian name
    #if pauli_terms is not None:
    hamlib_utils.load_hamlib_file(filename=hamiltonian_name)

    # assume default init_state if not given
    if init_state == None:
        init_state = "checkerboard"
        
    # Parameters of simulation
    if K is None:
        K = 5
        
    if t is None:
        t = 1.0
    
    ################################
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, type, num_shots):
        # Determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, expectation_a = analyze_and_print_result(
                    qc, result, num_qubits, type, num_shots,
                    hamiltonian,
                    method,
                    t,
                    random_pauli_flag,
                    do_sqrt_fidelity,
                    init_state
                )
        metrics.store_metric(num_qubits, type, 'fidelity', expectation_a)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)
    
    # For CUDA-Q, cannot yet use method 1 as it uses Aer for simulation
    # use method 2 instead
    if api == "cudaq" and method == 1:
        print(f"WARNING: method 1 not supported for {api} API, use method 2 instead")
        method = 2
        
    # Force the name "nvidia" if cudaq used but no backend_id provided:
    # shouldn't have to do this, but needed so we don't get a _data folder called None
    if api == "cudaq" and backend_id == None:
        backend_id = "nvidia"
        
    # DEVNOTE: this is necessary, since we get the Hamiltonian pauli terms when circuit execution is launched.
    # need to wait until it completes since we need to have access to those terms.  They are currently global.
    # Need to fix this
    if method == 2:
        ex.max_jobs_active = 1

    # build list of qubit sizes within the specificed range for which a Hamiltonian is available
    valid_qubits = hamlib_utils.get_valid_qubits(min_qubits, max_qubits, skip_qubits, hamiltonian_params)
    
    if len(valid_qubits) < 1:
        print(f"ERROR: No matching datasets for the requested Hamiltonian name and parameters.")
        print(f"       Terminating this benchmark.")
        return
    
    # metrics storage for observables, until we update the metrics module for use here
    metrics_array = []
    
    for num_qubits in valid_qubits:
        global sparse_pauli_terms
    
        # Reset random seed
        np.random.seed(0)

        # Determine number of circuits to execute for this group
        num_circuits = max(1, max_circuits)
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        
        # use the given Hamiltonian, if provided
        if pauli_terms is not None:
            sparse_pauli_terms = pauli_terms
            dataset_name = "<None>"
            
        # use a sparse Pauli list of terms queried from the open HamLib file
        else:    
            sparse_pauli_terms, dataset_name = hamlib_utils.get_hamlib_sparsepaulilist(num_qubits=num_qubits,
                                                                params=hamiltonian_params)
        print(f"... dataset_name = {dataset_name}")
        if verbose:
            print(f"... hamiltonian_params = \n{hamiltonian_params}")
            print(f"... sparse_pauli_terms = {sparse_pauli_terms}")
      
        num_hamiltonian_terms = len(sparse_pauli_terms)
        print(f"... number of terms in Hamiltonian = {num_hamiltonian_terms}")
        
        if save_dataset_file and mpi.leader():
            save_one_hamlib_dataset(dataset = sparse_pauli_terms, dataset_name = dataset_name)
         
        metrics_object = {}
        metrics_object["group"] = num_qubits
        metrics_object["term_count"] = num_hamiltonian_terms
        metrics_object["group_method"] = group_method
        
        #######################################################################
     
        # in the case of random paulis, method = 3: loop over multiple random pauli circuits
        # otherwise, loop over the same circuit, executing it num_circuits times 
        for circuit_id in range(num_circuits):
            mpi.barrier()
            ts = time.time()
            
            ##############################
            # Observables Grouping Options
            
            # NOTE: the label "group" is used here to mean "commuting groups" 
       
            if do_observables:

                # use this to track how many circuits will be executed
                num_circuits_to_execute = 0
                
                # group options only used in Qiskit version, for now
                if api == None or api == 'qiskit':
                    
                    # if NOT using Estimator
                    if group_method != "estimator":
                    
                        # arrange Hamiltonian terms into groups as specified
                        if group_method == None or group_method == 'simple':
                        
                            # Flag to control optimize by use of commuting groups
                            use_commuting_groups = False
                            if group_method == 'simple':
                                use_commuting_groups = True
                            
                            # group Pauli terms for quantum execution, optionally combining commuting terms into groups.
                            pauli_term_groups, pauli_str_list = observables.group_pauli_terms_for_execution(
                                    num_qubits, sparse_pauli_terms, use_commuting_groups)
                 
                        # arrange terms using k-commuting groups
                        elif group_method != "estimator":
                            # treat "N" specially, converting to num_qubits
                            if group_method == "N":
                                this_group_method = num_qubits
                            else:
                                this_group_method = int(group_method)

                            from generate_pauli_groups import compute_groups
                            pauli_term_groups = compute_groups(
                                            this_group_method, sparse_pauli_terms, 1)
                                            
                            # for each group, create a merged pauli string from all the terms in the group
                            # DEVNOTE: move these 4 lines to a function in observables
                            pauli_str_list = []
                            for group in pauli_term_groups:
                                merged_pauli_str = observables.merge_pauli_terms(group, num_qubits)
                                pauli_str_list.append(merged_pauli_str)
        
                        num_circuits_to_execute = len(pauli_term_groups)
                        
            #######################
            # Base Circuit Creation            
            
            #used to store random pauli correct bitstrings
            global bitstring_dict
            
            # create the HamLibSimulation kernel, random pauli bitstring, from the given Hamiltonian operator
            qc, bitstring = HamiltonianSimulation(
                num_qubits = num_qubits,
                ham_op = sparse_pauli_terms,               
                K = K,
                t = t,         
                init_state = init_state,
                append_measurements = False if do_observables else True,
                method = method, 
                use_inverse_flag = use_inverse_flag,
                random_pauli_flag = random_pauli_flag, 
                random_init_flag = random_init_flag)
                
            # this only works for qiskit circuits
            if "name" in qc:
                bitstring_dict[qc.name] = bitstring
            
            
            ####################################
            # Execution for Fidelity Computation  
            
            # NOTE: the label "group" here mean the "number of qubits" as an index into stored metrics
            
            # execute for fidelity benchmarks
            if not do_observables:
            
                metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time() - ts)

                # Submit circuit for execution on target (simulator, cloud simulator, or hardware)
                ex.submit_circuit(qc, num_qubits, circuit_id, num_shots)
            
            
            ######################################
            # Execution for Observable Computation 
            
            else:               
                if api == None or api == 'qiskit':
                    
                    # if NOT using Estimator
                    if group_method != "estimator":
   
                        # generate an array of circuits, one for each pauli_string in list
                        circuits = hamlib_simulation_kernel.create_circuits_for_pauli_terms(
                                qc, num_qubits, pauli_str_list)
                        
                        if verbose:                 
                            for circuit, group in list(zip(circuits, pauli_term_groups)):
                                print(group)
                                #print(circuit)

                        # call api-specific function to execute circuits
                        if not distribute_shots:
                            print(f"... number of shots per circuit = {int(num_shots / len(circuits))}")
                            # execute the entire list of circuits, same shots each
                            results = execute_circuits(
                                    backend_id = backend_id,
                                    circuits = circuits,
                                    num_shots = int(num_shots / len(circuits))
                                    )
                        else:
                            # execute with shots distributed by weight of coefficients
                            results, pauli_term_groups = execute_circuits_distribute_shots(
                                    backend_id = backend_id,
                                    circuits = circuits,
                                    num_shots = num_shots,
                                    groups = pauli_term_groups
                                    )
                                
                        # Compute the total energy for the Hamiltonian
                        total_energy, term_contributions = observables.calculate_expectation_from_measurements(
                                                                num_qubits, results, pauli_term_groups)
                        total_energy = np.real(total_energy)
                        
                    # if using Qiskit Estimator
                    else:
                        print("... using Qiskit Estimator primitive.")
                        
                        # DEVNOTE: We may want to surface the actual Estimator call instead. 

                        # Ensure that the pauli_terms are in 'full' format, not 'sparse' - convert if necessary
                        est_pauli_terms = observables.ensure_pauli_terms(sparse_pauli_terms, num_qubits=num_qubits)
                        est_pauli_terms = observables.swap_pauli_list(est_pauli_terms)

                        #ts = time.time()

                        # DEVNOTE: backend_id not actually used yet
                        estimator_energy = observables.estimate_expectation_with_estimator(
                                backend_id, qc, est_pauli_terms, num_shots=num_shots)

                        estimator_time = round(time.time()-ts, 3)
                        print(f"... Estimator computation time = {estimator_time} sec")

                        print(f"... Expectation value, computed using Qiskit Estimator: {round(np.real(estimator_energy), 4)}\n")
                         
                        total_energy = estimator_energy
                        term_contributions = None
                
                # special case for CUDA Q Observables
                elif api == "cudaq":
                    if group_method != "SpinOperator":
                        print(f"... executing circuits via sampling, without using CUDA-Q Observe, group_method = {group_method}")
                        # Generate circuits for each Pauli term
                        pauli_term_groups, pauli_str_list = observables.group_pauli_terms_for_execution(
                            num_qubits, sparse_pauli_terms,
                            True if group_method is not None else False
                        )
                        
                        num_circuits_to_execute = len(pauli_term_groups)
                        
                        circuits = hamlib_simulation_kernel.create_circuits_for_pauli_terms(
                            qc, num_qubits, pauli_str_list
                        ) # qc is an array with the kernel and dependent parameters
                        
                        print(f"... number of circuits to execute: {len(circuits)}")

                        if verbose:
                            for circuit, group in zip(circuits, pauli_term_groups):
                                print(group)

                        # Execute circuits
                        if not distribute_shots:
                            print(f"... number of shots per circuit = {int(num_shots / len(circuits))}")
                            results = execute_circuits(
                                backend_id=backend_id,
                                circuits=circuits,
                                num_shots=int(num_shots / len(circuits)),
                            )
                        else:
                            results, pauli_term_groups = execute_circuits_distribute_shots(
                                backend_id=backend_id,
                                circuits=circuits,
                                num_shots=num_shots,
                                groups=pauli_term_groups,
                            )

                        # Compute total energy from measurements
                        total_energy, term_contributions = observables.calculate_expectation_from_measurements(
                            num_qubits, results, pauli_term_groups
                        )
                        total_energy = np.real(total_energy)

                    else:
                        #print("... using CUDA Q Observe.")

                        total_energy = hamlib_simulation_kernel.get_expectation(
                                qc, num_qubits, sparse_pauli_terms)
                                
                        term_contributions = None
                        
                # Record relevant performance metrics
                computed_time = round((time.time() - ts), 3)
                metrics_object["exp_time_computed"] = computed_time

                total_energy = round(total_energy, 4)  
                metrics_object["exp_value_computed"] = total_energy

                metrics_object["num_circuits_to_execute"] = num_circuits_to_execute

                if num_circuits_to_execute > 0:
                    print(f"... number of circuits executed = {num_circuits_to_execute}")

                print(f"... quantum execution time = {computed_time}")
               
                
                ##############################################
                # Compute exact expectation value classically
                
                if num_qubits <= max_qubits_exact:
                
                    if verbose:
                        print(f"... begin exact computation for id={type} ...")
                    
                    ts = time.time()
                    
                    ################ Using newer evolution_exact code:
                    
                    correct_exp, _ = evolution_exact.compute_expectation_exact(
                            init_state,
                            observables.ensure_pauli_terms(sparse_pauli_terms, num_qubits),
                            t        # time
                            )
                            
                    correct_exp = round(correct_exp, 4)                    
                    metrics_object["exp_value_exact"] = correct_exp
                            
                    exact_time = round((time.time() - ts), 3)
                    metrics_object["exp_time_exact"] = exact_time
                    
                    #if verbose:
                    print(f"... exact computation time = {exact_time} sec")
                    
                else:
                    correct_exp = None
                    exact_time = None
                    metrics_object["exp_value_exact"] = correct_exp
                    metrics_object["exp_time_exact"] = exact_time
                
                
                ################
                # Report results 
                
                if correct_exp is not None and correct_exp != 0.0:  
                    simulation_quality = round(total_energy / correct_exp, 3)
                else:
                    simulation_quality = 0.0
                    
                metrics_object["simulation_quality"] = simulation_quality
    
                print("")
                if exact_time is not None:
                    print(f"    Exact expectation value, computed classically: {round(correct_exp, 4)}")
                print(f"    Estimated expectation value, from quantum algorithm: {round(total_energy, 4)}")
                if verbose: print(f"    Term Contributions: {term_contributions}")
                
                #print("") 
                print(f"    ==> Simulation Quality: {simulation_quality}")
                
                print("")
                
                metrics_array.append(metrics_object)
                
                # we want to write file every time we have new data.
                # but if file is busy, we get error, do it at end for now
                ###app_name = f"HamLib-obs-{hamiltonian_name}"
                ###store_app_metrics(app_name, backend_id, metrics_array)
 
 
        ##############################
        # Finalize current Qubit Wdith
                
        # Wait for some active circuits to complete; report metrics when groups complete
        if api != "cudaq" or do_observables == False:
            ex.throttle_execution(metrics.finalize_group)
    
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)
    
    # we want to write file every time we have new data.
    # but if file is busy, we get error, do it at end for now
    app_name = f"HamLib-obs-{hamiltonian_name}"
    if mpi.leader():
        store_app_metrics(app_name, backend_id, metrics_array)

    ########################
    # Display Sample Circuit
    
    if draw_circuits and mpi.leader():
        kernel_draw(hamiltonian, method)
    
    ##########################
    # Display Plots of Results
    
    # Plot metrics for all circuit sizes
    base_ham_name = os.path.basename(hamiltonian)
    if do_observables:
        options = {"ham": base_ham_name,
                "params": hamiltonian_params,
                "method": method,
                "gm": group_method,
                "K": K,
                "t": t,
                "shots": num_shots,
                "reps": max_circuits} 
    else:
        options = {"ham": base_ham_name,
                #"params": hamiltonian_params,
                "method": method,
                #"gm": group_method,
                "K": K,
                "t": t,
                "shots": num_shots,
                "reps": max_circuits} 
    

    if not plot_results:
        return
        
    if mpi.leader() and not do_observables:
        metrics.plot_metrics(f"Benchmark Results - {benchmark_name} - Qiskit", options=options)
    
    if do_observables:
        #plot_results_from_data(**dict_of_inputs)
        
        ############## expectation value plot
        suptitle = f"Benchmark Results - {benchmark_name} ({method}) - {api if api else 'Qiskit'}"
        
        # should not be needed; needs investigation, saving image fails if command line invocation
        # and non-observable case works fine.
        if backend_id is None:
            backend_id = "qasm_simulator"
        
        if mpi.leader():
            plot_from_data(suptitle, metrics_array, backend_id, options)

  
    
########################################
# CUSTOM ADAPTATION OF EXECUTE FUNCTIONS

# This code is provided here to augment the default API/execute functions,
# specifically to enable execution of an array of circuits for observable calculations.
# This code will be moved up into the _common/API/execute methods later (210131).

def execute_circuits(
        backend_id: str = None,
        circuits: list = None,
        num_shots: int = 100
    ) -> list:

    if verbose:
        print(f"... execute_cicuits({backend_id}, {len(circuits)}, {num_shots})")

    if backend_id == None:
        backend_id == "qasm_simulator"

    # if backend_id == "nvidia":
    if api_ == "cudaq":
        counts_array = []
        for circuit in circuits:
            result = ex.execute_circuit_immed(circuit, num_shots)
            counts_array.append(result.get_counts())
        
        if len(counts_array) < 2:
            results = ExecResult(counts_array[0])
        else:
            results = ExecResult(counts_array)

    # Set up the backend for execution
    elif backend_id == "qasm_simulator" or backend_id == "statevector_simulator":
    
        # Initialize simulator backend
        from qiskit_aer import Aer
        if backend_id == "statevector_simulator":
            #backend = Aer.get_backend('statevector_simulator')
            backend = Aer.get_backend('qasm_simulator')
        else:
            backend = Aer.get_backend('qasm_simulator')
            
        #print(f"... backend_id = {backend_id}")
   
        # Execute all of the circuits to obtain array of result objects
        if backend_id != "statevector_simulator" and ex.noise is not None:
            #print("**************** executing with noise")
            noise_model = ex.noise
            
        else:
            noise_model = None
        
        # all circuits get the same number of shots as given 
        results = backend.run(circuits, shots=num_shots, noise_model=noise_model).result()
    
    # handle special case using IBM Runtime Sampler Primitive
    elif ex.sampler is not None:
        print("... using Qiskit Runtime Sampler")
        
        from qiskit import transpile

        # circuits need to be transpiled first, post Qiskit 1.0
        trans_qcs = transpile(circuits, ex.backend)
        
        # execute the circuits using the Sampler Primitive (required for IBM Runtime Qiskit 1.3
        job = ex.sampler.run(trans_qcs, shots=num_shots)
        
        # wrap the Sampler result object's data in a compatible Result object 
        sampler_result = job.result()
        results = BenchmarkResult(sampler_result)
     
    # handle all other backends here
    else:
        print(f"... using Qiskit run() with {backend_id}")
        
        from qiskit import transpile
        
        # DEVNOTE: This line is specific to IonQ Aria-1 simulation; comment out
        # ex.backend.set_options(noise_model="aria-1")
        
        # circuits need to be transpiled first, post Qiskit 1.0
        trans_qcs = transpile(circuits, ex.backend)
        
        # execute the circuits using backend.run()
        job = ex.backend.run(trans_qcs, shots=num_shots)
        
        results = job.result()
          
    return results
        

# class BenchmarkResult is made for Sampler runs. This is because
# qiskit primitive job result instances don't have a get_counts method 
# like backend results do. As such, a get counts method is calculated
# from the quasi distributions and shots taken.
class BenchmarkResult:

    def __init__(self, qiskit_result):
        super().__init__()
        self.qiskit_result = qiskit_result
        self.metadata = qiskit_result.metadata

    def get_counts(self):
        count_array = []
        for result in self.qiskit_result:    
            # convert the quasi distribution bit values to shots distribution
            bitvals = next(iter(result.data.values()))
            counts = bitvals.get_counts()
            count_array.append(counts)
            
        return count_array

# class ExecResult is made for multi-circuit runs. 
class ExecResult(object):

    def __init__(self, counts_array):
        super().__init__()
        #self.qiskit_result = qiskit_result
        #self.metadata = qiskit_result.metadata
        self.counts = counts_array

    def get_counts(self, qc=0):
        # counts= self.qiskit_result.quasi_dists[0].binary_probabilities()
        # for key in counts.keys():
        #     counts[key] = int(counts[key] * self.qiskit_result.metadata[0]['shots'])
        #qc_index = 0 # this should point to the index of the circuit in a pub
        #bitvals = next(iter(self.qiskit_result[qc_index].data.values()))
        #counts = bitvals.get_counts()
        return self.counts
 
 
#########################################
# EXECUTE CIRCUITS WITH DISTRIBUTED SHOTS

# 250302 TL: Leaving these options in until we are sure all works well.
new_way = True
debug = False
    
def execute_circuits_distribute_shots(
        backend_id: str = None,
        circuits: list = None,
        num_shots: int = 100,
        groups: list = None,
        ds_method: str = 'max_sq',
    ) -> list:

    if verbose or debug:
        print(f"... execute_circuits_distribute_shots({backend_id}, {len(circuits)}, {num_shots}, {groups})")
                  
    # distribute shots; obtain total and distribute according to weights
    # (weighting not implemented yet)
    circuit_count = len(circuits)
    total_shots = num_shots         # to match current behavior
    if verbose or debug:
        print(f"... distributing shots, total shots = {total_shots} shots")
    
    # determine the number of shots to execute for each circuit, weighted by largest coefficient
    num_shots_list = get_distributed_shot_counts(total_shots, groups, ds_method)

    if verbose or debug:
        print(f"  ... num_shots_list = {num_shots_list}")  
    
    # The "new" approach that uses bucketing, to reduce number of circuits to be executed
    if new_way:
        if debug:
            print("************* NEW WAY")
            for group in groups:
                print(group)
                   
            # print(f"  in circuits = {circuits}")
        
        # determine optimal bucketing for these circuits, based on distribution of shots needed
        from shot_distribution import bucket_numbers_kmeans, compute_bucket_averages
        
        # get buckets of terms with similar shots counts, and index of original position
        max_buckets = 3 if len(groups) < 50 else 4
        buckets_kmeans, indices_kmeans = bucket_numbers_kmeans(num_shots_list, max_buckets=max_buckets)
        
        # find the average number of shots required for each bucket
        # (sum of all shots for all circuits, nested, should be same as the incoming total)
        bucket_avg_shots = compute_bucket_averages(buckets_kmeans)
        
        if debug:
            print('  ... bucket kmeans:', buckets_kmeans)
            print('  ... indices_kmeans:', indices_kmeans)
            print('  ... bucket_avg_shots:', bucket_avg_shots)
        
        circuit_list = [[circuits[idx] for idx in indices] for indices in indices_kmeans]   
        group_list = [[groups[idx] for idx in indices] for indices in indices_kmeans]
        
        if debug:
#             print(f"  circuit_list after bucketing = {circuit_list}")
#             print(f"  ... group_list after bucketing = {group_list}") 
            print(f"  ... group_list after bucketing....") 
            for group in group_list:
                for g in group:
                    print(g)
                print('-----')

        
        if verbose or debug:
            print(f"  ... bucketed shots list after bucketing = {buckets_kmeans} avg = {bucket_avg_shots}")
                
        counts_array = []
        #for circuit in circuits:
        for circuits, num_shots in zip(circuit_list, bucket_avg_shots):
        
            if debug:
                # print(f"  ...    cccc = {circuits}")
                print(f"... len circs = {len(circuits)}")
            
            # execute this list of circuits, same shots each
            results = execute_circuits(
                    backend_id = backend_id,
                    #circuits = [circuit],
                    circuits = circuits,
                    num_shots = num_shots
                    )
            
            # Qiskit returns and array if array executed, but single counts for one circuit
            if len(circuits) > 1:                           
                counts = results.get_counts()
            else:
                counts = [results.get_counts()]
                
            for counts2 in counts:
                counts_array.append(counts2)

        # similarly, construct a Result object with counts structure to match circuits
        if len(counts_array) < 2:
            results = ExecResult(counts_array[0])
        else:
            results = ExecResult(counts_array)
             
        group_list = [item for sublist in group_list for item in sublist]
        
        if debug:
            print(f"... results.get_counts() = {results.get_counts()}")
            print(f"... results.get_counts() ({len(results.get_counts())}) = {results.get_counts()}")
            print(f"... group_list len = {len(group_list)}")
            print(f"  ... group_list after subgroups = ")
            for group in group_list:
                print(group)

        return results, group_list
    
    # The "old" approach that executes every circuit, but with weighted num shots
    else:
        if debug:
            print("************* OLD WAY")
            
        counts_array = []
        for circuit, num_shots in zip(circuits, num_shots_list):
            
            # execute this list of circuits, same shots each
            results = execute_circuits(
                    backend_id = backend_id,
                    circuits = [circuit],
                    num_shots = num_shots
                    )
                                           
            counts = results.get_counts()
            counts_array.append(counts)
        
        if len(circuits) < 2:
            results = ExecResult(counts)
        else:
            results = ExecResult(counts_array)
        
        if debug:        
            print(f"... results.get_counts() ({len(results.get_counts())}) = {results.get_counts()}")
            print(f"... groups len = {len(groups)}")
            
        return results, groups

    
# From the given list of term groups, distribute the total shot count, returning num_shots by group
def get_distributed_shot_counts(
        num_shots: int = 100,
        groups: list = None,
        ds_method: str = 'max_sq',
    ) -> List:
    
#     # loop over all groups, to find the largest coefficient in each group
#     max_weights = []
#     for group in groups:
#         #print(group)
#         max_weight = 0
#         for pauli, coeff in group:
#             #print(f"  ... coeff = {coeff}")
#             max_weight = max(max_weight, np.real(abs(coeff)))
            
#         max_weights.append(max_weight)

    # loop over all groups, to find the sum of coefficient in each group
#     norm_weights = []
#     for group in groups:
#         #print(group)
#         sum_weight = 0
#         for pauli, coeff in group:
#             #print(f"  ... coeff = {coeff}")
#             sum_weight += np.real(abs(coeff))
            
#         norm_weights.append(sum_weight/len(group))

        
# #     # compute a normalized distribution over all groups
# #     total_weights = sum(max_weights)
# #     max_weights_normalized = [max_weight / total_weights for max_weight in max_weights]
    
# #     # compute shots counts based on these weights
# #     num_shots_list = [int(mwn * num_shots) for mwn in max_weights_normalized]
     
#     # compute a normalized distribution over all groups
#     total_weights = sum(norm_weights)
#     norm_weights_normalized = [norm_weight / total_weights for norm_weight in norm_weights]
    
#     # compute shots counts based on these weights
#     num_shots_list = [int(mwn * num_shots) for mwn in norm_weights_normalized]
    
    
    # add shots to first group until the total is same as the given total shot count
#     one_norm_weights = [sum(abs(coeff) for pauli, coeff in group) / len(group) for group in groups]
    weights = []
    for group in groups:
        w_sqs =  [abs(coeff)**2 for pauli, coeff in group] 
        ws = [abs(coeff) for pauli, coeff in group]
    
        if ds_method == 'max_sq':
            weights.append(max(w_sqs))
        elif ds_method == 'mean_sq':
            weights.append(sum(w_sqs)/len(w_sqs))
        elif ds_method == 'max':
            weights.append(max(ws))
        else:
            weights.append(sum(ws)/len(ws))
        
    # Step 2: Normalize weights to compute shot proportions
    total_weight = sum(weights)
    shot_allocations = [int((w / total_weight) * num_shots) for w in weights]

    # Step 3: Adjust to ensure total shots match exactly
    while sum(shot_allocations) < num_shots:
        max_index = np.argmax(shot_allocations)
        shot_allocations[max_index] += 1
    # print('shot allocation:', shot_allocations)
       
    return shot_allocations
 
 
########################################
# UTILITY FUNCTIONS (TEMPORARY)

# The functions included in this section will be moved/merged with other lower-level functions.
# These have been developed iteratively up in the example notebooks and the code is being incrementally 
# moved down to lower modules.  For now, these functions are shared by several of the demo notebooks
# as they are being developed.

def find_pauli_groups(num_qubits, sparse_pauli_terms, group_method, k=None):
    """
    Group the Pauli terms accourding to the given group method: "None", "simple", "N"
    """
    # have to do this here, due to logic of the "api" code; improve these imports later 
    import observables
    
    ### print(f"... using group method: {group_method}")

    mpi.barrier()
    ts = time.time()
    
    # use no grouping or the most basic method "simple"
    if group_method == None or group_method == "simple":
    
        # Flag to control optimize by use of commuting groups
        use_commuting_groups = False
        if group_method == "simple":
            use_commuting_groups = True
    
        # group Pauli terms for quantum execution, optionally combining commuting terms into groups.
        pauli_term_groups, pauli_str_list = observables.group_pauli_terms_for_execution(
                num_qubits, sparse_pauli_terms, use_commuting_groups)
    
    # use k-commuting algorithm
    else:
        from generate_pauli_groups import compute_groups
        pauli_term_groups = compute_groups(num_qubits, sparse_pauli_terms, k)
    
    #print(f"\n... Number of groups created: {len(pauli_term_groups)}")
    #print(f"... Pauli Term Groups:")
    #for group in pauli_term_groups:
        #print(group)
    
    group_time = round(time.time()-ts, 3)
    #print(f"\n... finished grouping terms, total grouping time = {group_time} sec.\n")
    
    # for each group, create a merged pauli string from all the terms in the group
    # DEVNOTE: move these 4 lines to a function in observables
    pauli_str_list = []
    for group in pauli_term_groups:
        merged_pauli_str = observables.merge_pauli_terms(group, num_qubits)
        pauli_str_list.append(merged_pauli_str)
    
    #print(f"\n... Merged Pauli strings, one per group:\n  {pauli_str_list}\n")

    return pauli_term_groups, pauli_str_list



#######################
# DATA FILE FUNCTIONS  
  
# This code needs to be moved or merged with code in _common/metrics.py (250202)

# It is also inefficient in that it is loading the existing metrics from file every time
# We should load only at start of the benchmark run
# To avoid this, for now we are writing the data only at the end of the whole benchmark 
  
# Save the application metrics data to a shared file for the current device
def store_app_metrics (app_name, backend_id, metrics_array):
    # print(f"... storing metrics for {app_name} executed on {backend_id}")
    
    # don't leave slashes in the filename
    if backend_id is not None: backend_id = backend_id.replace("/", "_")
    app_name = app_name.replace("/", "_")
    
    ##### load content of exising data file and merge new metrics_array into it
    
    # load the current data file for this app
    current_metrics_array = load_app_metrics(app_name, backend_id)
    
    # Convert list to a dictionary using multiple keys
    merged_dict = {get_key(d): d for d in current_metrics_array}

    # Update existing records or add new ones
    merged_dict.update({get_key(d): d for d in metrics_array})

    # Convert back to a list
    metrics_array = list(merged_dict.values())

    ##### Write out the merged metrics_array to the data file

    # be sure we have a __data directory and backend_id directory under it
    if not os.path.exists("__data"): os.makedirs("__data")
    if not os.path.exists(f"__data/{backend_id}{data_suffix}"):
        os.makedirs(f"__data/{backend_id}{data_suffix}")
    
    # create filename based on the backend_id and optional data_suffix
    filename = f"__data/{backend_id}{data_suffix}/{app_name}.json"

    # overwrite the existing file with the merged data
    with open(filename, "w") as f:
        json.dump(metrics_array, f, indent=2, sort_keys=True)
        f.close()

# Load the application metrics from the given data file
# Returns a dict containing the metrics
def load_app_metrics (app_name, backend_id):
    # print(f"... load metrics for {app_name} on {backend_id}")

    # don't leave slashes in the filename
    if backend_id is not None: backend_id = backend_id.replace("/", "_")
    app_name = app_name.replace("/", "_")
    
    # create filename based on the backend_id and optional data_suffix
    #ilename = f"__data/DATA-{backend_id}{data_suffix}.json"
    filename = f"__data/{backend_id}{data_suffix}/{app_name}.json"
  
    metrics_array = None
    
    # attempt to load metrics data from file
    if os.path.exists(filename) and os.path.isfile(filename):
        with open(filename, 'r') as f:
            
            # attempt to load shared_data dict as json
            try:
                metrics_array = json.load(f)
                
            except:
                pass
            
    # create empty shared_data dict if not read from file
    if metrics_array == None:
        print(f"WARNING: cannot load metrics file: {filename}")
        metrics_array = []
        
    return metrics_array

# Define a function to create a unique key using multiple keys
def get_key(d):
    return (d["group"], d["group_method"])  # Unique key based on id and type

def query_dict_array(data, query):
    """
    Filters an array of dictionaries based on a given query dictionary.

    Args:
        data (list of dict): The dataset to search.
        query (dict): A dictionary of key-value pairs to match.

    Returns:
        list: A list of dictionaries that match all query criteria.
    """
    return [row for row in data if all(row.get(k) == v for k, v in query.items())]   

def save_one_hamlib_dataset(dataset: list, dataset_name: str):
    
    # create filename based on the dataset_name
    filename = f"{dataset_name}.json"
    
    dataset = convert_coefficients_to_real(dataset)
    
    # overwrite the existing file with the merged data
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=2)
        f.close()

def convert_coefficients_to_real(pauli_terms):
    """
    Convert complex coefficients to their real parts in a list of Pauli terms.
    
    Parameters:
        pauli_terms (list of tuples): [(pauli_dict, complex_coefficient), ...]
    
    Returns:
        list of tuples: [(pauli_dict, real_coefficient), ...]
    """
    return [(pauli_dict, coeff.real) for pauli_dict, coeff in pauli_terms]


       
################################################
# PLOT METHODS

# %% Loading saved data (from json files)

def load_data_and_plot(folder=None, backend_id=None, **kwargs):
    """
    The highest level function for loading stored data from a previous run
    and plotting optgaps and area metrics

    Parameters
    ----------
    folder : string
        Directory where json files are saved.
    """
    _gen_prop = load_all_metrics(folder, backend_id=backend_id)
    if _gen_prop is not None:
        gen_prop = {**_gen_prop, **kwargs}
        plot_results_from_data(**gen_prop)
        
def plot_results_from_data(
    ):
    
    pass
    
    
def plot_from_data(suptitle: str, metrics_array: list, backend_id: str, options):

    # extract data arrays metrics_array for plotting 
    groups = [m["group"] for m in metrics_array]
    exp_values_computed = [m["exp_value_computed"] for m in metrics_array]
    exp_values_exact = [m["exp_value_exact"] for m in metrics_array]
    exp_times_computed = [m["exp_time_computed"] for m in metrics_array]
    exp_times_exact = [m["exp_time_exact"] for m in metrics_array]
    
    # remove None values from some arrays
    exp_times_computed = [x for x in exp_times_computed if x is not None]
    exp_times_exact = [x for x in exp_times_exact if x is not None]
   

    # plot all line metrics, including solution quality and accuracy ratio
    # vs iteration count and cumulative execution time
    metric_plots.plot_expectation_value_metrics(
        suptitle,
        backend_id=backend_id,
        options=options,
        
        groups=groups,
        expectation_values_exact=exp_values_exact,
        expectation_values_computed=exp_values_computed,   
    )
    
    # expectation time plot
    # plot all line metrics, including solution quality and accuracy ratio
    # vs iteration count and cumulative execution time
    metric_plots.plot_expectation_time_metrics(
        suptitle,
        backend_id=backend_id,
        options=options,
        
        groups=groups,
        expectation_times_exact=exp_times_exact,
        expectation_times_computed=exp_times_computed,
    )
 

 
#######################
# MAIN

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Bernstei-Vazirani Benchmark")
    parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    parser.add_argument("--target", "-t", default=None, help="Target Backend", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits (min = max = N)", type=int)
    parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=1, help="Maximum circuit repetitions", type=int) 
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument("--hamiltonian", "-ham", default="TFIM", help="Name of Hamiltonian", type=str)
    parser.add_argument("--parameters", "-params", default=None, help="Hamiltonian parameters, e.g 'enc:bk,h:2'")
    parser.add_argument("--num_steps", "-steps", default=None, help="Number of Trotter steps", type=int)
    parser.add_argument("--time", "-time", default=None, help="Time of evolution", type=float)
    parser.add_argument("--do_observables", "-obs", action="store_true", help="Compute observable values")
    parser.add_argument("--group_method", "-gm", default=None, help="Method for creating commuting groups, e.g. 'simple','1','2', 'N'")   
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--use_inverse_flag", "-inverse", action="store_true", help="Use inverse evolution")
    parser.add_argument("--do_sqrt_fidelity", "-sqrt", action="store_true", help="Return square root of fidelities")
    parser.add_argument("--random_pauli_flag", "-ranp", action="store_true", help="Gen random paulis")
    parser.add_argument("--random_init_flag", "-rani", action="store_true", help="Gen random initialization")
    parser.add_argument("--init_state", "-init", default=None, help="initial state", type=str)  
    parser.add_argument("--noplot", "-nop", action="store_true", help="Do not plot results")
    parser.add_argument("--nodraw", "-nod", action="store_true", help="Do not draw circuit diagram")
    parser.add_argument("--data_suffix", "-suffix", default=None, help="Suffix appended to data file name", type=str)
    parser.add_argument("--profile", "-prof", action="store_true", help="Profile with cProfile") 
    return parser.parse_args()
    
def parse_name_value_pairs(input_string: str) -> Dict[str, str]:
    """
    Parses a string of name-value pairs separated by colons and commas.

    Args:
        input_string (str): Input string, e.g., "name1:value1,name2:,name3:value3".

    Returns:
        dict[str, str]: Dictionary of name, value entries. If the value is missing, it defaults to an empty string.
    """
    pairs = input_string.split(",")  # Split into individual pairs
    result = {}
    for pair in pairs:
        if ":" in pair:
            name, value = pair.split(":", 1)  # Split name and value
            result[name] = value
        else:
            # If there's no colon, the value is empty
            result[name] = ''
    return result

def do_run(args):

    # execute benchmark program
    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        method=args.method,
        hamiltonian=args.hamiltonian,
        hamiltonian_params=hamiltonian_params,
        do_observables=args.do_observables,
        group_method=args.group_method,
        random_pauli_flag=args.random_pauli_flag,
        random_init_flag=args.random_init_flag,
        use_inverse_flag=args.use_inverse_flag,
        do_sqrt_fidelity=args.do_sqrt_fidelity,
        init_state = args.init_state,
        K = args.num_steps,
        t = args.time,
        #theta=args.theta,
        plot_results=not args.noplot,
        draw_circuits=not args.nodraw,
        backend_id=args.backend_id,
        exec_options = {"noise_model" : None} if args.nonoise else {},
        api=args.api
        )

import cProfile

# if main, execute method
if __name__ == '__main__':   
    args = get_args()
    hamiltonian_params = None
    if args.parameters is not None:
        hamiltonian_params = parse_name_value_pairs(args.parameters)
    
    # configure the QED-C Benchmark package for use with the given API
    # (done here so we can set verbose for now)
    HamiltonianSimulation, kernel_draw = qedc_benchmarks_init(args.api)
    
    # special argument handling
    ex.verbose = args.verbose
    verbose = args.verbose
    hamlib_simulation_kernel.verbose = args.verbose
    hamlib_utils.verbose = args.verbose
    
    if args.data_suffix is not None:
        metrics.data_suffix = args.data_suffix
    
    if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits
    
    # Execute the benchmark, with profiling if requested
    if args.profile:
        print("\n... running benchmark with cProfile for performance profiling ...\n")
        cProfile.run('do_run(args)', sort='cumtime')
    else:
        do_run(args)
        
    

