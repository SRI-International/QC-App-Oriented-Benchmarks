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

sys.path[1:1] = ["_common"]

from hamiltonian_simulation_exact import HamiltonianSimulationExact, HamiltonianSimulation_Noiseless

import evolution_exact

############### Configure API
#
## DEVNOTE: This functiion may be more complicated than is needed; simplify if possible
## It is basically used to perform imports relative to the existing path and using subdirectories
## based on the "api" on which the benchmark executes.

# Configure the QED-C Benchmark package for use with the given API
def qedc_benchmarks_init(api: str = "qiskit"):

    if api == None: api = "qiskit"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    down_dir = os.path.abspath(os.path.join(current_dir, f"{api}"))
    sys.path = [down_dir] + [p for p in sys.path if p != down_dir]

    up_dir = os.path.abspath(os.path.join(current_dir, ".."))
    common_dir = os.path.abspath(os.path.join(up_dir, "_common"))
    sys.path = [common_dir] + [p for p in sys.path if p != common_dir]
    
    api_dir = os.path.abspath(os.path.join(common_dir, f"{api}"))
    sys.path = [api_dir] + [p for p in sys.path if p != api_dir]

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
    from hamlib_simulation_kernel import HamiltonianSimulation, kernel_draw, initial_state
    globals()["HamiltonianSimulation"] = HamiltonianSimulation
    globals()["kernel_draw"] = kernel_draw
    globals()["initial_state"] = initial_state
    
    return HamiltonianSimulation, kernel_draw
    

# Benchmark Name
benchmark_name = "Hamiltonian Simulation"

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
            random_pauli_flag: bool,
            do_sqrt_fidelity: bool,
            init_state: str) -> tuple:
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
        
        # DEVNOTE: ideally, we can remove this next line somehow      
        # create quantum circuit with initial state
        qc_initial = initial_state(n_spins=num_qubits, init_state=init_state)
        """                  
        # compute the expected  distribution after exact evolution
        #correct_dist = HamiltonianSimulationExact(qc_initial, n_spins=num_qubits,
        correct_dist, correct_exp = HamiltonianSimulationExact(qc_initial, n_spins=num_qubits,
                hamiltonian_op=hamlib_simulation_kernel.ensure_sparse_pauli_op(sparse_pauli_terms, num_qubits),
                time=1.0)
                
        # DEVNOTE: the following code is WIP ... 
        # it is for testing the new versions of exact expectation and distribution calculations.
        # The compute_expectation_exact_spo_scipy version is resulting in slightly lower values for fidelity.
        # while the new compute_expectation_exact version is way too slow.  Still debugging.
        """        
        correct_exp, correct_dist = evolution_exact.compute_expectation_exact_spo_scipy(
                init_state, 
                qc_initial,
                num_qubits,
                hamlib_simulation_kernel.ensure_sparse_pauli_op(sparse_pauli_terms, num_qubits),
                1.0            # time
                )
            
        if verbose:
            print(f"... exact computation time = {round((time.time() - ts), 3)} sec")
        print(f"... exact computation time = {round((time.time() - ts), 3)} sec")
        
        #print_top_measurements(f"Correct dist = ", correct_dist, 100)
        #print(f"Expectation = {correct_exp}")
        
        ###### Test of the newer evolution_exact code:
        """
        ts = time.time()
        
        expectation, distribution = evolution_exact.compute_expectation_exact(
                init_state,
                observables.ensure_pauli_terms(sparse_pauli_terms, num_qubits),
                1.0            # time
                )
        
        if verbose:
            print(f"... exact computation time (2) = {round((time.time() - ts), 3)} sec")  
            
        print(f"... exact computation time (2) = {round((time.time() - ts), 3)} sec")
        
        #print_top_measurements(f"Correct dist (2) = ", distribution, 100)
        print(f"Expectation (2) = {expectation}")
        
        #correct_exp = expectation
        #correct_dist = distribution
        
        """

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
        print(f"'{measurement}': {round(count,6)}", end="")
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
        method: int = 1,
        do_observables = False,
        random_pauli_flag: bool = False, 
        random_init_flag: bool = False, 
        use_inverse_flag: bool = False,
        do_sqrt_fidelity: bool = False,
        init_state: str = None,
        K: int = None, t: float = None,
        backend_id: str = None, provider_backend = None,
        hub: str = "ibm-q", group: str = "open", project: str = "main", exec_options = None,
        context = None, api = None):
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
        K (int): Number of Trotter steps for the simulation. 
                 This is a crucial parameter for the precision of the Trotterized simulation.
        t (float): Total simulation time. This parameter is used to determine the evolution time for the Hamiltonian.
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
    if min_qubits % 2 == 1: min_qubits += 1  # min_qubits must be even (DEVNOTE: is this True?)
    skip_qubits = max(1, skip_qubits)
    
    hamiltonian_name = hamiltonian
    
    if verbose:
        print(f"... hamiltonian and params = {hamiltonian_name}, {hamiltonian_params}")
    
    # load the HamLib file for the given hamiltonian name
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
        counts, expectation_a = analyze_and_print_result(qc, result, num_qubits, type, num_shots, hamiltonian, method, random_pauli_flag, do_sqrt_fidelity, init_state)
        metrics.store_metric(num_qubits, type, 'fidelity', expectation_a)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)
    
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
    
    for num_qubits in valid_qubits:
        global sparse_pauli_terms
    
        # Reset random seed
        np.random.seed(0)

        # Determine number of circuits to execute for this group
        num_circuits = max(1, max_circuits)
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        
        # return a sparse Pauli list of terms queried from the open HamLib file
        sparse_pauli_terms, dataset_name = hamlib_utils.get_hamlib_sparsepaulilist(num_qubits=num_qubits,
                                                                params=hamiltonian_params)
        print(f"... dataset_name = {dataset_name}")
        if verbose:
            print(f"... hamiltonian_params = \n{hamiltonian_params}")
            print(f"... sparse_pauli_terms = {sparse_pauli_terms}")

        #######################################################################

        # in the case of random paulis, method = 3: loop over multiple random pauli circuits
        # otherwise, loop over the same circuit, executing it num_circuits times 
        for circuit_id in range(num_circuits):

            ts = time.time()
            
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
               
            bitstring_dict[qc.name] = bitstring
            
            if not do_observables:
            
                metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time() - ts)

                # Submit circuit for execution on target (simulator, cloud simulator, or hardware)
                ex.submit_circuit(qc, num_qubits, circuit_id, num_shots)
            else:
        
                # Flag to control optimize by use of commuting groups
                use_commuting_groups = True

                # group Pauli terms for quantum execution, optionally combining commuting terms into groups.
                pauli_term_groups, pauli_str_list = observables.group_pauli_terms_for_execution(
                        num_qubits, sparse_pauli_terms, use_commuting_groups)

                # generate an array of circuits, one for each pauli_string in list
                circuits = hamlib_simulation_kernel.create_circuits_for_pauli_terms(qc, num_qubits, pauli_str_list)
                
                if verbose:                 
                    for circuit, group in list(zip(circuits, pauli_term_groups)):
                        print(group)
                        #print(circuit)
                  
                # Initialize simulator backend
                from qiskit_aer import Aer
                backend = Aer.get_backend('qasm_simulator')
               
                # Execute all of the circuits to obtain array of result objects
                results = backend.run(circuits, num_shots=num_shots).result()
                
                # Compute the total energy for the Hamiltonian
                total_energy, term_contributions = observables.calculate_expectation_from_measurements(
                                                            num_qubits, results, pauli_term_groups)
                print("************")
                print(f"... total execution time = {round(time.time()-ts, 3)}")
                print(f"Total Energy: {total_energy}")
                print(f"Term Contributions: {term_contributions}")
                print("")
                
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
    
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    ##########
    
    # draw a sample circuit
    kernel_draw(hamiltonian, method)
       
    # Plot metrics for all circuit sizes
    base_ham_name = os.path.basename(hamiltonian)
    options = {"ham": base_ham_name, "method":method, "shots": num_shots, "reps": max_circuits}   
    metrics.plot_metrics(f"Benchmark Results - {benchmark_name} - Qiskit", options=options)


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
    parser.add_argument("--hamiltonian", "-ham", default="TFIM", help="Name of Hamiltonian", type=str)
    parser.add_argument("--do_observables", "-obs", action="store_true", help="Compute observable values")
    parser.add_argument("--parameters", "-params", default=None, help="Hamiltonian parameters, e.g 'enc:bk,h:2'")  
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument("--data_suffix", "-suffix", default=None, help="Data File Suffix", type=str)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--num_steps", "-steps", default=None, help="Number of Trotter steps", type=int)
    parser.add_argument("--time", "-time", default=None, help="Time of evolution", type=float)
    parser.add_argument("--use_inverse_flag", "-inverse", action="store_true", help="Use inverse evolution")
    parser.add_argument("--do_sqrt_fidelity", "-sqrt", action="store_true", help="Return square root of fidelities")
    parser.add_argument("--random_pauli_flag", "-ranp", action="store_true", help="Gen random paulis")
    parser.add_argument("--random_init_flag", "-rani", action="store_true", help="Gen random initialization")
    parser.add_argument("--init_state", "-init", default=None, help="initial state")  
    parser.add_argument("--profile", "-prof", action="store_true", help="Profile with cProfile")    
    return parser.parse_args()
    
def parse_name_value_pairs(input_string: str) -> dict[str, str]:
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
        hamiltonian=args.hamiltonian,
        hamiltonian_params=hamiltonian_params,
        do_observables=args.do_observables,
        method=args.method,
        random_pauli_flag=args.random_pauli_flag,
        random_init_flag=args.random_init_flag,
        use_inverse_flag=args.use_inverse_flag,
        do_sqrt_fidelity=args.do_sqrt_fidelity,
        init_state = args.init_state,
        K = args.num_steps,
        t = args.time,
        #theta=args.theta,
        backend_id=args.backend_id,
        exec_options = {"noise_model" : None} if args.nonoise else {},
        #api=args.api
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
        
    

