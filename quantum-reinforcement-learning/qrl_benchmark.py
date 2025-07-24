'''
Quantum Reinforcement Learning Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''
import os, sys
import time, random
import numpy as np

############### Configure API
# 
# Configure the QED-C Benchmark package for use with the given API
def qedc_benchmarks_init(api: str = "qiskit"):
    """
    Initialize the QED-C Benchmark environment for the specified API.
    Sets up sys.path for API-specific and common modules, imports and initializes
    required modules, and returns kernel functions.

    Inputs:
        api (str): Name of the quantum programming API to use (default: "qiskit").
    Outputs:
        generate_pqc_circuit (function): Function to generate parameterized quantum circuits.
        ideal_simulation (function): Function to simulate the ideal output of a circuit.
        kernel_draw (function): Function to draw a sample circuit.
    """
    if api == None: 
        api = "qiskit"  # Default to Qiskit if no API is specified

    # Get the current directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Add the API-specific directory to sys.path
    down_dir = os.path.abspath(os.path.join(current_dir, f"{api}"))
    sys.path = [down_dir] + [p for p in sys.path if p != down_dir]

    # Add the _common directory to sys.path
    up_dir = os.path.abspath(os.path.join(current_dir, ".."))
    common_dir = os.path.abspath(os.path.join(up_dir, "_common"))
    sys.path = [common_dir] + [p for p in sys.path if p != common_dir]
    
    # Add the API directory inside _common to sys.path
    api_dir = os.path.abspath(os.path.join(common_dir, f"{api}"))
    sys.path = [api_dir] + [p for p in sys.path if p != api_dir]

    # Import and initialize the MPI module for parallel execution
    import qcb_mpi as mpi
    globals()["mpi"] = mpi
    mpi.init()

    # Import the execution module and make it globally accessible
    import execute as ex
    globals()["ex"] = ex

    # Import the metrics module and make it globally accessible
    import metrics as metrics
    globals()["metrics"] = metrics

    # Import kernel functions for PQC generation, simulation, and drawing
    from qrl_kernel import generate_pqc_circuit, ideal_simulation, kernel_draw
    
    # Return the kernel functions for use in the benchmark
    return generate_pqc_circuit, ideal_simulation, kernel_draw


benchmark_name = 'Quantum-Reinforcement-Learning'

# Set the random seed for reproducibility
np.random.seed(0)

verbose = False
	

############### Result Data Analysis

# Analyze and print measured results
# Expected result is simulated

def analyze_and_print_result (qc, result):
    """
    Analyze the result of a quantum circuit execution.

    Inputs:
        qc: Quantum circuit object.
        result: Result object from circuit execution.

    Outputs:
        counts: Dictionary of measured counts from the result object.
        fidelity: Polarization fidelity with respect to the ideal simulation.
    """
    # INPUT: qc (quantum circuit), result (execution result)
    # OUTPUT: counts (dict), fidelity (float)
    counts = result.get_counts(qc)
    
    # correct distribution is simulated classically
    correct_dist = ideal_simulation(qc)
    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)
        
    return counts, fidelity

############### Convert integer state to bitlist

# Convert initial state passed as integer to a bitlist of length num_qubits
# Ex Input (init_string = 3, num_qubits = 4) -> Output [0, 0, 1, 1]

def int_to_bitlist(init_string: int, num_qubits: int):
    """
    Convert an integer to a bitlist of length num_qubits.
    Raises ValueError if the integer cannot be represented in the given number of bits.

    Inputs:
        init_string (int): Integer to convert.
        num_qubits (int): Length of the output bitlist.

    Outputs:
        bitlist (list of int): Bitlist representation of the integer.
    """
    # INPUT: init_string (int), num_qubits (int)
    # OUTPUT: bitlist (list of int)
    if init_string >= 2**num_qubits:
        raise ValueError(f"{init_string} cannot be represented in {num_qubits} bits.")
    return [int(b) for b in format(init_string, f'0{num_qubits}b')]

############### Generate list of random parameters

# generate a random list of initial parameters 

def generate_rotation_params(num_layers: int, num_qubits: int, num_op_scaling: int = 0, seed: int = 0):
    """
    Generate a list of random parameters for the PQC.
    Optionally includes scaling parameters and uses a seed for reproducibility.

    Inputs:
        num_layers (int): Number of layers in the PQC.
        num_qubits (int): Number of qubits.
        num_op_scaling (int): Number of scaling parameters to generate (default: 0).
        seed (int): Random seed for reproducibility (default: 0).

    Outputs:
        params (list of float): List of random parameters for the PQC.
    """
    # INPUT: num_layers (int), num_qubits (int), num_op_scaling (int), seed (int)
    # OUTPUT: params (list of float)
    if seed is not None:
        random.seed(seed)  # Optional for reproducibility

    # Main parameters: 2 per layer per qubit, plus one per qubit
    main_param_count = 2 * num_layers * num_qubits + num_qubits
    main_params = [random.uniform(0, 2 * np.pi) for _ in range(main_param_count)]
    # Optional scaling parameters
    scaling_params = [random.uniform(0, 1) for _ in range(num_op_scaling)]

    return main_params + scaling_params

############### Mean Squared Error Loss 

def mse_loss(td_target, old_vals):
    """
    Computes the mean squared error between predictions and targets using lists.

    Args:
        predictions (list of float): Predicted values (Q-values)
        targets (list of float): Target values (TD targets)

    Returns:
        float: Mean squared error
    """
    assert len(old_vals) == len(td_target), "Input lists must be the same length."

    squared_errors = [(p - t) ** 2 for p, t in zip(td_target, old_vals)]
    return sum(squared_errors) / len(squared_errors)

############### Argument parser

import argparse
def get_args():
    """
    Parse command-line arguments for the QRL benchmark.

    Inputs:
        None (reads from sys.argv)

    Outputs:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # INPUT: None (reads from sys.argv)
    # OUTPUT: args (argparse.Namespace)
    parser = argparse.ArgumentParser(description="Quantum-Reinforcement-Learning Benchmark")
    parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    parser.add_argument("--target", "-t", default=None, help="Target Backend", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits", type=int)
    parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)  
    parser.add_argument("--num_layers", "-l", default=2, help="Number of layers", type=int)  
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument("--init_state", "-state", default=1, help="Initial State to be encoded", type=int)
    parser.add_argument("--n_measurements", "-nmeas", default=0, help="Number of measurement operations", type=int)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    return parser.parse_args()

################ QRL Schedules

def schedule (exploration, step):
    """
    Linear schedule for exploration rate in reinforcement learning.

    Inputs:
        exploration (float): Total number of exploration steps.
        step (int): Current step.

    Outputs:
        exploration_probability (float): Exploration probability for the given step.
    """
    # INPUT: exploration (float), step (int)
    # OUTPUT: exploration_probability (float)
    exp_max = 1.0
    exp_min = 0.01

    slope = (exp_min - exp_max) / exploration

    return max(slope * step + exp_max, exp_min)

################ Process the counts to give qvals 

def normalize_counts(counts, num_qubits=None):
    """
    Normalize the counts to get probabilities and convert to bitstrings.
    If num_qubits is not specified, it is inferred from the keys.

    Inputs:
        counts (dict): Dictionary of counts from quantum measurement.
        num_qubits (int, optional): Number of qubits (used for padding bitstrings).

    Outputs:
        probabilities (dict): Dictionary of bitstrings to normalized probabilities.
    """
    # INPUT: counts (dict), num_qubits (int or None)
    # OUTPUT: probabilities (dict)
    normalizer = sum(counts.values())

    try:
        # Try to convert keys to integers to check if they are bitstrings
        dict({str(int(key, 2)): value for key, value in counts.items()})
        if num_qubits is None:
            num_qubits = max(len(key) for key in counts)
        # Pad bitstrings to the correct length
        bitstrings = {key.zfill(num_qubits): value for key, value in counts.items()}
    except ValueError:
        # If keys are not bitstrings, use as is
        bitstrings = counts

    # Normalize to probabilities
    probabilities = dict({key: value / normalizer for key, value in bitstrings.items()})
    assert abs(sum(probabilities.values()) - 1) < 1e-9
    return probabilities

################ Process the counts to give qvals 

def process_result(results, num_actions):
    """
    Process the result object to extract Q-values for each action.

    Inputs:
        results: Result object from circuit execution.
        num_actions (int): Number of actions (qubits).

    Outputs:
        qvals (list of float): List of Q-values, one per action.
    """
    # INPUT: results (result object), num_actions (int)
    # OUTPUT: qvals (list of float)
    counts = results.get_counts()
    counts = normalize_counts(counts, num_actions)
    qvals = [0.0]*num_actions
    
    # For each bit in the bitstring, accumulate the probability if the bit is '1'
    for key, val in counts.items():
        for i, bit in enumerate(key):
            if bit == '1':
                qvals[i] += val
    
    return qvals

################ Process the counts to give qvals 

def calculate_gradients (num_qubits: int, num_layers: int, init_state: list, params: list, percentage_update: float):
    return None

################ Benchmark Loop

# Execute program with default parameters
def run (min_qubits=3, max_qubits=6, skip_qubits=1, max_circuits = 3, num_shots=100,
        method=1, num_layers = 2, init_state = 1, n_measurements = 0, backend_id=None, provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None, api=None, get_circuits=False):
    """
    Main benchmark loop for Quantum Reinforcement Learning.
    Executes the benchmark for a range of qubit sizes and collects metrics.

    Inputs:
        min_qubits (int): Minimum number of qubits.
        max_qubits (int): Maximum number of qubits.
        skip_qubits (int): Step size for qubit range.
        max_circuits (int): Number of circuit repetitions per qubit size.
        num_shots (int): Number of shots per circuit.
        method (int): Algorithm method to use.
        num_layers (int): Number of layers in the PQC.
        init_state (int): Initial state to encode.
        n_measurements (int): Number of measurement operations.
        backend_id (str): Backend identifier.
        provider_backend: Provider backend object.
        hub (str): IBMQ hub.
        group (str): IBMQ group.
        project (str): IBMQ project.
        exec_options (dict): Execution options.
        context (str): Context identifier.
        api (str): Quantum programming API.
        get_circuits (bool): If True, only generate and return circuits.

    Outputs:
        If get_circuits is True:
            all_qcs (dict): Dictionary of generated circuits.
            metrics.circuit_metrics (dict): Circuit metrics.
        Otherwise:
            None (results are handled via metrics and plots).
    """
    # INPUT: see docstring above
    # OUTPUT: see docstring above
    # configure the QED-C Benchmark package for use with the given API
    generate_pqc_circuit, ideal_simulation, kernel_draw = qedc_benchmarks_init(api)
    
    print(f"{benchmark_name} ({method}) Benchmark Program")

    if method == 1:
        # validate parameters (smallest circuit is 3 qubits)
        max_qubits = max(3, max_qubits)
        min_qubits = min(max(3, min_qubits), max_qubits)
        skip_qubits = max(1, skip_qubits)

        # create context identifier
        if context is None: 
            context = f"{benchmark_name} ({method}) Benchmark"
        
        ##########

        # Variable to store all created circuits to return and their creation info
        if get_circuits:
            all_qcs = {}

        # Initialize metrics module
        metrics.init_metrics()

        # Define custom result handler for circuit execution
        def execution_handler (qc, result, num_qubits, idx, num_shots):  
            # INPUT: qc (quantum circuit), result (execution result), num_qubits (int), idx (int), num_shots (int)
            # OUTPUT: None (stores metrics)
            # determine fidelity of result set
            num_qubits = int(num_qubits)
            counts, fidelity = analyze_and_print_result(qc, result)
            metrics.store_metric(num_qubits, idx, 'fidelity', fidelity)

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
            if get_circuits:
                print(f"************\nCreating circuit with num_qubits = {num_qubits}")
                all_qcs[str(num_qubits)] = {}
            else:
                print(f"************\nExecuting circuit with num_qubits = {num_qubits}")
                # Initialize dictionary to store circuits for this qubit group. 
                
                init_state_list = int_to_bitlist(init_state, num_qubits)

                for idx in range(max_circuits):
                    params = generate_rotation_params(num_layers, num_qubits, seed = idx)
                    # create the circuit for given qubit size and secret string, store time metric
                    mpi.barrier()
                    ts = time.time()
                    qc = generate_pqc_circuit(num_qubits, num_layers, init_state_list, params, n_measurements)       
                    metrics.store_metric(num_qubits, idx, 'create_time', time.time()-ts)

                    # If we only want the circuits:
                    if get_circuits:    
                        all_qcs[str(num_qubits)][idx] = qc
                        # Continue to skip submitting the circuit for execution. 
                        continue
                    
                    # submit circuit for execution on target (simulator, cloud simulator, or hardware)
                    ex.submit_circuit(qc, num_qubits, idx, shots=num_shots)
                
            # Wait for some active circuits to complete; report metrics when groups complete
            ex.throttle_execution(metrics.finalize_group)
        
        # Early return if we just want the circuits
        if get_circuits:
            print(f"************\nReturning circuits and circuit information")
            return all_qcs, metrics.circuit_metrics

        # Wait for all active circuits to complete; report metrics when groups complete
        ex.finalize_execution(metrics.finalize_group)
        
        ##########
        
        # draw a sample circuit
        kernel_draw()

        # Plot metrics for all circuit sizes
        metrics.plot_metrics(f"Benchmark Results - {benchmark_name} ({method}) - Qiskit")
    
    elif method == 2:
        # Method 2: Quantum Reinforcement Learning with environment and replay buffer
        try:
            from _common.env_utils import Environment, ReplayBuffer
            from _common.optimizers import Adam
        except Exception as e:
            print(f"{benchmark_name} ({method}) Benchmark cannot run due to \t {e!r}.")

        # Define execution handler for QRL
        def execution_handler(qc, result, num_qubits, s_int, num_shots):
            # INPUT: qc (quantum circuit), result (execution result), num_qubits (int), s_int (int), num_shots (int)
            # OUTPUT: None (stores result globally)
            # Stores the results to the global saved_result variable
            global saved_result
            saved_result = result
        
        
        ex.init_execution(execution_handler)
        ex.max_jobs_active = 1

        result_array = []
        learning_start = 5
        target_update = 5
        lr = 0.0001
        batch_size = 5
        gamma = 0.95

        # Initialize environment and replay buffer
        e = Environment()
        e.make_env()
        rb = ReplayBuffer(batch_size)
        
        # Calculate parameters for quantum circuits
        num_qubits = int(np.sqrt(e.get_observation_size()))
        num_actions = e.get_num_of_actions()
        total_steps = 100 # Keep the defaults and expose this to the 
        exploration_fraction = 0.35 # Expose run method

        # init params
        params = generate_rotation_params(num_layers, num_qubits, num_actions)
        target_network_params = params
        opt = Adam(params, lr = lr)

        obs = e.reset()

        for step in range(total_steps):
            #
            eps = schedule(exploration_fraction * total_steps, step)

            if random.random() < eps:
                # Exploration: sample a random action
                action = e.sample()
            else:
                # Exploitation: use quantum circuit to select action
                print(step)
                init_state_list = int_to_bitlist(obs, num_qubits)
                qc = generate_pqc_circuit(num_qubits, num_layers, init_state_list, params, num_actions)
                uid = "qrl_" + str(obs) + "_" + str(step)
                ex.submit_circuit(qc, num_qubits, uid, shots = num_shots)
                ex.finalize_execution(None, report_end=False)
                global saved_result
                result_array.append(saved_result)
                qvals = process_result(result_array[-1], num_actions)
                action = qvals.index(max(qvals))
                print(step, obs, action, qvals)

            # Take the action in the environment
            next_obs, reward, term, trunc, info = e.step(action)

            rb.add_buffer_item(obs, next_obs, action, reward, term)
            obs = next_obs

            if term:
                obs = e.reset()

            if step > learning_start:
                if step % target_update == 0:
                    batch = rb.sample_batch_from_buffer(batch_size)
                    #ex.max_jobs_active = batch_size
                    qc_arr = []
                    td_res_arr = []
                    td_vals = []
                    td_target = []
                    old_vals = []

                    for state, done, reward in zip(batch[rb.next_obs_idx], batch[rb.dones_idx], batch[rb.rewards_idx]):
                        uid = "qrl_target_params_batch_" + str(state) + "_" + str(step)
                        init_state_list = int_to_bitlist(state, num_qubits)
                        qc_arr.append(generate_pqc_circuit(num_qubits, num_layers, init_state_list, target_network_params, num_actions))
                        ex.submit_circuit(qc_arr[-1], num_qubits, uid, shots = num_shots)
                        ex.finalize_execution(None, report_end=False)
                        #global saved_result
                        td_res_arr.append(saved_result)
                        td_vals.append(max(process_result(td_res_arr[-1], num_actions)))
                        td_target.append(reward + gamma * td_vals[-1] * (1 - done))
                    
                    for state, action in zip(batch[rb.obs_idx], batch[rb.actions_idx]):
                        uid = "qrl_old_params_batch_" + str(state) + "_" + str(step)
                        init_state_list = int_to_bitlist(state, num_qubits)
                        qc_arr.append(generate_pqc_circuit(num_qubits, num_layers, init_state_list, params, num_actions))
                        ex.submit_circuit(qc_arr[-1], num_qubits, uid, shots = num_shots)
                        ex.finalize_execution(None, report_end=False)
                        #global saved_result
                        old_vals.append(process_result(saved_result, num_actions)[action])

                    loss = mse_loss(td_target, old_vals)
                    print(step, loss)
                        
    else:
        print(f"{benchmark_name} ({method}) Benchmark Program not supported yet")

#######################

# if main, execute method
if __name__ == '__main__': 
    # Parse command-line arguments
    args = get_args()
    
    # configure the QED-C Benchmark package for use with the given API
    # (done here so we can set verbose for now)
    generate_pqc_circuit, ideal_simulation, kernel_draw = qedc_benchmarks_init(args.api)
    
    # special argument handling
    ex.verbose = args.verbose
    verbose = args.verbose
    
    # If num_qubits is specified, override min and max qubits
    if args.num_qubits > 0: 
        args.min_qubits = args.max_qubits = args.num_qubits
    
    # execute benchmark program
    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits,
        max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        method=args.method,
        num_layers=args.num_layers,
        init_state=args.init_state,
        n_measurements= args.n_measurements,
        backend_id=args.backend_id,
        exec_options = {"noise_model" : None} if args.nonoise else {},
        api=args.api
        )