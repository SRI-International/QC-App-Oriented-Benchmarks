'''
Quantum Reinforcement Learning Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''
import os, sys, copy
import time, random
import numpy as np

############### Configure API

def qedc_benchmarks_init(api: str = "qiskit"):
    """
    Initialize the QED-C Benchmark environment for the specified API.
    Sets up sys.path for API-specific and common modules, imports and initializes
    required modules, and returns kernel functions.

    Args:
        api (str): Name of the quantum programming API to use (default: "qiskit").

    Returns:
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

def analyze_and_print_result(qc, result):
    """
    Analyze the result of a quantum circuit execution by comparing measured results
    to the ideal simulation and computing the polarization fidelity.

    Args:
        qc: Quantum circuit object.
        result: Result object from circuit execution.

    Returns:
        counts (dict): Dictionary of measured counts from the result object.
        fidelity (float): Polarization fidelity with respect to the ideal simulation.
    """
    counts = result.get_counts(qc)
    # Simulate the correct distribution classically
    correct_dist = ideal_simulation(qc)
    # Use polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)
    return counts, fidelity

############### Convert integer state to bitlist

def int_to_bitlist(init_string: int, num_qubits: int):
    """
    Convert an integer to a bitlist of length num_qubits.
    Raises ValueError if the integer cannot be represented in the given number of bits.

    Args:
        init_string (int): Integer to convert.
        num_qubits (int): Length of the output bitlist.

    Returns:
        bitlist (list of int): Bitlist representation of the integer.
    Example:
        int_to_bitlist(3, 4) -> [0, 0, 1, 1]
    """
    if init_string >= 2**num_qubits:
        raise ValueError(f"{init_string} cannot be represented in {num_qubits} bits.")
    return [int(b) for b in format(init_string, f'0{num_qubits}b')]

############### Generate list of random parameters

def generate_rotation_params(num_layers: int, num_qubits: int, num_op_scaling: int = 0, seed: int = 0, data_reupload: bool = False):
    """
    Generate a list of random parameters for the PQC.
    Optionally includes scaling parameters and uses a seed for reproducibility.

    Args:
        num_layers (int): Number of layers in the PQC.
        num_qubits (int): Number of qubits.
        num_op_scaling (int): Number of scaling parameters to generate (default: 0).
        seed (int): Random seed for reproducibility (default: 0).

    Returns:
        params (list of float): List of random parameters for the PQC.
    """
    if seed is not None:
        random.seed(seed)  # Optional for reproducibility

    params = []

    for layer in range(num_layers):
        # --- RX block (only if layer == 0 or data_reupload is True)
        if layer == 0 or data_reupload:
            rx_params = [np.pi for _ in range(num_qubits)]
            params.extend(rx_params)
        else:
            # Skip over RX block
            pass

        # --- RY block
        ry_params = [random.uniform(0, 2 * np.pi) for _ in range(num_qubits)]
        params.extend(ry_params)

        # --- RZ block
        rz_params = [random.uniform(0, 2 * np.pi) for _ in range(num_qubits)]
        params.extend(rz_params)

    # --- Optional scaling params at the end
    scaling_params = [random.uniform(0, 1) for _ in range(num_op_scaling)]
    params.extend(scaling_params)

    return params

############### Mean Squared Error Loss 

def mse_loss(td_target, old_vals):
    """
    Computes the mean squared error between predictions and targets using lists.

    Args:
        td_target (list of float): Target values (TD targets)
        old_vals (list of float): Predicted values (Q-values)

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

    Returns:
        args (argparse.Namespace): Parsed command-line arguments.
    """
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
    parser.add_argument("--data_reupload", "-r", action="store_true", help="Enable data reupload")
    return parser.parse_args()

################ QRL Schedules

def schedule(exploration, step):
    """
    Linear schedule for exploration rate in reinforcement learning.

    Args:
        exploration (float): Total number of exploration steps.
        step (int): Current step.

    Returns:
        exploration_probability (float): Exploration probability for the given step.
    """
    exp_max = 1.0
    exp_min = 0.05
    slope = (exp_min - exp_max) / exploration
    return max(slope * step + exp_max, exp_min)

################ Process the counts to give qvals 

def normalize_counts(counts, num_qubits=None):
    """
    Normalize the counts to get probabilities and convert to bitstrings.
    If num_qubits is not specified, it is inferred from the keys.

    Args:
        counts (dict): Dictionary of counts from quantum measurement.
        num_qubits (int, optional): Number of qubits (used for padding bitstrings).

    Returns:
        probabilities (dict): Dictionary of bitstrings to normalized probabilities.
    """
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

def process_result(results, num_actions, op_scaling = []):
    """
    Process the result object to extract Q-values for each action.

    Args:
        results: Result object from circuit execution.
        num_actions (int): Number of actions (qubits).

    Returns:
        qvals (list of float): List of Q-values, one per action.
    """
    counts = results.get_counts()
    counts = normalize_counts(counts, num_actions)
    qvals = [0.0]*num_actions
    
    # For each bit in the bitstring, accumulate the probability if the bit is '1'
    for key, val in counts.items():
        for i, bit in enumerate(key):
            if bit == '1':
                qvals[i] += val

    if len(op_scaling) == 0:
        op_scaling = [1.0] * num_actions
    
    for i in range(len(qvals)):
        qvals[i] = 1 - 2*qvals[i]
        qvals[i] *= op_scaling[i] 

    return qvals

################ Calculate parameter gradients using parameter-shift rule

def calculate_gradients(num_qubits: int, num_layers: int, batch_obs: list, params: list, n_measurements: int, num_shots: int, td_targets: list, actions: list, data_reupload: bool, ex, qrl_metrics):
    """
    Calculate gradients of the loss function with respect to each parameter using the parameter-shift rule.

    Args:
        num_qubits (int): Number of qubits.
        num_layers (int): Number of layers in the PQC.
        batch_obs (list): Batch of observations (states).
        params (list): Current parameter values.
        n_measurements (int): Number of measurement operations.
        num_shots (int): Number of shots per circuit.
        td_targets (list): Target values for each sample in the batch.
        actions (list): Actions taken for each sample in the batch.
        ex: Execution module.

    Returns:
        parameter_shift (function): Function that computes the gradient for a given parameter index.
    """
    def parameter_shift(idx: int):
        grad = 0.0
        for init_state, td_target, action in zip(batch_obs, td_targets, actions):
            init_state_list = int_to_bitlist(init_state, num_qubits)
            ex.max_jobs_active = 1
            params_p = copy.deepcopy(params)
            params_p[idx] += np.pi / 2
            qc = generate_pqc_circuit(num_qubits, num_layers, init_state_list, params_p, n_measurements, data_reupload = data_reupload)
            uid = "qrl_grad_calc_" + str(init_state) + "_plus" 
            c_time = time.time()
            ex.submit_circuit(qc, num_qubits, uid, shots = num_shots)
            qrl_metrics.call_history.append("quantum")
            ex.finalize_execution(None, report_end = False)
            qrl_metrics.quantum_time += (time.time() - c_time)
            res_p = process_result(saved_result, n_measurements)
            qrl_metrics.circuit_evaluations += 1

            params_p[idx] -= np.pi 
            qc = generate_pqc_circuit(num_qubits, num_layers, init_state_list, params_p, n_measurements, data_reupload = data_reupload)
            uid = "qrl_grad_calc_" + str(init_state) + "_minus"
            c_time = time.time()
            ex.submit_circuit(qc, num_qubits, uid, shots = num_shots)
            qrl_metrics.call_history.append("quantum")
            ex.finalize_execution(None, report_end = False)
            qrl_metrics.quantum_time += (time.time() - c_time)
            res_n = process_result(saved_result, n_measurements)
            qrl_metrics.circuit_evaluations += 1

            p_loss = mse_loss([td_target], [res_p[action]])
            n_loss = mse_loss([td_target], [res_n[action]])

            grad += 0.5 * (p_loss - n_loss)
            qrl_metrics.call_history.append("Classical")
            qrl_metrics.gradient_evaluations += 1

        return grad/len(batch_obs)           
    return parameter_shift

################ Calculate loss func

def calculate_loss(num_qubits: int, num_layers: int, batch_obs: list, n_measurements: int, num_shots: int, td_targets: list, actions: list, data_reupload: bool, ex, qrl_metrics):
    """
    Returns a loss function for the QRL benchmark, which computes the mean squared error
    between target values and PQC outputs for a batch of observations/actions.

    Args:
        num_qubits (int): Number of qubits in the PQC.
        num_layers (int): Number of layers in the PQC.
        batch_obs (list): Batch of initial states (as integers).
        n_measurements (int): Number of measurement operations.
        num_shots (int): Number of shots per circuit execution.
        td_targets (list): Target values for each sample in the batch.
        actions (list): Actions taken for each sample in the batch.
        ex: Execution module for submitting and running circuits.
        qrl_metrics: Metrics object for tracking circuit and gradient evaluations.

    Returns:
        loss_fn (function): A function that takes PQC parameters and returns the batch MSE loss.
    """
    def loss_fn(x0):
        """
        Computes the mean squared error loss for a batch of observations/actions
        given a set of PQC parameters.

        Args:
            x0 (list): Current PQC parameter values.

        Returns:
            float: Mean squared error loss for the batch.
        """
        qvals = []  # List to store Q-values (PQC outputs) for each sample in the batch
        for init_state, action in zip(batch_obs, actions):
            # Convert integer state to bitlist for circuit initialization
            init_state_list = int_to_bitlist(init_state, num_qubits)
            # Limit to one active job at a time (serial execution)
            ex.max_jobs_active = 1
            # Generate the parameterized quantum circuit for this sample
            qc = generate_pqc_circuit(num_qubits, num_layers, init_state_list, x0, n_measurements, data_reupload = data_reupload)
            # Unique identifier for this circuit execution
            uid = "qrl_grad_calc_" + str(init_state)
            # Submit the circuit for execution
            c_time = time.time()
            ex.submit_circuit(qc, num_qubits, uid, shots=num_shots)
            # Finalize execution (wait for results)
            ex.finalize_execution(None, report_end=False)
            qrl_metrics.quantum_time += (time.time() - c_time)
            # Process the result to extract Q-values
            res_p = process_result(saved_result, n_measurements)
            # Update metrics for circuit and gradient evaluations
            qrl_metrics.circuit_evaluations += 1
            qrl_metrics.gradient_evaluations += 1
            # Append the Q-value for the taken action
            qvals.append(res_p[action])
        # Compute and return the mean squared error loss for the batch
        return mse_loss(td_targets, qvals)
    return loss_fn
    
################ Benchmark Loop

def run(min_qubits=3, max_qubits=6, skip_qubits=1, max_circuits=3, num_shots=100,
        method=1, num_layers=2, init_state=1, n_measurements=0, num_qubits=3, backend_id=None, provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None, api=None, get_circuits=False, data_reupload = False):
    """
    Main benchmark loop for Quantum Reinforcement Learning.
    Executes the benchmark for a range of qubit sizes and collects metrics.

    Args:
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

    Returns:
        If get_circuits is True:
            all_qcs (dict): Dictionary of generated circuits.
            metrics.circuit_metrics (dict): Circuit metrics.
        Otherwise:
            None (results are handled via metrics and plots).
    """
    # configure the QED-C Benchmark package for use with the given API
    generate_pqc_circuit, ideal_simulation, kernel_draw = qedc_benchmarks_init(api)
    globals()['ideal_simulation'] = ideal_simulation
    
    print(f"{benchmark_name} ({method}) Benchmark Program")

    if method == 1:
        # Method 1: Standard PQC benchmarking over a range of qubit sizes
        max_qubits = max(3, max_qubits)
        min_qubits = min(max(3, min_qubits), max_qubits)
        skip_qubits = max(1, skip_qubits)

        # create context identifier
        if context is None: 
            context = f"{benchmark_name} ({method}) Benchmark"
        
        # Variable to store all created circuits to return and their creation info
        if get_circuits:
            all_qcs = {}

        # Initialize metrics module
        metrics.init_metrics()

        # Define custom result handler for circuit execution
        def execution_handler(qc, result, num_qubits, idx, num_shots):  
            """
            Handle the execution result for a single circuit, storing fidelity metric.

            Args:
                qc: Quantum circuit object.
                result: Execution result object.
                num_qubits (int): Number of qubits.
                idx (int): Circuit index.
                num_shots (int): Number of shots.
            """
            num_qubits = int(num_qubits)
            counts, fidelity = analyze_and_print_result(qc, result)
            metrics.store_metric(num_qubits, idx, 'fidelity', fidelity)

        # Initialize execution module using the execution result handler above and specified backend_id
        ex.init_execution(execution_handler)
        ex.set_execution_target(backend_id, provider_backend=provider_backend,
                hub=hub, group=group, project=project, exec_options=exec_options,
                context=context)
        
        ex.max_active_jobs = int((max_qubits + 1 - min_qubits) * (max_circuits / skip_qubits))  

        # for noiseless simulation, set noise model to be None
        # ex.set_noise_model(None)

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
                    params = generate_rotation_params(num_layers, num_qubits, seed=idx, data_reupload=data_reupload)
                    # create the circuit for given qubit size and secret string, store time metric
                    mpi.barrier()
                    ts = time.time()
                    qc = generate_pqc_circuit(num_qubits, num_layers, init_state_list, params, n_measurements, data_reupload=data_reupload)       
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
        
        # draw a sample circuit
        kernel_draw()

        # Plot metrics for all circuit sizes
        metrics.plot_metrics(f"Benchmark Results - {benchmark_name} ({method}) - Qiskit")
    
    elif method == 2:
        # Method 2: Quantum Reinforcement Learning with environment and replay buffer
        try:
            from _common.env_utils import Environment, ReplayBuffer
            from _common.optimizers import Adam, SPSA
            from _common.qrl_metrics import qrl_metrics
        except Exception as e:
            print(f"{benchmark_name} ({method}) Benchmark cannot run due to \t {e!r}.")

        # Define execution handler for QRL
        def execution_handler(qc, result, num_qubits, s_int, num_shots):
            """
            Handle the execution result for a QRL circuit, storing result globally.

            Args:
                qc: Quantum circuit object.
                result: Execution result object.
                num_qubits (int): Number of qubits.
                s_int (int): State integer.
                num_shots (int): Number of shots.
            """
            global saved_result
            saved_result = result
        
        ex.init_execution(execution_handler)
        ex.max_jobs_active = 1

        result_array = []
        learning_start = 500
        target_update = 10
        params_update = 10
        lr = 0.01
        batch_size = 32
        gamma = 0.95
        total_steps = 100 # Keep the defaults and expose this to the 
        exploration_fraction = 0.5 # Expose run method
        tau = 0.9
        buffer_size = 2000
        qrl_metrics = qrl_metrics()
        metric_print_interval = 50

        total_time = time.time()
        # Initialize environment and replay buffer
        e = Environment()
        e.make_env()
        rb = ReplayBuffer(buffer_size)
        
        # Calculate parameters for quantum circuits
        num_qubits = int(np.sqrt(e.get_observation_size()))
        num_actions = e.get_num_of_actions()

        # init params
        params = generate_rotation_params(num_layers, num_qubits, 0, data_reupload=data_reupload)
        target_network_params = copy.deepcopy(params)

        opt = SPSA(params)
        #opt = Adam(params, lr) # Uncomment to run

        obs = e.reset()
        qrl_metrics.env_evals += 1  
        loss = 0.0
        for step in range(total_steps):
            # Compute exploration probability for this step
            step_time = time.time()
            qrl_metrics.steps += 1
            eps = schedule(exploration_fraction * total_steps, step)

            if random.random() < eps:
                # Exploration: sample a random action
                env_start = time.time()
                action = e.sample()
                qrl_metrics.environment_time += (time.time() - env_start)
                qrl_metrics.env_evals += 1
                qrl_metrics.explore_steps += 1
            else:
                # Exploitation: use quantum circuit to select action
                init_state_list = int_to_bitlist(obs, num_qubits)
                qc = generate_pqc_circuit(num_qubits, num_layers, init_state_list, params, num_actions, data_reupload=data_reupload)
                uid = "qrl_" + str(obs) + "_" + str(step)
                circ_start = time.time()
                ex.submit_circuit(qc, num_qubits, uid, shots=num_shots)
                ex.finalize_execution(None, report_end=False)
                qrl_metrics.quantum_time += (time.time() - circ_start)
                qrl_metrics.circuit_evaluations += 1
                global saved_result
                result_array.append(saved_result)
                qvals = process_result(result_array[-1], num_actions)
                action = qvals.index(max(qvals))
                qrl_metrics.exploit_steps += 1
            # Take the action in the environment
            env_start = time.time()
            next_obs, reward, term, trunc, info = e.step(action)
            qrl_metrics.environment_time += (time.time() - env_start)
            qrl_metrics.env_evals += 1  

            rb.add_buffer_item(obs, next_obs, action, reward, term)
            obs = next_obs

            if term:
                obs = e.reset()
                qrl_metrics.env_evals += 1
                qrl_metrics.num_episodes += 1
            
            if reward == 1.0:
                qrl_metrics.num_success += 1

            if (step + 1) > learning_start:
                if (step + 1) % params_update == 0:
                    batch = rb.sample_batch_from_buffer(batch_size)
                    #ex.max_jobs_active = batch_size
                    qc_arr = []
                    td_res_arr = []
                    td_vals = []
                    td_targets = []
                    old_vals = []

                    for state, done, reward in zip(batch[rb.next_obs_idx], batch[rb.dones_idx], batch[rb.rewards_idx]):
                        uid = "qrl_target_params_batch_" + str(state) + "_" + str(step)
                        init_state_list = int_to_bitlist(state, num_qubits)
                        qc_arr.append(generate_pqc_circuit(num_qubits, num_layers, init_state_list, target_network_params, num_actions, data_reupload=data_reupload))
                        circ_start = time.time()
                        ex.submit_circuit(qc_arr[-1], num_qubits, uid, shots=num_shots)
                        ex.finalize_execution(None, report_end=False)
                        qrl_metrics.quantum_time += (time.time() - circ_start)
                        qrl_metrics.circuit_evaluations += 1
                        td_res_arr.append(saved_result)
                        td_vals.append(max(process_result(td_res_arr[-1], num_actions)))
                        td_targets.append(reward + gamma * td_vals[-1] * (1 - done))
                    
                    
                    for state, action in zip(batch[rb.obs_idx], batch[rb.actions_idx]):
                        uid = "qrl_old_params_batch_" + str(state) + "_" + str(step)
                        init_state_list = int_to_bitlist(state, num_qubits)
                        qc_arr.append(generate_pqc_circuit(num_qubits, num_layers, init_state_list, params, num_actions, data_reupload=data_reupload))
                        circ_start = time.time()
                        ex.submit_circuit(qc_arr[-1], num_qubits, uid, shots=num_shots)
                        ex.finalize_execution(None, report_end=False)
                        qrl_metrics.quantum_time += (time.time() - circ_start)
                        qrl_metrics.circuit_evaluations += 1
                        old_vals.append(process_result(saved_result, num_actions)[action])
                    
                    loss = mse_loss(td_targets, old_vals)
                    qrl_metrics.loss_history.append(loss)
                    
                    # Compute gradients and update parameters
                    grad_time = time.time()
                    #grad_fn = calculate_gradients(num_qubits, num_layers, batch[rb.obs_idx], params, num_actions, num_shots, td_targets, batch[rb.actions_idx], data_reupload, ex, qrl_metrics)
                    grad_fn = calculate_loss(num_qubits, num_layers, batch[rb.obs_idx], num_actions, num_shots, td_targets, batch[rb.actions_idx], data_reupload, ex, qrl_metrics)
                    params = opt.step(grad_fn)
                    qrl_metrics.gradient_time += (time.time() - grad_time)
                
                if (step + 1) % target_update == 0:
                    for i, (t_param, param) in enumerate(zip(target_network_params, params)):
                        target_network_params[i] = tau * param + (1 - tau) * t_param
                
            if (step + 1) % metric_print_interval == 0:
                qrl_metrics.total_time = time.time() - total_time
                qrl_metrics.print_metrics()

            qrl_metrics.step_time += (time.time() - step_time)
            qrl_metrics.update_history()
        qrl_metrics.plot_metrics()

    # This is a measurement sweep over the number of measurement. Currently WIP. Do not use.
    elif method == 3:
        # Method 3: Measuements pass
        max_measurements = n_measurements
        # create context identifier
        if context is None: 
            context = f"{benchmark_name} ({method}) Benchmark"
        
        # Variable to store all created circuits to return and their creation info
        if get_circuits:
            all_qcs = {}

        # Initialize metrics module
        metrics.init_metrics()

        # Define custom result handler for circuit execution
        def execution_handler(qc, result, num_qubits, idx, num_shots):  
            """
            Handle the execution result for a single circuit, storing fidelity metric.

            Args:
                qc: Quantum circuit object.
                result: Execution result object.
                num_qubits (int): Number of qubits.
                idx (int): Circuit index.
                num_shots (int): Number of shots.
            """
            num_qubits = int(num_qubits)
            counts, fidelity = analyze_and_print_result(qc, result)
            metrics.store_metric(num_qubits, idx, 'fidelity', fidelity)

        # Initialize execution module using the execution result handler above and specified backend_id
        ex.init_execution(execution_handler)
        ex.set_execution_target(backend_id, provider_backend=provider_backend,
                hub=hub, group=group, project=project, exec_options=exec_options,
                context=context)

        # Execute Benchmark Program N times for multiple circuit sizes
        # Accumulate metrics asynchronously as circuits complete
        for num_qubits_meas in range(1, max_measurements + 1, 1):
            if get_circuits:
                print(f"************\nCreating circuit with num_qubits = {num_qubits_meas}")
                all_qcs[str(num_qubits_meas)] = {}
            else:
                print(f"************\nExecuting circuit with num_qubits = {num_qubits_meas}")
                # Initialize dictionary to store circuits for this qubit group. 
                init_state_list = int_to_bitlist(init_state, num_qubits)

                for idx in range(max_circuits):
                    params = generate_rotation_params(num_layers, num_qubits, seed=idx, data_reupload=data_reupload)
                    # create the circuit for given qubit size and secret string, store time metric
                    mpi.barrier()
                    ts = time.time()
                    qc = generate_pqc_circuit(num_qubits, num_layers, init_state_list, params, num_qubits_meas, data_reupload=data_reupload)       
                    metrics.store_metric(num_qubits_meas, idx, 'create_time', time.time()-ts)

                    # If we only want the circuits:
                    if get_circuits:    
                        all_qcs[str(num_qubits)][idx] = qc
                        # Continue to skip submitting the circuit for execution. 
                        continue
                    
                    # submit circuit for execution on target (simulator, cloud simulator, or hardware)
                    ex.submit_circuit(qc, num_qubits_meas, idx, shots=num_shots)
                
            # Wait for some active circuits to complete; report metrics when groups complete
            ex.throttle_execution(metrics.finalize_group)
        
        # Early return if we just want the circuits
        if get_circuits:
            print(f"************\nReturning circuits and circuit information")
            return all_qcs, metrics.circuit_metrics

        # Wait for all active circuits to complete; report metrics when groups complete
        ex.finalize_execution(metrics.finalize_group)
        
        # draw a sample circuit
        kernel_draw()

        # Plot metrics for all circuit sizes
        metrics.plot_metrics(f"Benchmark Results - {benchmark_name} ({method}) - {num_qubits} Qubits")
                        
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
        n_measurements=args.n_measurements,
        backend_id=args.backend_id,
        exec_options={"noise_model": None} if args.nonoise else {},
        api=args.api,
        data_reupload=args.data_reupload,
        num_qubits=args.num_qubits
        )