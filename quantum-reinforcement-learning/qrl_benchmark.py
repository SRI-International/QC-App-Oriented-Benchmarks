import os, sys
import time, random
import numpy as np

############### Configure API
# 
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

    import qcb_mpi as mpi
    globals()["mpi"] = mpi
    mpi.init()

    import execute as ex
    globals()["ex"] = ex

    import metrics as metrics
    globals()["metrics"] = metrics

    from qrl_kernel import generate_pqc_circuit, ideal_simulation, kernel_draw
    
    return generate_pqc_circuit, ideal_simulation, kernel_draw


benchmark_name = 'Quantum-Reinforcement-Learning'

np.random.seed(0)

verbose = False
	

############### Result Data Analysis

# Analyze and print measured results
# Expected result is simulated

def analyze_and_print_result (qc, result):
	# obtain counts from the result object
	counts = result.get_counts(qc)
	
	# correct distribution is measuring the key 100% of the time
	correct_dist = ideal_simulation(qc)
	# use our polarization fidelity rescaling
	fidelity = metrics.polarization_fidelity(counts, correct_dist)
		
	return counts, fidelity

def int_to_bitlist(init_string: int, num_qubits: int):
    if init_string >= 2**num_qubits:
        raise ValueError(f"{init_string} cannot be represented in {num_qubits} bits.")
    return [int(b) for b in format(init_string, f'0{num_qubits}b')]

def generate_rotation_params(num_layers, num_qubits, seed=0):
    if seed is not None:
        random.seed(seed)  # Optional for reproducibility

    total_params = 2 * num_layers * num_qubits
    return [random.uniform(0, 2 * np.pi) for _ in range(total_params)]

import argparse
def get_args():
	parser = argparse.ArgumentParser(description="Quantum-Reinforcement-Learning Benchmark")
	parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
	parser.add_argument("--target", "-t", default=None, help="Target Backend", type=str)
	parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
	parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
	parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits", type=int)
	parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
	parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
	parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
	parser.add_argument("--num_layers", "-l", default=2, help="Number of layers", type=int)  
	parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
	parser.add_argument("--init_state", "-state", default=1, help="Initial State to be encoded", type=int)
	parser.add_argument("--n_measurements", "-nmeas", nargs='+', default=[], help="List of measurement operations indices", type=int)
	parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
	parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
	return parser.parse_args()

################ Benchmark Loop

# Execute program with default parameters
def run (min_qubits=3, max_qubits=6, skip_qubits=1, num_shots=100,
		method=1, num_layers = 2, init_state = 1, n_meas = [], backend_id=None, provider_backend=None,
		hub="ibm-q", group="open", project="main", exec_options=None,
		context=None, api=None, get_circuits=False):

	# configure the QED-C Benchmark package for use with the given API
	generate_pqc_circuit, ideal_simulation, kernel_draw = qedc_benchmarks_init(api)
	
	print(f"{benchmark_name} ({method}) Benchmark Program ")

	if method == 1:
		# validate parameters (smallest circuit is 3 qubits)
		max_qubits = max(3, max_qubits)
		min_qubits = min(max(3, min_qubits), max_qubits)
		skip_qubits = max(1, skip_qubits)
		#print(f"min, max qubits = {min_qubits} {max_qubits}")

		# create context identifier
		if context is None: context = f"{benchmark_name} ({method}) Benchmark"
		
		##########

		# Variable to store all created circuits to return and their creation info
		if get_circuits:
			all_qcs = {}

		
		# Initialize metrics module
		metrics.init_metrics()

		# Define custom result handler
		def execution_handler (qc, result, num_qubits, init_state, num_shots):  
		
			# determine fidelity of result set
			num_qubits = int(num_qubits)
			counts, fidelity = analyze_and_print_result(qc, result)
			metrics.store_metric(num_qubits, str(init_state), 'fidelity', fidelity)

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

				# create the circuit for given qubit size and secret string, store time metric
				ts = time.time()

				params = generate_rotation_params(num_layers, num_qubits)
				qc = generate_pqc_circuit(num_qubits, num_layers, init_state_list, params, n_meas)	   
				metrics.store_metric(num_qubits, str(init_state), 'create_time', time.time()-ts)

				# If we only want the circuits:
				if get_circuits:	
					all_qcs[str(num_qubits)] = qc
					# Continue to skip sumbitting the circuit for execution. 
					continue
				
				# submit circuit for execution on target (simulator, cloud simulator, or hardware)
				ex.submit_circuit(qc, num_qubits, str(init_state), shots=num_shots)
				
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
	else:
		print(f"{benchmark_name} ({method}) Benchmark Program not supported yet")

#######################

# if main, execute method
if __name__ == '__main__': 
	args = get_args()
	
	# configure the QED-C Benchmark package for use with the given API
	# (done here so we can set verbose for now)
	generate_pqc_circuit, ideal_simulation, kernel_draw = qedc_benchmarks_init(args.api)
	
	# special argument handling
	ex.verbose = args.verbose
	verbose = args.verbose
	
	if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits
	
	# execute benchmark program
	run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
		skip_qubits=args.skip_qubits,
		num_shots=args.num_shots,
		method=args.method,
		num_layers=args.num_layers,
		init_state = args.init_state,
		n_meas= args.n_measurements,
		backend_id=args.backend_id,
		exec_options = {"noise_model" : None} if args.nonoise else {},
		api=args.api
		)
  