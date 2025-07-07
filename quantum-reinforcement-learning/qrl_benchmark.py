import os, sys
import time
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

    from qrl_kernel import generate_pqc_circuits, ideal_simulation, kernel_draw
    
    return generate_pqc_circuits,ideal_simulation, kernel_draw


benchmark_name = 'Quantum-Reinforcement-Learning'

np.random.seed(0)

verbose = False
	

############### Result Data Analysis

# Analyze and print measured results
# Expected result is simulated

def analyze_and_print_result (qc, result, num_qubits, num_shots):
    # size of input is one less than available qubits
	
	# obtain counts from the result object
	counts = result.get_counts(qc)
	
	# correct distribution is measuring the key 100% of the time
	correct_dist = ideal_simulation(qc)
	# use our polarization fidelity rescaling
	fidelity = metrics.polarization_fidelity(counts, correct_dist)
		
	return counts, fidelity


      

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
	parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)  
	parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
	parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
	parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
	return parser.parse_args()
	
# if main, execute method
if __name__ == '__main__': 
	args = get_args()
	
	# configure the QED-C Benchmark package for use with the given API
	# (done here so we can set verbose for now)
	generate_pqc_circuits, kernel_draw = qedc_benchmarks_init(args.api)
	
	# special argument handling
	ex.verbose = args.verbose
	verbose = args.verbose
	
	if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits
	
	# execute benchmark program
	run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
		skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
		num_shots=args.num_shots,
		method=args.method,
		backend_id=args.backend_id,
		exec_options = {"noise_model" : None} if args.nonoise else {},
		api=args.api
		)
   

      