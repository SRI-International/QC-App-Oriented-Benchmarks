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

    from qrl_kernel import generate_pqc_circuits, get_gradient_cirucits, kernel_draw
    
    return generate_pqc_circuits, get_gradient_cirucits, kernel_draw


benchmark_name = 'Quantum-Reinforcement-Learning'

np.random.seed(0)

verbose = False

############### Result Data Analysis

# Analyze and print measured results
# Expected result is simulated

def analyze_and_print_result (qc, result, num_qubits, secret_int, num_shots):
    # size of input is one less than available qubits
	input_size = num_qubits - 1
	
	# obtain counts from the result object
	counts = result.get_counts(qc)
	
	# correct distribution is measuring the key 100% of the time
	correct_dist = 
	# use our polarization fidelity rescaling
	fidelity = metrics.polarization_fidelity(counts, correct_dist)
		
	return counts, fidelity