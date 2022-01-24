"""
Amplitude Estimation Benchmark Program via Phase Estimation - Braket
"""
import time
import sys
import numpy as np
from ae_utils import adjoint
from braket.circuits import Circuit

sys.path[1:1] = ["_common", "_common/braket", "quantum-fourier-transform/braket"]
sys.path[1:1] = ["../../_common", "../../_common/braket", "../../quantum-fourier-transform/braket"]
import execute as ex
import metrics as metrics

# saved subcircuits circuits for printing
A_ = None
Q_ = None
cQ_ = None
QC_ = None
QFTI_ = None

############### Circuit Definition

def AmplitudeEstimation(num_state_qubits, num_counting_qubits, a, psi_zero=None, psi_one=None):
    qc = Circuit()

    # create the Amplitude Generator circuit
    A = A_gen(num_state_qubits, a, psi_zero, psi_one)

    # create the Quantum Operator circuit and a controlled version of it
    cQ, Q = Ctrl_Q(num_state_qubits, A)

    # TODO ...

# Construct A operator that takes |0>_{n+1} to sqrt(1-a) |psi_0>|0> + sqrt(a) |psi_1>|1>
def A_gen(num_state_qubits, a, psi_zero=None, psi_one=None):
    
    if psi_zero == None:
        psi_zero = '0'*num_state_qubits
    if psi_one == None:
        psi_one = '1'*num_state_qubits

    theta = 2 * np.arcsin(np.sqrt(a))

    # Let the objective be qubit index n; state is on qubits 0 through n-1
    qc_A = Circuit()

    # takes state to |0>_{n} (sqrt(1-a) |0> + sqrt(a) |1>)
    qc_A.ry(num_state_qubits, theta)

    # takes state to sqrt(1-a) |psi_0>|0> + sqrt(a) |0>_{n}|1>
    qc_A.x(num_state_qubits)
    for i in range(num_state_qubits):
        if psi_zero[i] == '1':
            qc_A.cnot(num_state_qubits, i)
    qc_A.x(num_state_qubits)

    # takes state to sqrt(1-a) |psi_0>|0> + sqrt(a) |psi_1>|1>
    for i in range(num_state_qubits):
        if psi_one[i] == '1':
            qc_A.cnot(num_state_qubits, i)
    
    return qc_A

# Construct the gover-like operator and a controlled version of it
def Ctrl_Q(num_state_qubits, A_circ):
    
    # index n is the objective qubit, and indexes 0 through n-1 are state qubits
    qc = Circuit()
    
    temp_A = A_circ.copy()
    A_inverse = adjoint(temp_A)

    ### Each cycle in Q applies in order: -S_chi, A_circ_inverse, S_0, A_circ
    # -S_chi
    qc.x(num_state_qubits)
    qc.z(num_state_qubits)
    qc.x(num_state_qubits)

    # A_circ_inverse
    qc.add_circuit(A_inverse)

    # S_0
    for i in range(num_state_qubits + 1):
        qc.x(i)
    qc.h(num_state_qubits)

    # TODO ... Work on MCX gate implementation


    


# Analyze and print measured results
# Expected result is always the secret_int, so fidelity calc is simple
def analyze_and_print_result(qc, result, num_counting_qubits, s_int, num_shots):
    print("Results: ", results)

def bitstring_to_a(counts, num_counting_qubits):
    est_counts = {}
    m = num_counting_qubits
    precision = int(num_counting_qubits / (np.log2(10))) + 2
    for key in counts.keys():
        r = counts[key]
        num = int(key,2) / (2**m)
        a_est = round((np.sin(np.pi * num) )** 2, precision)
        if a_est not in est_counts.keys():
            est_counts[a_est] = 0
        est_counts[a_est] += r
    return est_counts


def a_from_s_int(s_int, num_counting_qubits):
    theta = s_int * np.pi / (2**num_counting_qubits)
    precision = int(num_counting_qubits / (np.log2(10))) + 2
    a = round(np.sin(theta)**2, precision)
    return a

################ Benchmark Loop

# Because circuit size grows significantly with num_qubits
# limit the max_qubits here ...
MAX_QUBITS=8

# Execute program with default parameters
def run(min_qubits=3, max_qubits=8, max_circuits=3, num_shots=100,
        num_state_qubits=1, # default, not exposed to users
        backend_id='simulator'):
    
    print("Amplitude Estimation Benchmark Program - Braket")

    # Clamp the maximum number of qubits
    if max_qubits > MAX_QUBITS:
        print(f"INFO: Amplitude Estimation benchmark is limited to a maximum of {MAX_QUBITS} qubits.")
        max_qubits = MAX_QUBITS
    
    # validate parameters (smallest circuit is 3 qubits)
    num_state_qubits = max(1, num_state_qubits)
    if max_qubits < num_state_qubits + 2:
        print(f"ERROR: AE Benchmark needs at least {num_state_qubits + 2} qubits to run")
        return
    min_qubits = max(max(3, min_qubits), num_state_qubits + 2)

    # Initialize metrics module
    metrics.init_metrics()

    # define custom result handler
    def execution_handler(qc, result, num_qubits, s_int, num_shots):

        # determine fidelity of result set
        num_counting_qubits = int(num_qubits) - num_state_qubits - 1
        counts, fidelity = analyze_and_print_result(qc, result, num_counting_qubits, int(s_int), num_shots)
        metrics.score_metric(num_qubits, s_int, 'fidelity', fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id)

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1):

        # as circuit width grows, the number of counting qubits is increased
        num_counting_qubits = num_qubits - num_state_qubits - 1

        # determine number of circuits to execute for this group
        num_circuits = min(2 ** (num_counting_qubits), max_circuits)

        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")

        # determine range of secret strings to loop over
        if 2**(num_counting_qubits) <= max_circuits:
            s_range = list(range(num_circuits))
        else:
            s_range = np.random.choice(2**(num_counting_qubits), num_circuits, False)

        # loop over limited # of secret strings for this
        for s_int in s_range:
            # create the circuit for given qubit size and secret string, stor time metric
            ts = time.time()

            a_ = a_from_s_int(s_int, num_counting_qubits)

            qc = AmplitudeEstimation(num_state_qubits, num_counting_qubits, a_)
            metrics.store_metric(num_qubits, s_int, 'create_time', time.time() - ts)

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc, num_qubits, s_int, num_shots)
        
        # execute all circuits for this group, aggregate and report metrics when complete
        ex.execute_circuits()
        metrics.aggregate_metrics_for_group(num_qubits)
        metrics.report_metrics_for_group(num_qubits)

    # Alternatively, execute all circuits, aggregate and report metrics
    # ex.execute_circuits()
    # metrics.aggregate_metrics_for_group(input_size)
    # metrics.report_metrics_for_group(input_size)

    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    print("\nControlled Quantum Operator 'cQ' ="); print(cQ_ if cQ_ != None else " ... too large!")
    print("\nQuantum Operator 'Q' ="); print(Q_ if Q_ != None else " ... too large!")
    print("\nAmplitude Generator 'A' ="); print(A_ if A_ != None else " ... too large!")
    print("\nInverse QFT Circuit ="); print(QFTI_ if QC_ != None else "  ... too large!")

    # Plot metrics for all circuit sizes
    metrics.plot_metrics("Benchmark Results - Amplitude Estimation - Braket")

# if main, execute method
if __name__ == '__main__': run()