#
# Nvidia cuQuantum Executor Interface 
#
# This module implements the 'executor' intercept for execution on Nvidia cuQuantum Simulators
#
#

from qiskit import transpile, Aer
import time

import sys
sys.path.insert(1, "_common")
sys.path.insert(1, "_common/qiskit")

import execute

# Set False to disable calculation of 'normalized depth' (long time for large circuits)
# (if False, only 'algorithmic' depth will be displayed in plots)
execute.use_normalized_depth = False
execute.max_jobs_active = 1

verbose = False

# Save handle to the cuQuantum Aer simulator
simulator = None

use_cusvaer = True
precision = "single"

# This class CuQuantumResult provides the return data from Nvidia cuQuantum runs. 
# It simply adds a get_counts() method with the optional qc argument.
class CuQuantumResult(object):

    def __init__(self, bluequbit_result):
        super().__init__()
        #self.bluequbit_result = bluequbit_result
        #self.counts = bluequbit_result.get_counts()

    def get_counts(self, qc=None):
        counts = self.counts       
        return counts

# This function is called by the QED-C execution pipeline when specified as the 'executor'  
# The device argument is passed as an "execute option"     
def run(qc, backend_name, backend, shots=100, device='cpu'):
    global simulator
    
    if verbose:
        print(f"  ... executing circuit on backend={backend_name} device={device}")
    
    # first time, create the simulator backend
    if simulator is None:
        if verbose:
            print("... create aer_simulator_statevector backend")

        simulator = Aer.get_backend('aer_simulator_statevector')

        simulator.set_option('cusvaer_enable', use_cusvaer)
        simulator.set_option('precision', precision)

    st = time.time()

    # transpile first
    qc_transpiled = transpile(qc, simulator)

    # Perform execution of circuit on the cuQuantum simulator
    job = simulator.run(qc_transpiled)
    cq_result = job.result()

    if verbose:
        print(f"... result = {cq_result}")
    
    # wrap the BlueQubit result in a (semi-)standard Result object with a counts dict
    # This Result object will be passed to the result_handler callback for each benchmark
    #result = BlueQubitResult(bq_result)
    
    # store the execution time calculated here
    cq_result.exec_time = time.time() - st
    
    return cq_result
