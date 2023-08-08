#
# BlueQubit Executor Interface 
#
# This module implements the 'executor' intercept for execution on BlueQubit Quantum Simulators
#
#

import bluequbit

# This class BlueQubitResult provides the return data from Blue Qubit runs. 
# It simply adds a get_counts() method with the optional qc argument.
# (the BQ Result object only has getcounts(), but the benchmarks require get_counts(qc))
class BlueQubitResult(object):

    def __init__(self, bluequbit_result):
        super().__init__()
        self.bluequbit_result = bluequbit_result
        self.counts = bluequbit_result.get_counts()

    def get_counts(self, qc=None):
        counts = self.counts       
        return counts

# This function is called by the QED-C execution pipeline when specified as the 'executor'       
def run(qc, backend_name, backend_provider, shots=100):

    if backend_name == 'BlueQubit-CPU': device = 'cpu'
    if backend_name == 'BlueQubit-GPU': device = 'gpu'
    
    bq_result = backend_provider.run(qc, device=device, shots=shots)
    
    #print(f"... result = {bq_result}\n{bq_result.__dir__()}")
    
    # wrap the BlueQubit result in a (semi-)standard Result object with a counts dict
    # This Result object will be passed to the result_handler callback for each benchmark
    result = BlueQubitResult(bq_result)
    
    # store the execution time reported by BlueQubit
    result.exec_time = bq_result.run_time_ms / 1000
    
    return result