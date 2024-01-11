#
# BlueQubit Executor Interface 
#
# This module implements the 'executor' intercept for execution on BlueQubit Quantum Simulators
#
#


verbose = False

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
# The device argument is passed as an "execute option"     
def run(qc, backend_name, backend, shots=100, device='cpu'):
    
    if verbose:
        print(f"  ... executing circuit on backend={backend_name} device={device}")
    
    # Perform execution of circuit on the BlueQubit device
    bq_result = backend.run(qc, device=device, shots=shots)
    
    if verbose:
        print(f"... result = {bq_result}")
    
    # wrap the BlueQubit result in a (semi-)standard Result object with a counts dict
    # This Result object will be passed to the result_handler callback for each benchmark
    result = BlueQubitResult(bq_result)
    
    # store the execution time reported by BlueQubit
    result.exec_time = bq_result.run_time_ms / 1000
    
    return result