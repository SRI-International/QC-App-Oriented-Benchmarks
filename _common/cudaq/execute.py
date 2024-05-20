# (C) Quantum Economic Development Consortium (QED-C) 2024.
# Technical Advisory Committee on Standards and Benchmarks (TAC)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################
# Execute Module - CUDA Quantum
#
# This module provides a way to submit a series of circuits to be executed in a batch.
# When the batch is executed, each circuit is launched as a 'job' to be executed on the target system.
# Upon completion, the results from each job are processed in a custom 'result handler' function 
# in order to calculate metrics such as fidelity. Relevant benchmark metrics are stored for each circuit
# execution, so they can be aggregated and presented to the user.
#

import time
import copy
import metrics

import cudaq

###import cirq
###backend = cirq.Simulator()      # Use Cirq Simulator by default

#noise = 'DEFAULT'
noise=None

# Initialize circuit execution module
# Create array of batched circuits and a dict of active circuits 
# Configure a handler for processing circuits on completion

batched_circuits = [ ]
active_circuits = { }
result_handler = None

device=None

#######################
# SUPPORTING CLASSES

# class BenchmarkResult is made for sessions runs. This is because
# qiskit primitive job result instances don't have a get_counts method 
# like backend results do. As such, a get counts method is calculated
# from the quasi distributions and shots taken.
class BenchmarkResult(object):

    def __init__(self, cq_result):
        super().__init__()
        self.cq_result = cq_result

    def get_counts(self, qc=0):
        #counts = None
        '''
        self.qiskit_result.quasi_dists[0].binary_probabilities()
        for key in counts.keys():
            counts[key] = int(counts[key] * self.qiskit_result.metadata[0]['shots']) 
        '''
        counts = self.cq_result
        return counts

# Special Job object class to hold job information for custom executors
class Job:
    local_job = True
    unique_job_id = 1001
    executor_result = None
    
    def __init__(self):
        Job.unique_job_id = Job.unique_job_id + 1
        self.this_job_id = Job.unique_job_id      
        
    def job_id(self):
        return self.this_job_id
    
    def status(self):
        return JobStatus.DONE
        
    def result(self):
        return self.executor_result

#######################
# 

# Initialize the execution module, with a custom result handler
def init_execution (handler):
    global batched_circuits, result_handler
    batched_circuits.clear()
    active_circuits.clear()
    result_handler = handler
    
    # create an informative device name
    # this should be move to set_execution_target method later
    device_name = "simulator"
    metrics.set_plot_subtitle(f"Device = {device_name}")


# Set the backend for execution
def set_execution_target(backend_id='simulator', provider_backend=None,
        hub=None, group=None, project=None, exec_options=None,
        context=None):
    """
    Used to run jobs on a real hardware
    :param backend_id:  device name. List of available devices depends on the provider
    :provider_backend: a custom backend object created and passed in, use backend_id as identifier
    
    example usage.
    set_execution_target(backend_id='aqt_qasm_simulator', 
                        provider_backende=aqt.backends.aqt_qasm_simulator)
    """
    global backend   
    
    # if a custom provider backend is given, use it ...
    if provider_backend != None:
        backend = provider_backend
    '''   
    # otherwise test for simulator
    elif backend_id == 'simulator':
        backend = cirq.Simulator()
       
    # nothing else is supported yet, default to simulator       
    else:
        print(f"ERROR: Unknown backend_id: {backend_id}, defaulting to Cirq Simulator")
        backend = cirq.Simulator()
        backend_id = "simulator"
    '''
    # create an informative device name
    device_name = backend_id
    metrics.set_plot_subtitle(f"Device = {device_name}")


def set_noise_model(noise_model = None):
    # see reference on NoiseModel here https://quantumai.google/cirq/noise
    global noise
    noise = noise_model


# Submit circuit for execution
# This version executes immediately and calls the result handler
def submit_circuit (qc, group_id, circuit_id, shots=100):

    # store circuit in array with submission time and circuit info
    batched_circuits.append(
        { "qc": qc, "group": str(group_id), "circuit": str(circuit_id),
            "submit_time": time.time(), "shots": shots }
    )
    #print("... submit circuit - ", str(batched_circuits[len(batched_circuits)-1]))
    
    
# Launch execution of all batched circuits
def execute_circuits ():
    for batched_circuit in batched_circuits:
        execute_circuit(batched_circuit)
    batched_circuits.clear()
    
# Launch execution of one batched circuit
def execute_circuit (batched_circuit):

    active_circuit = copy.copy(batched_circuit)
    active_circuit["launch_time"] = time.time()
    
    num_shots = batched_circuit["shots"]
    
    # Initiate execution 
    circuit = batched_circuit["qc"]
    '''
    if type(noise) == str and noise == "DEFAULT":
        # depolarizing noise on all qubits
        circuit = circuit.with_noise(cirq.depolarize(0.05))
    elif noise is not None:
        # otherwise we expect it to be a NoiseModel
        # see documentation at https://quantumai.google/cirq/noise
        circuit = circuit.with_noise(noise)
    
    # experimental, for testing AQT device
    if device != None:
        circuit.device=device
        device.validate_circuit(circuit)

    job.result = backend.run(circuit, repetitions=shots)
    '''
    
    # create a pseudo-job to perform metrics processing upon return
    job = Job()
    
    #print(cudaq.draw(circuit[0], circuit[1], circuit[2], circuit[3]))
    
    ts = time.time()
    result = cudaq.sample(circuit[0], circuit[1], circuit[2], circuit[3], shots_count=num_shots)
    exec_time = time.time() - ts
    
    # store the result object on the job for processing in job_complete
    job.executor_result = result 
    job.exec_time = exec_time
    
    print(f"... result = {result}")
    
    # put job into the active circuits with circuit info
    active_circuits[job] = active_circuit
    #print("... active_circuit = ", str(active_circuit))
    
    ##############
    # Here we complete the job immediately 
    job_complete(job)

# klunky way to know the last group executed 
last_group = None 

# Process a completed job
def job_complete (job):
    active_circuit = active_circuits[job]
        
    # get job result (DEVNOTE: this might be different for diff targets)
    cq_result = job.result()
    print("... result2 = ", str(cq_result))
    
    result = BenchmarkResult(cq_result)
    
    # counts = result.get_counts(qc)
    # print("Total counts are:", counts)
    '''
    # get measurement array and shot count
    measurements = result.measurements['result']
    actual_shots = len(measurements)
    #print(f"actual_shots = {actual_shots}")
    
    if actual_shots != active_circuit["shots"]:
        print(f"WARNING: requested shots not equal to actual shots: {actual_shots}")
    '''   
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'elapsed_time',
        time.time() - active_circuit["submit_time"])
       
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_time',
        job.exec_time)
    
    # If a handler has been established, invoke it here with result object
    if result_handler:
        result_handler(active_circuit["qc"],
            result, active_circuit["group"], active_circuit["circuit"], active_circuit["shots"])
    

    group = active_circuit["group"]
    global last_group
    last_group = group
                           
    del active_circuits[job]
 
        
######################################################################
# JOB MANAGEMENT METHODS

# Job management involves coordinating the batching, queueing,
# and completion processing of circuits that are submitted for execution. 

# Throttle the execution of active and batched jobs.
# Wait for active jobs to complete.  As each job completes,
# check if there are any batched circuits waiting to be executed.
# If so, execute them, removing them from the batch.
# Execute the user-supplied completion handler to allow user to 
# check if a group of circuits has been completed and report on results.
# Then, if there are no more circuits remaining in batch, exit,
# otherwise continue to wait for additional active circuits to complete.

def throttle_execution(completion_handler=metrics.finalize_group):
    #logger.info('Entering throttle_execution')

    #if verbose:
        #print(f"... throttling execution, active={len(active_circuits)}, batched={len(batched_circuits)}")

    global last_group
    group = last_group
    
    # call completion handler with the group id
    if completion_handler != None:
        completion_handler(group)
                
    '''
    # check and sleep if not complete
    done = False
    pollcount = 0
    while not done:
    
        # check if any jobs complete
        check_jobs(completion_handler)

        # return only when all jobs complete
        if len(batched_circuits) < 1:
            break
            
        # delay a bit, increasing the delay periodically 
        sleeptime = 0.25
        if pollcount > 6: sleeptime = 0.5
        if pollcount > 60: sleeptime = 1.0
        time.sleep(sleeptime)
        
        pollcount += 1
    
    if verbose:
        if pollcount > 0: print("") 
        #print(f"... throttling execution(2), active={len(active_circuits)}, batched={len(batched_circuits)}")
    '''
    
    
# Wait for all active and batched circuits to complete.
# Execute the user-supplied completion handler to allow user to 
# check if a group of circuits has been completed and report on results.
# Return when there are no more active circuits.
# This is used as a way to complete all groups of circuits and report results.

def finalize_execution(completion_handler=metrics.finalize_group, report_end=True):

    #if verbose:
        #print("... finalize_execution")
    '''
    # check and sleep if not complete
    done = False
    pollcount = 0
    while not done:
    
        # check if any jobs complete
        check_jobs(completion_handler)

        # return only when all jobs complete
        if len(active_circuits) < 1:
            break
            
        # delay a bit, increasing the delay periodically 
        sleeptime = 0.10                        # was 0.25
        if pollcount > 6: sleeptime = 0.20      # 0.5
        if pollcount > 60: sleeptime = 0.5      # 1.0
        time.sleep(sleeptime)
        
        pollcount += 1
    
    if verbose:
        if pollcount > 0: print("")
    '''
    # indicate we are done collecting metrics (called once at end of app)
    if report_end:
        metrics.end_metrics()
    '''   
    # also, close any active session at end of the app
    global session
    if report_end and session != None:
        if verbose:
            print(f"... closing active session: {session_count}\n")
        
        session.close()
        session = None
    '''
 
# Wait for all executions to complete
def wait_for_completion():

    # check and sleep if not complete
    pass
    
    # return only when all circuits complete


# Test circuit execution
def test_execution():
    pass
    