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

import os, sys
import time
import copy
import qcb_mpi as mpi

# import metrics module relative to top level of package
current_dir = os.path.dirname(os.path.abspath(__file__))

common_dir = os.path.abspath(os.path.join(current_dir, ".."))
#sys.path = [common_dir] + [p for p in sys.path if p != common_dir]

top_dir = os.path.abspath(os.path.join(common_dir, ".."))
sys.path = [top_dir] + [p for p in sys.path if p != top_dir]

#print(sys.path)

# DEVNOTE: for some reason, this does not work - why?
#import _common.metrics as metrics

# instead, need to include the common directory in path and do this
import metrics

# import the CUDA-Q package
import cudaq

verbose = False

#noise = 'DEFAULT'
noise=None

# Initialize circuit execution module
# Create array of batched circuits and a dict of active circuits 
# Configure a handler for processing circuits on completion

batched_circuits = [ ]
active_circuits = { }
result_handler = None

# save the executing device (backend_id) here
device = None

#######################
# SUPPORTING CLASSES

# class BenchmarkResult is used as a compatible return object from execution
class BenchmarkResult(object):

    def __init__(self, cq_result):
        super().__init__()
        self.cq_result = cq_result

    def get_counts(self, qc=0):
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
def set_execution_target(backend_id=None, provider_backend=None,
        hub=None, group=None, project=None, exec_options=None,
        context=None):
    """
    Set the backend execution target.
    :param backend_id:  device name. List of available devices depends on the provider
    :provider_backend: a custom backend object created and passed in, use backend_id as identifier
    
    example usage.
    set_execution_target(backend_id='aqt_qasm_simulator', 
                        provider_backende=aqt.backends.aqt_qasm_simulator)
    """
    global backend   
    
    # in case anyone uses a name similar to that used in other APIs
    if backend_id == "cudaq_simulator":
        backend_id = None
    
    # default to nvidia execution engine if None passed in
    if backend_id == None:
        backend_id="nvidia"
        
    # if a custom provider backend is given, set it; currently unused
    if provider_backend != None:
        backend = provider_backend
    
    # now set the execution target to the given backend_id
    if mpi.enabled():
        option_string="mgpu,fp32"
    else:
        option_string="fp32"
    cudaq.set_target(backend_id, option=option_string)
    
    # create an informative device name used by the metrics module
    device_name = backend_id
    metrics.set_plot_subtitle(f"Device = {device_name}")
    
    print(f"... configure execution for target backend_id = {backend_id}")

# CUDA-Q supports several different models of noise. In this default
# case, we use the modeling of depolarization noise only. This
# depolarization will result in the qubit state decaying into a mix
# of the basis states, |0> and |1>, with a user provided probability.
# DEVNOTE: there are no two qubit error settings here
def set_default_noise_model():
    global noise
    
    # We will begin by defining an empty noise model that we will add
    # our depolarization channel to.
    noise = cudaq.NoiseModel()

    # We define a depolarization channel setting the probability
    # of the qubit state being scrambled to `1.0`.
    depolarization = cudaq.DepolarizationChannel(0.04)
    
    phase_flip = cudaq.PhaseFlipChannel(0.2)

    for i in range(30):
        noise.add_channel('x', [i], depolarization)
        noise.add_channel('y', [i], depolarization)
        noise.add_channel('z', [i], depolarization)
        
        noise.add_channel('h', [i], depolarization)
        
        # consider adding this
        #noise.add_channel('x', [i], phase_flip)
        #noise.add_channel('y', [i], phase_flip)
        #noise.add_channel('z', [i], phase_flip)
        
        noise.add_channel('rx', [i], depolarization)
        noise.add_channel('ry', [i], depolarization)
        noise.add_channel('rz', [i], depolarization)
    
    if verbose:
        print(f"  ... just set DEFAULT noise model to: {noise}")

# Configure execution to use the given noise model
def set_noise_model(noise_model = None):
    
    global noise
    noise = noise_model
    
    if verbose:
        print(f"... just set noise model to: {noise}")


# Submit circuit for execution
# This version executes immediately and calls the result handler
def submit_circuit (qc, group_id, circuit_id, shots=100):

    # store circuit in array with submission time and circuit info
    batched_circuits.append(
        { "qc": qc, "group": str(group_id), "circuit": str(circuit_id),
            "submit_time": time.time(), "shots": shots }
    )
    #print("... submit circuit - ", str(batched_circuits[len(batched_circuits)-1]))
    
    # DEVNOTE: execute immediately for now, so that we don't accumulate elapsed time while in queue
    execute_circuits()
    
    
# Launch execution of all batched circuits
def execute_circuits ():
    for batched_circuit in batched_circuits:
        execute_circuit(batched_circuit)
    batched_circuits.clear()
    
# Launch execution of one batched circuit
def execute_circuit (batched_circuit):
    if verbose:
        print(f'... execute_circuit({batched_circuit["group"]}, {batched_circuit["circuit"]})')

    active_circuit = copy.copy(batched_circuit)
    active_circuit["launch_time"] = time.time()
    
    num_shots = batched_circuit["shots"]
    
    # Initiate execution 
    circuit = batched_circuit["qc"]
    
    # create a pseudo-job to perform metrics processing upon return
    job = Job()
    
    # draw the circuit, but only for debugging
    # print(cudaq.draw(circuit[0], *circuit[1]))
    
    ts = time.time()
    
    # call sample() on circuit with its list of arguments
    if verbose: print(f"... during exec, noise model is: {noise}")
    if noise is None:
        if verbose: print("... executing without noise")
        result = cudaq.sample(circuit[0], *circuit[1], shots_count=num_shots)
    else:
        if verbose: print("... executing WITH noise")
        result = cudaq.sample(circuit[0], *circuit[1], shots_count=num_shots, noise_model=noise)
    
    # control results print at benchmark level
    #if verbose: print(result)
        
    exec_time = time.time() - ts
    
    # store the result object on the job for processing in job_complete
    job.executor_result = result 
    job.exec_time = exec_time
    
    if verbose:
        print(f"... result = {len(result)} {result}")
        ''' for debugging, a better way to see the counts, as the type of result is something Quake
        for key, val in result.items():
            print(f"... {key}:{val}")
        '''
        print(f"... register names = {result.register_names}")
        print(result.dump())
        #print(f"... register dump = {result.get_register_counts('__global__').dump()}")
        #result.get_register_counts("b1").dump()
        #print(result.get_sequential_data())
        
    # put job into the active circuits with circuit info
    active_circuits[job] = active_circuit
    #print("... active_circuit = ", str(active_circuit))
    
    # ***********************************
    
    # store circuit dimensional metrics
    # DEVNOTE: this is not accurate; it is provided so the volumetric plots show something
    
    # compute depth and gate counts based on number of qubits
    qc_size = int(active_circuit["group"])
    qc_depth = 4 * pow(qc_size, 2)

    qc_xi = 0.5

    qc_n2q = int(qc_depth * 0.75)
    qc_tr_depth = qc_depth
    qc_tr_size = qc_size
    qc_tr_xi = qc_xi
    qc_tr_n2q = qc_n2q
    
    # store circuit dimensional metrics
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'depth', qc_depth)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'size', qc_size)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'xi', qc_xi)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'n2q', qc_n2q)

    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_depth', qc_tr_depth)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_size', qc_tr_size)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_xi', qc_tr_xi)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_n2q', qc_tr_n2q)
    
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
    
    # create a compatible Result object to return to the caller
    result = BenchmarkResult(cq_result)
    
    # counts = result.get_counts(qc)
    # print("Total counts are:", counts)

    # store time metrics
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'elapsed_time',
        time.time() - active_circuit["submit_time"])
       
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_time',
        job.exec_time)
    
    # If a handler has been established, invoke it here with result object
    if result_handler:
        result_handler(active_circuit["qc"],
            result, active_circuit["group"], active_circuit["circuit"], active_circuit["shots"])
    
    # DEVNOTE: a hack to store the last group identifier for use in the job management functions
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

    if verbose:
        print(f"... throttling execution, active={len(active_circuits)}, batched={len(batched_circuits)}")
    
    # DEVNOTE: execution is currently synchronous, so force execution of any batched circuits
    execute_circuits()

    global last_group
    group = last_group
    
    # call completion handler with the group id
    if completion_handler != None:
        completion_handler(group)
                
    '''  DEVNOTE: this or something similar could be used later for throtttling
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

    if verbose:
        print("... finalize_execution")
        
    # DEVNOTE: execution is currently synchronous, so force execution of any batched circuits
    execute_circuits()
    
    '''   DEVNOTE: this or something similar could be used later for throtttling
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
 
 
###################################################################
# IMMEDIATE EXECUTION

# Execute a circuit with its parameters and return result without batching

# Launch execution of one batched circuit
def execute_circuit_immed (circuit: list, num_shots: int):
    if verbose:
        print(f'... execute_circuit_immed({circuit}, {num_shots})')

    #active_circuit = copy.copy(batched_circuit)
    #active_circuit["launch_time"] = time.time()
    
    #num_shots = batched_circuit["shots"]
    
    # Initiate execution 
    #circuit = batched_circuit["qc"]
    
    # create a pseudo-job to perform metrics processing upon return
    job = Job()
    
    # draw the circuit, but only for debugging
    # print(cudaq.draw(circuit[0], *circuit[1]))
    
    ts = time.time()
    
    # call sample() on circuit with its list of arguments
    if verbose: print(f"... during exec, noise model is: {noise}")
    if noise is None:
        if verbose: print("... executing without noise")
        result = cudaq.sample(circuit[0], *circuit[1], shots_count=num_shots)
    else:
        if verbose: print("... executing WITH noise")
        result = cudaq.sample(circuit[0], *circuit[1], shots_count=num_shots, noise_model=noise)
    
    # control results print at benchmark level
    #if verbose: print(result)
        
    exec_time = time.time() - ts
    
    # store the result object on the job for processing in job_complete
    job.executor_result = result 
    job.exec_time = exec_time
    
    if verbose:
        print(f"... result = {len(result)} {result}")
        ''' for debugging, a better way to see the counts, as the type of result is something Quake
        for key, val in result.items():
            print(f"... {key}:{val}")
        '''
        print(f"... register names = {result.register_names}")
        print(result.dump())
        #print(f"... register dump = {result.get_register_counts('__global__').dump()}")
        #result.get_register_counts("b1").dump()
        #print(result.get_sequential_data())
        
    # put job into the active circuits with circuit info
    #active_circuits[job] = active_circuit
    #print("... active_circuit = ", str(active_circuit))
    
    # ***********************************
    
    # store circuit dimensional metrics
    # DEVNOTE: this is not accurate; it is provided so the volumetric plots show something
    """
    # compute depth and gate counts based on number of qubits
    qc_size = int(active_circuit["group"])
    qc_depth = 4 * pow(qc_size, 2)

    qc_xi = 0.5

    qc_n2q = int(qc_depth * 0.75)
    qc_tr_depth = qc_depth
    qc_tr_size = qc_size
    qc_tr_xi = qc_xi
    qc_tr_n2q = qc_n2q
    
    # store circuit dimensional metrics
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'depth', qc_depth)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'size', qc_size)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'xi', qc_xi)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'n2q', qc_n2q)

    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_depth', qc_tr_depth)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_size', qc_tr_size)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_xi', qc_tr_xi)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_n2q', qc_tr_n2q)
    
    ##############
    # Here we complete the job immediately 
    job_complete(job)
    """
    """
    # klunky way to know the last group executed 
    #last_group = None 

    # Process a completed job
    def job_complete (job):
        #active_circuit = active_circuits[job]
    """         
    # get job result (DEVNOTE: this might be different for diff targets)
    #cq_result = job.result()
    cq_result = result

    # create a compatible Result object to return to the caller
    result = BenchmarkResult(cq_result)
    
    # counts = result.get_counts(qc)
    # print("Total counts are:", counts)
    """
    # store time metrics
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'elapsed_time',
        time.time() - active_circuit["submit_time"])
       
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_time',
        job.exec_time)
    
    # If a handler has been established, invoke it here with result object
    if result_handler:
        result_handler(active_circuit["qc"],
            result, active_circuit["group"], active_circuit["circuit"], active_circuit["shots"])
    
    # DEVNOTE: a hack to store the last group identifier for use in the job management functions
    group = active_circuit["group"]
    global last_group
    last_group = group
                           
    del active_circuits[job]
    """
    
    # return a Qiskit-like result object 
    return result
 
    
