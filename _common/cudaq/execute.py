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
import json
import math
import os

from _common import qcb_mpi as mpi
from _common import metrics

# import the CUDA-Q package
import cudaq

verbose = False

# Parallel execution configuration
_parallel_config = {
    "mode": None,
    "num_gpus": 0,
    "initialized": False
}


###################################################################
# PARALLEL EXECUTION SUPPORT
#
# These functions enable parallel execution of independent circuits
# across multiple GPUs using either:
# - nvidia-mqpu target with cudaq.sample_async()
# - MPI rank distribution with gather/scatter
###################################################################

def _distribute_circuits_contiguous(circuits: list, num_workers: int) -> list:
    """
    Distribute circuits into contiguous blocks for each worker.

    Args:
        circuits: List of circuits to distribute
        num_workers: Number of workers (GPUs or MPI ranks)

    Returns:
        List of lists, where each inner list is the circuits for one worker
    """
    n = len(circuits)
    if num_workers <= 0 or n == 0:
        return [circuits]

    # Calculate base size and remainder
    base_size = n // num_workers
    remainder = n % num_workers

    blocks = []
    start = 0
    for i in range(num_workers):
        # First 'remainder' workers get one extra circuit
        size = base_size + (1 if i < remainder else 0)
        if size > 0:
            blocks.append(circuits[start:start + size])
        else:
            blocks.append([])
        start += size

    return blocks


def _get_block_indices(total_items: int, num_workers: int) -> list:
    """
    Get (start, end) indices for contiguous blocks.

    Returns:
        List of (start, end) tuples for each worker
    """
    if num_workers <= 0 or total_items == 0:
        return [(0, total_items)]

    base_size = total_items // num_workers
    remainder = total_items % num_workers

    indices = []
    start = 0
    for i in range(num_workers):
        size = base_size + (1 if i < remainder else 0)
        indices.append((start, start + size))
        start += size

    return indices


def _detect_available_gpus() -> int:
    """
    Detect the number of available GPUs for parallel execution.
    First checks environment variable, then queries cudaq.

    Returns:
        int: Number of available GPUs, or 1 if detection fails
    """
    # Check environment variable first
    ngpus_env = os.environ.get("CUDAQ_MQPU_NGPUS")
    if ngpus_env:
        try:
            return int(ngpus_env)
        except ValueError:
            pass

    # Try to query cudaq for GPU count
    try:
        # Note: This requires the nvidia-mqpu target to be available
        # We'll return 1 as safe default if we can't detect
        return 1
    except Exception:
        return 1


def _execute_parallel_mqpu(circuits: list, num_shots: int, num_gpus: int = None) -> list:
    """
    Execute circuits in parallel using nvidia-mqpu target with sample_async.

    Args:
        circuits: List of [kernel, params] circuit tuples
        num_shots: Number of shots per circuit
        num_gpus: Number of GPUs to use (None = auto-detect)

    Returns:
        List of count dictionaries in original circuit order
    """
    # Set target for multi-QPU execution (nvidia-mqpu is the correct target name)
    # Can also set CUDAQ_MQPU_NGPUS environment variable to control GPU count
    cudaq.set_target("nvidia-mqpu")

    # Get number of QPUs actually available from the platform
    target = cudaq.get_target()
    available_qpus = target.num_qpus()

    # Use requested GPUs if specified, but cap at available QPUs
    if num_gpus is not None:
        if num_gpus > available_qpus:
            print(f"... mqpu mode: requested {num_gpus} GPUs but only {available_qpus} available")
        qpu_count = min(num_gpus, available_qpus)
    else:
        qpu_count = available_qpus

    print(f"... mqpu mode: using {qpu_count} of {available_qpus} available GPU(s)")

    if qpu_count < 2:
        # Fall back to sequential if only one GPU
        print(f"... falling back to sequential (need 2+ GPUs for parallel)")
        return _execute_sequential(circuits, num_shots)

    print(f"... executing {len(circuits)} circuits across {qpu_count} GPUs (mqpu parallel)")

    # Distribute circuits into contiguous blocks per GPU
    blocks = _distribute_circuits_contiguous(circuits, qpu_count)
    block_indices = _get_block_indices(len(circuits), qpu_count)

    # Submit all circuits asynchronously
    # Track (global_index, future) pairs to maintain order
    async_results = []

    for gpu_id, block in enumerate(blocks):
        start_idx, _ = block_indices[gpu_id]
        for local_idx, circuit in enumerate(block):
            global_idx = start_idx + local_idx
            kernel, params = circuit[0], circuit[1]

            # Submit asynchronously to specific GPU
            if noise is None:
                future = cudaq.sample_async(kernel, *params,
                                           shots_count=num_shots, qpu_id=gpu_id)
            else:
                future = cudaq.sample_async(kernel, *params,
                                           shots_count=num_shots, qpu_id=gpu_id,
                                           noise_model=noise)
            async_results.append((global_idx, future))

    # Collect results in original order
    all_results = [None] * len(circuits)
    for global_idx, future in async_results:
        counts = future.get()  # Blocks until this circuit completes
        # Convert cudaq result to dict
        count_dict = {k: v for k, v in counts.items()}
        all_results[global_idx] = count_dict

    return all_results


def _execute_parallel_mpi(circuits: list, num_shots: int) -> list:
    """
    Execute circuits in parallel using MPI rank distribution.

    Each MPI rank processes a contiguous block of circuits on its assigned GPU.
    Results are gathered to rank 0.

    Args:
        circuits: List of [kernel, params] circuit tuples
        num_shots: Number of shots per circuit

    Returns:
        List of count dictionaries in original circuit order (complete on rank 0,
        partial results on other ranks)
    """
    if not mpi.enabled():
        return _execute_sequential(circuits, num_shots)

    rank = mpi.rank
    size = mpi.size

    # All ranks need to know total circuit count for distribution
    num_circuits = len(circuits)

    # Calculate this rank's block
    block_indices = _get_block_indices(num_circuits, size)
    my_start, my_end = block_indices[rank]

    # Get this rank's circuits
    my_circuits = circuits[my_start:my_end]

    if mpi.leader():
        print(f"... executing {num_circuits} circuits across {size} MPI ranks")

    # Execute local circuits sequentially on this rank's GPU
    local_results = []
    for circuit in my_circuits:
        kernel, params = circuit[0], circuit[1]
        if noise is None:
            counts = cudaq.sample(kernel, *params, shots_count=num_shots)
        else:
            counts = cudaq.sample(kernel, *params, shots_count=num_shots, noise_model=noise)
        # Convert to dict
        local_results.append({k: v for k, v in counts.items()})

    # Gather results to rank 0
    all_results = mpi.gather(local_results)

    if mpi.leader():
        # Flatten gathered results in correct order
        flattened = []
        for rank_results in all_results:
            flattened.extend(rank_results)
        return flattened
    else:
        # Non-leader ranks return their local results
        return local_results


def _execute_sequential(circuits: list, num_shots: int) -> list:
    """
    Execute circuits sequentially (fallback mode).

    Args:
        circuits: List of [kernel, params] circuit tuples
        num_shots: Number of shots per circuit

    Returns:
        List of count dictionaries
    """
    results = []
    for circuit in circuits:
        kernel, params = circuit[0], circuit[1]
        if noise is None:
            counts = cudaq.sample(kernel, *params, shots_count=num_shots)
        else:
            counts = cudaq.sample(kernel, *params, shots_count=num_shots, noise_model=noise)
        results.append({k: v for k, v in counts.items()})
    return results

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
    backend_options = {"option" : "fp32"}
    if mpi.enabled():
        backend_options = {"option" : "mgpu,fp32"}
    elif exec_options is not None and isinstance(exec_options, str):
        try:
            backend_options = json.loads(exec_options)
            for key, value in backend_options.items():
                if not isinstance(key, str):
                    raise ValueError("`exec_options` keys must be strings")
                if not isinstance(value, (str, int, float, bool)):
                    raise ValueError("`exec_options` values must be str, int, float, or bool")
        except:
            print(f"    ... Invalid `exec_options`; using default options.")
            
    cudaq.set_target(backend_id, **backend_options)
    
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
    
    ############
    
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
    
    # number of qubits is stored in the "group" field
    qc_size = int(active_circuit["group"])
    
    # obtain initial circuit metrics
    qc_depth, qc_size, qc_count_ops, qc_xi, qc_n2q = get_circuit_metrics(circuit, qc_size)
    
    qc_tr_depth = qc_depth
    qc_tr_size = qc_size
    qc_tr_count_ops = qc_count_ops
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
 

# Get circuit metrics fom the circuit passed in
def get_circuit_metrics(qc, qc_size):
    
    # the new code is implemented in CUDA-Q ver 0.13+, if it fails fall back to old code
    try:
    
        # get resource info from cudaq
        resources = cudaq.estimate_resources(qc[0], *qc[1])
        #resources_str = str(resources)
        
        #print(resources)
        
        # Get total gates (not needed as we use the .count() function)
        # import re
        # total_match = re.search(r'Total # of gates:\s*(\d+)', resources_str)
        # total_gates = int(total_match.group(1)) if total_match else 0
        
        total_gates = resources.count()
        
        two_qubit_gates = 0
        two_qubit_gates += resources.count_controls('x', 1)
        two_qubit_gates += resources.count_controls('y', 1)
        two_qubit_gates += resources.count_controls('z', 1)
        two_qubit_gates += resources.count_controls('r1', 1)
        two_qubit_gates += resources.count_controls('rx', 1)
        two_qubit_gates += resources.count_controls('ry', 1)
        two_qubit_gates += resources.count_controls('rz', 1)
        
        #print(f"... depth = {resources.count_depth()}") # doesn't exist yet
        #print(f"Total: {total_gates}, 2-qubit: {two_qubit_gates}")

        # obtain an estimate of circuit depth
        qc_depth = estimate_depth(qc_size, total_gates, two_qubit_gates)
        #print(qc_depth)
        
        # this exact computation is not used, currently, as it is slow
        #qc_depth = compute_circuit_depth(qc[0](*qc[1]))
        #print(qc_depth)
        
        qc_xi = two_qubit_gates / max(total_gates, 1)
        qc_n2q = two_qubit_gates
        
        # not currently used
        qc_count_ops = total_gates
        
    # this is used for CUDA-Q versions 0.12 or earlier   
    except:    
        
        # compute depth and gate counts based on number of qubits
        qc_depth = 4 * pow(qc_size, 2)

        qc_xi = 0.5

        qc_n2q = int(qc_depth * 0.75)
        qc_tr_depth = qc_depth
        qc_tr_size = qc_size
        qc_tr_xi = qc_xi
        qc_tr_n2q = qc_n2q
        
        # not currently used
        qc_count_ops = qc_depth
         
    return qc_depth, qc_size, qc_count_ops, qc_xi, qc_n2q


# Make estimate of circuit depth using heuristic approach; depth not provided by cudaq
# This is somewhat simplistic, but we have not found better solution
def estimate_depth(num_qubits, total_gates, two_qubit_gates):
    N = num_qubits
    K = two_qubit_gates
    S = total_gates - K
    
    # Theoretical minimum (perfect packing - 2Qs and 1Qs and MZ)
    depth_min = math.ceil(2*K / N) + math.ceil(S / N) + 1
    
    # Theoretical maximum, sparse distribution
    depth_max = K + S + 1
    
    # use xi factor as proxy for level of sparseness in layout (larger = sparser)
    qc_xi = two_qubit_gates / max(total_gates, 1)
    depth_range = depth_max - depth_min
    depth_estimate = math.ceil(depth_min + depth_range * qc_xi)
    
    #print(f"  ... min = {depth_min}, max = {depth_max}, depth = {depth_estimate}")
    
    # Realistic estimate (assume 40% packing efficiency)  (used with min only)
    # depth_estimate = depth_min * 2.5
    
    return depth_estimate
           

#########################################################################  

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

# NEW CODE 

# The following functions have been moved here from the hamlib benchmark.
# This transition and merge of new code developed in the hamlib benchmark is a work-in-progress.
# The code below will be gradually integrated into this module in stages (TL: 250519)

# Execute a circuit with its parameters and return result without batching

# Launch execution of one batched circuit
def execute_circuit_immed (circuit: list, num_shots: int):
    if verbose:
        print(f'... execute_circuit_immed({circuit}, {num_shots})')

    try:
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
            
    except Exception as ex:
        print(f"ERROR attempting to compute cicuit metrics")
        print(ex)
        
    # put job into the active circuits with circuit info
    #active_circuits[job] = active_circuit
    #print("... active_circuit = ", str(active_circuit))
    
    # ***********************************
    qc_depth, qc_size, qc_count_ops, qc_xi, qc_n2q = 0, 0, 0, 0, 0
    try:
        # number of qubits is stored in the "group" field
        #qc_size = int(circuit["group"])
        qc_size = circuit[1][0]   # the first item after the kernel is alway num_qubits
        
        # obtain initial circuit metrics
        qc_depth, qc_size, qc_count_ops, qc_xi, qc_n2q = get_circuit_metrics(circuit, qc_size)

    except Exception as ex:
        print(f"ERROR attempting to compute cicuit metrics")
        print(ex)
    
    qc_tr_depth = qc_depth
    qc_tr_size = qc_size
    qc_tr_count_ops = qc_count_ops
    qc_tr_xi = qc_xi
    qc_tr_n2q = qc_n2q
    
    """
    # store circuit dimensional metrics
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'depth', qc_depth)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'size', qc_size)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'xi', qc_xi)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'n2q', qc_n2q)

    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_depth', qc_tr_depth)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_size', qc_tr_size)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_xi', qc_tr_xi)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_n2q', qc_tr_n2q)
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

# This function performs multiple circuit execution
def execute_circuits_immed(
        backend_id: str = None,
        circuits: list = None,
        num_shots: int = 100,
        parallel_mode: str = "sequential",
        num_gpus: int = None
    ) -> list:
    """
    Execute a list of circuits on the given backend with the given number of shots.

    Args:
        backend_id: Backend identifier (currently unused for cudaq)
        circuits: List of [kernel, params] circuit tuples
        num_shots: Number of shots per circuit
        parallel_mode: Execution mode:
            - "sequential": Execute circuits one at a time (default)
            - "mqpu": Use nvidia-mqpu target with sample_async for parallel execution
            - "mpi": Use MPI rank distribution for parallel execution
            - "auto": Automatically select best mode based on available resources
        num_gpus: Number of GPUs to use for parallel execution (None = auto-detect)

    Returns:
        ExecResult object with get_counts() method
    """

    if verbose:
        print(f"... execute_circuits_immed({backend_id}, {len(circuits)}, {num_shots}, mode={parallel_mode})")

    # Handle empty or single circuit case
    if not circuits or len(circuits) == 0:
        return ExecResult([])

    if len(circuits) == 1:
        # Single circuit - no benefit from parallelization
        result = execute_circuit_immed(circuits[0], num_shots)
        return ExecResult([result.get_counts()])

    # Report the requested execution mode
    if parallel_mode != "sequential":
        print(f"... execute_circuits_immed: {len(circuits)} circuits, mode={parallel_mode}")

    # Choose execution path based on mode
    if parallel_mode == "mqpu":
        try:
            counts_array = _execute_parallel_mqpu(circuits, num_shots, num_gpus)
        except Exception as ex:
            print(f"... MQPU parallel execution failed: {ex}")
            print("... falling back to sequential execution")
            counts_array = _execute_sequential_with_metrics(circuits, num_shots)

    elif parallel_mode == "mpi":
        if mpi.enabled():
            try:
                counts_array = _execute_parallel_mpi(circuits, num_shots)
            except Exception as ex:
                print(f"... MPI parallel execution failed: {ex}")
                print("... falling back to sequential execution")
                counts_array = _execute_sequential_with_metrics(circuits, num_shots)
        else:
            if verbose:
                print("... MPI not enabled, using sequential execution")
            counts_array = _execute_sequential_with_metrics(circuits, num_shots)

    elif parallel_mode == "auto":
        # Auto-select: prefer MPI if enabled, then MQPU if multiple GPUs, else sequential
        if mpi.enabled() and mpi.size > 1:
            try:
                counts_array = _execute_parallel_mpi(circuits, num_shots)
            except Exception as ex:
                print(f"... MPI parallel execution failed: {ex}")
                counts_array = _execute_sequential_with_metrics(circuits, num_shots)
        else:
            # Try MQPU if we might have multiple GPUs
            detected_gpus = num_gpus if num_gpus else _detect_available_gpus()
            if detected_gpus > 1:
                try:
                    counts_array = _execute_parallel_mqpu(circuits, num_shots, num_gpus)
                except Exception as ex:
                    print(f"... MQPU parallel execution failed: {ex}")
                    counts_array = _execute_sequential_with_metrics(circuits, num_shots)
            else:
                counts_array = _execute_sequential_with_metrics(circuits, num_shots)

    else:
        # Default: sequential execution (original behavior)
        counts_array = _execute_sequential_with_metrics(circuits, num_shots)

    # Construct a Result object with counts structure to match circuits
    results = ExecResult(counts_array)

    return results


def _execute_sequential_with_metrics(circuits: list, num_shots: int) -> list:
    """
    Execute circuits sequentially using existing execute_circuit_immed.
    This preserves the original behavior including any metrics collection.
    """
    counts_array = []
    for circuit in circuits:
        result = execute_circuit_immed(circuit, num_shots)
        counts_array.append(result.get_counts())
    return counts_array
    
        
# class ExecResult is made for multi-circuit runs. 
class ExecResult(object):

    def __init__(self, counts):
        super().__init__()
        
        # Store the count distributions as they will be returned
        # A single count object for one circuit, and an array of count object for array of circuits
        if isinstance(counts, list):
            if len(counts) < 2:
                self.counts = counts[0]
            else:
                self.counts = counts
        else:
            self.counts = counts

    def get_counts(self, qc=0):
        return self.counts       
