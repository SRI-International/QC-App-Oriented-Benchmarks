# (C) Quantum Economic Development Consortium (QED-C) 2021.
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
###########################
# Execute Module - Qiskit
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
import os

# AWS imports: Import Braket SDK modules
from braket.aws import AwsDevice
from braket.devices import LocalSimulator

# Enter the S3 bucket you created during onboarding in the environment variables queried here
my_bucket = os.environ.get("AWS_BRAKET_S3_BUCKET", "my_bucket") # the name of the bucket
my_prefix = os.environ.get("AWS_BRAKET_S3_PREFIX", "my_prefix") # the name of the folder in the bucket#
s3_folder = (my_bucket, my_prefix)

# the selected Braket device
device = None

# default noise model, can be overridden using set_noise_model
#noise = NoiseModel()
noise = None

# Initialize circuit execution module
# Create array of batched circuits and a dict of active circuits 
# Configure a handler for processing circuits on completion

batched_circuits = []
active_circuits = {}
result_handler = None

verbose = False

# Print additional time metrics for each stage of execution
verbose_time = False

import logging
# logger for this module
logger = logging.getLogger(__name__)

# Option to compute normalized depth during execution (can disable to reduce overhead in large circuits)
use_normalized_depth = True

# Option to perform explicit transpile to collect depth metrics
do_transpile_metrics = True

# Special object class to hold job information and used as a dict key
class Job:
    pass

# Initialize the execution module, with a custom result handler
def init_execution(handler):
    global batched_circuits, result_handler
    batched_circuits.clear()
    active_circuits.clear()
    result_handler = handler

    # On initialize, always set trnaspilation for metrics and execute to True
    set_transpilation_flags(do_transpile_metrics=True, do_transpile_for_execute=True)

# Set the backend for execution
def set_execution_target(backend_id='simulator'):
    """
    Used to run jobs on a real hardware
    :param backend_id:  device name. List of available devices depends on the provider

    example usage.

    set_execution_target(backend_id='arn:aws:braket:::device/quantum-simulator/amazon/sv1')
    """
    global device

    if backend_id == None or backend_id == "":
        device = None
        
    elif backend_id == "simulator":
        device = LocalSimulator()
        
    else:
        device = AwsDevice(backend_id)
    
    if verbose:
        print(f"... using Braket device = {device}")
    
    # create an informative device name
    device_name = device.name
    device_str = str(device)
    if device_str.find(":device/") > 0:
        idx = device_str.rindex(":device/")
        device_name = device_str[idx+8:-1]
        
    metrics.set_plot_subtitle(f"Device = {device_name}")
        
    return device

'''
def set_noise_model(noise_model = None):
    # see reference on setting up noise in Braket here: 
    # https://github.com/aws/amazon-braket-examples/blob/main/examples/braket_features/Simulating_Noise_On_Amazon_Braket.ipynb
    global noise
    noise = noise_model
'''

# Submit circuit for execution
# This version executes immediately and calls the result handler
def submit_circuit(qc, group_id, circuit_id, shots=100):
    # store circuit in array with submission time and circuit info
    batched_circuits.append(
        { "qc": qc, "group": str(group_id), "circuit": str(circuit_id),
            "submit_time": time.time(), "shots": shots }
    )
    # print("... submit circuit - ", str(batched_circuits[len(batched_circuits)-1]))


# Launch execution of all batched circuits
def execute_circuits():
    for batched_circuit in batched_circuits:
        execute_circuit(batched_circuit)
    batched_circuits.clear()


# Launch execution of one batched circuit
def execute_circuit(batched_circuit):
    active_circuit = copy.copy(batched_circuit)
    active_circuit["launch_time"] = time.time()
        
    # Initiate execution (currently, waits for completion)
    job = Job()
    circuit = batched_circuit["qc"]

    # obtain initial circuit metrics
    qc_depth, qc_size, qc_count_ops = get_circuit_metrics(circuit)

    # default the normalized transpiled metrics to the same, in case exec fails
    qc_tr_depth = qc_depth
    qc_tr_size = qc_size
    qc_tr_count_ops = qc_count_ops
    #print(f"... before tp: {qc_depth} {qc_size} {qc_count_ops}")

    try:    
        # transpile the circuit to obtain size metrics using normalized basis
        if do_transpile_metrics and use_normalized_depth:
            qc_tr_depth, qc_tr_size, qc_tr_count_ops = transpile_for_metrics(circuit)
            
            # we want to ignore elapsed time contribution of transpile for metrics (normalized depth)
            active_circuit["launch_time"] = time.time()

    except Exception as e:
        print(f'ERROR: Failed to execute circuit {active_circuit["group"]} {active_circuit["circuit"]}')
        print(f"... exception = {e}")
        return

    # store circuit dimensional metrics
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'depth', qc_depth)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'size', qc_size)

    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_depth', qc_tr_depth)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_size', qc_tr_size)
    
    job.result = braket_execute(batched_circuit["qc"], batched_circuit["shots"])
    
    # put job into the active circuits with circuit info
    active_circuits[job] = active_circuit
    # print("... active_circuit = ", str(active_circuit))

    ##############
    # Here we complete the job immediately 
    job_complete(job)


# Process a completed job
def job_complete(job):
    active_circuit = active_circuits[job]

    # get job result 
    result = job.result
    
    if result != None:
        #print(f"... result = {result}")
        #print(f"... result metadata = {result.task_metadata}")
        #print(f"... shots = {result.task_metadata.shots}")
        
        # this appears to include queueing time, so may not be what is needed
        if verbose:
            print(f"... task times = {result.task_metadata.createdAt} {result.task_metadata.endedAt}")
        
        # this only applies to simulator and does not appear to reflect actual exec time
        #print(f"... execution duration = {result.additional_metadata.simulatorMetadata.executionDuration}")
        
        # counts = result.measurement_counts
        # print("Total counts are:", counts)
        
        # obtain timing info from the results object
        '''
        result_obj = result.to_dict()
        results_obj = result.to_dict()['results'][0]
        #print(f"result_obj = {result_obj}")
        #print(f"results_obj = {results_obj}")
        
        if "time_taken" in result_obj:
            exec_time = result_obj["time_taken"]
        
        elif "time_taken" in results_obj:
            exec_time = results_obj["time_taken"]
            
        else:
            #exec_time = 0;
            exec_time = time.time() - active_circuit["launch_time"]
        '''
        # currently, we just use elapsed time, which includes queue time
        exec_time = time.time() - active_circuit["launch_time"]
        #print(f"exec time = {exec_time}")

        metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'elapsed_time',
                             time.time() - active_circuit["launch_time"])
       
        metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_time',
                             exec_time)

        # If a handler has been established, invoke it here with result object
        if result_handler:
            result_handler(active_circuit["qc"],
                           result, active_circuit["group"], active_circuit["circuit"])

    del active_circuits[job]
    job = None


# Wait for all executions to complete (not implemented yet)
def wait_for_completion():
    # check and sleep if not complete
    pass

    # return only when all circuits complete

# Braket circuit execution code; this code waits for completion and returns result object
def braket_execute(qc, shots=100):

    if qc == None:
        print("ERROR: No circuit to execute")
        return None
 
    # run circuit on selected device
    #print(f"device={device}")
    
    # immediate completion if Local Simulator
    if isinstance(device, LocalSimulator):
        result = device.run(qc, shots).result()
        
    # returns task object if managed device
    else:
        task = device.run(qc, s3_folder, shots)
    
        # if result is a Task object, loop until complete
        task_id = task.id
        
        # get status
        status = task.state()
        
        done = False
        pollcount = 0
        while not done:
            status = task.state()
            #print('-- Status of (reconstructed) task:', status)
            pollcount += 1
            
            if status == "FAILED":
                result = None   
                print("... circuit execution failed")
                break
                
            if status == "CANCELLED":
                result = None   
                print("... circuit execution cancelled")
                break
            
            elif status == "COMPLETED":
                result = task.result()
                break
            
            else:
                if verbose: 
                    #if pollcount <= 1: print('')
                    print('.', end='')

            # delay a bit, increasing the delay periodically 
            sleeptime = 1
            if pollcount > 10: sleeptime = 10 
            elif pollcount > 20: sleeptime = 30
            time.sleep(sleeptime)
            
        if pollcount > 1:
            if verbose: print('')
    
    # return result in either case     
    return result
       

# Test circuit execution
def test_execution():
    pass

###############
# Get circuit metrics fom the circuit passed in
def get_circuit_metrics(qc):

    logger.info('Entering get_circuit_metrics')
    # print(qc)
    
    # obtain initial circuit size metrics
    qc_depth = qc.depth
    qc_size = len(qc.instructions)    # total gate operations
    qc_count_ops = count_operations(qc)

    return qc_depth, qc_size, qc_count_ops


def count_operations(circuit):
    operation_counts = {}

    for instruction in circuit.instructions:
        operation = instruction.operator.name
        if operation in operation_counts:
            operation_counts[operation] += 1
        else:
            operation_counts[operation] = 1

    return operation_counts


######
# Set the state of the transpilation flags
def set_transpilation_flags(do_transpile_metrics = True, do_transpile_for_execute = True):
    globals()['do_transpile_metrics'] = do_transpile_metrics
    globals()['do_transpile_for_execute'] = do_transpile_for_execute

######
# Transpile the circuit to obtain normalized size metrics against a common basis gate set
def transpile_for_metrics(qc):

    logger.info('Entering transpile_for_metrics')
    #print("*** Before transpile ...")
    #print(qc)
    st = time.time()

    #kept Transpiled Depth and Algorithmic Depth same, to get the Volumetric Positioning Plot 
    qc_tr_depth = qc.depth
    qc_tr_size =len(qc.instructions)    # total gate operations 
    qc_tr_count_ops = count_operations(qc)
    # print(f"*** after transpile: 'qc_tr_depth' {qc_tr_depth} 'qc_tr_size' {qc_tr_size} 'qc_tr_count_ops' {qc_tr_count_ops}\n")
    
    
    logger.info(f'transpile_for_metrics - {round(time.time() - st, 5)} (ms)')
    if verbose_time: print(f"  *** transpile_for_metrics() time = {round(time.time() - st, 5)}")
    
    return qc_tr_depth, qc_tr_size, qc_tr_count_ops
