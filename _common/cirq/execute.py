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
# Execute Module - Cirq
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

import cirq
backend = cirq.Simulator()      # Use Cirq Simulator by default

#noise = 'DEFAULT'
noise=None

# Initialize circuit execution module
# Create array of batched circuits and a dict of active circuits 
# Configure a handler for processing circuits on completion

batched_circuits = [ ]
active_circuits = { }
result_handler = None

device=None

# Special object class to hold job information and used as a dict key
class Job:
    pass

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
def set_execution_target(backend_id='simulator', provider_backend=None):
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
        
    # otherwise test for simulator
    elif backend_id == 'simulator':
        backend = cirq.Simulator()
       
    # nothing else is supported yet, default to simulator       
    else:
        print(f"ERROR: Unknown backend_id: {backend_id}, defaulting to Cirq Simulator")
        backend = cirq.Simulator()
        backend_id = "simulator"

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
    
    shots = batched_circuit["shots"]
    
    # Initiate execution 
    job = Job()
    circuit = batched_circuit["qc"]
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
    
    # put job into the active circuits with circuit info
    active_circuits[job] = active_circuit
    #print("... active_circuit = ", str(active_circuit))
    
    ##############
    # Here we complete the job immediately 
    job_complete(job)
    

# Process a completed job
def job_complete (job):
    active_circuit = active_circuits[job]
    
    # get job result (DEVNOTE: this might be different for diff targets)
    result = job.result
    #print("... result = ", str(result))
    
    # counts = result.get_counts(qc)
    # print("Total counts are:", counts)
    
    # get measurement array and shot count
    measurements = result.measurements['result']
    actual_shots = len(measurements)
    #print(f"actual_shots = {actual_shots}")
    
    if actual_shots != active_circuit["shots"]:
        print(f"WARNING: requested shots not equal to actual shots: {actual_shots}")
        
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'elapsed_time',
        time.time() - active_circuit["submit_time"])
        
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_time',
        time.time() - active_circuit["launch_time"])
    
    # If a handler has been established, invoke it here with result object
    if result_handler:
        result_handler(active_circuit["qc"],
            result, active_circuit["group"], active_circuit["circuit"], active_circuit["shots"])
            
    del active_circuits[job]
 
 
# Wait for all executions to complete
def wait_for_completion():

    # check and sleep if not complete
    pass
    
    # return only when all circuits complete


# Test circuit execution
def test_execution():
    pass
    