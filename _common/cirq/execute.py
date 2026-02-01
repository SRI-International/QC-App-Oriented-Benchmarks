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
from _common import metrics

import cirq
backend = cirq.Simulator()      # Use Cirq Simulator by default

import logging
# logger for this module
logger = logging.getLogger(__name__)

from collections import OrderedDict
from typing import Dict, List

# Option to compute normalized depth during execution (can disable to reduce overhead in large circuits)
use_normalized_depth = True

# Option to perform explicit transpile to collect depth metrics
# (disabled after first circuit in iterative algorithms)
do_transpile_metrics = True

# Print progress of execution
verbose = False

# Print additional time metrics for each stage of execution
verbose_time = False

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

    # On initialize, always set trnaspilation for metrics and execute to True
    set_transpilation_flags(do_transpile_metrics=True, do_transpile_for_execute=True)
    
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

    # store circuit dimensional metrics
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'depth', qc_depth)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'size', qc_size)

    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_depth', qc_tr_depth)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_size', qc_tr_size)

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

#####

# Get circuit metrics fom the circuit passed in
def get_circuit_metrics(qc):

    logger.info('Entering get_circuit_metrics')
    
    # obtain initial circuit size metrics
    qc_depth = len(cirq.Circuit(qc.all_operations()))
    qc_size = count_gate_operations(qc)
    qc_count_ops = count_ops(qc)
    
    return qc_depth, qc_size, qc_count_ops

def count_ops(circuit: cirq.Circuit) -> OrderedDict[str, int]:
    """Count each operation kind in the circuit.

    Args:
        circuit: A Cirq circuit object.

    Returns:
        OrderedDict: a breakdown of how many operations of each kind, sorted by amount.
    """
    count_ops: Dict[str, int] = {}
    num_measurements = 0

    def process_op(op):
        nonlocal num_measurements

        # Handle MeasurementGate separately to count number of measurements
        if isinstance(op.gate, cirq.MeasurementGate):
            num_measurements += len(op.qubits)
            return  # Skip counting MeasurementGate separately

        # Handle custom gates by providing a meaningful name
        gate_name = str(op.gate)
        
        # Handle controlled gates specifically
        if isinstance(op.gate, cirq.ControlledGate):
            # Decompose the controlled gate and process each sub-operation
            decomposed_ops = cirq.decompose(op)
            # Check if decomposed_ops is a list
            if isinstance(decomposed_ops, list):
                for inner_op in decomposed_ops:
                    process_op(inner_op)
            else:
                for inner_op in decomposed_ops.all_operations():
                    process_op(inner_op)
        
        # Handle custom gates by decomposing them if they have a .to_gate method
        elif '.to_gate' in gate_name:
            qr_state = [cirq.GridQubit(i, 0) for i in range(op.gate.num_qubits)]
            decomposed_ops = cirq.Circuit(cirq.decompose(op.gate.on(*qr_state)))
            for inner_op in decomposed_ops.all_operations():
                process_op(inner_op)

        count_ops[gate_name] = count_ops.get(gate_name, 0) + 1


    # Process each operation in the circuit
    for moment in circuit:
        for op in moment.operations:
            process_op(op)

    # Include the number of measurements in the results
    count_ops["Measurement"] = num_measurements

    return OrderedDict(sorted(count_ops.items(), key=lambda kv: kv[1], reverse=True))
    
#####
# Set the state of the transpilation flags
def set_transpilation_flags(do_transpile_metrics = True, do_transpile_for_execute = True):
    globals()['do_transpile_metrics'] = do_transpile_metrics
    globals()['do_transpile_for_execute'] = do_transpile_for_execute

#####
def count_gate_operations(circuit: cirq.Circuit) -> int:
    """Calculate the total number of gate operations in the circuit.

    Args:
        circuit: A Cirq circuit object.

    Returns:
        int: Total number of gate operations in the circuit.
    """
    total_gate_operations = 0

    def process_op(op):
        nonlocal total_gate_operations

        if isinstance(op.gate, cirq.MeasurementGate):
            # Count each qubit in the MeasurementGate as a separate operation
            total_gate_operations += len(op.qubits)
        else:
            total_gate_operations += 1

            # Handle controlled gates specifically
            if isinstance(op.gate, cirq.ControlledGate):
                # Decompose the controlled gate and process each sub-operation
                decomposed_ops = cirq.decompose(op)
                # Check if decomposed_ops is a list
                if isinstance(decomposed_ops, list):
                    for inner_op in decomposed_ops:
                        process_op(inner_op)
                else:
                    for inner_op in decomposed_ops.all_operations():
                        process_op(inner_op)
                        
            # Handle custom gates by decomposing them if they have a .to_gate method
            elif hasattr(op.gate, 'to_gate'):
                qr_state = [cirq.GridQubit(i, 0) for i in range(op.gate.num_qubits())]
                decomposed_ops = cirq.decompose(op.gate.on(*qr_state))
                # Check if decomposed_ops is a list
                if isinstance(decomposed_ops, list):
                    for inner_op in decomposed_ops:
                        process_op(inner_op)
                else:
                    for inner_op in decomposed_ops.all_operations():
                        process_op(inner_op)
            else:
                # Count the operation based on the number of qubits
                total_gate_operations += len(op.qubits)
                
    # Process each operation in the circuit
    for moment in circuit:
        for op in moment.operations:
            process_op(op)

    return total_gate_operations
    
#####
# Transpile the circuit to obtain normalized size metrics against a common basis gate set
def transpile_for_metrics(qc):

    logger.info('Entering transpile_for_metrics')
    #print("*** Before transpile ...")
    st = time.time()

    # Compile the circuit for CZ Target Gateset.
    gateset = cirq.CZTargetGateset(allow_partial_czs=True)    
    optimized_circuit = cirq.optimize_for_target_gateset(qc, gateset=gateset)

    #To ensure optimized circuit is equivalent to original circuit in terms of final measurements
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(qc, optimized_circuit)     
    # print("\nCircuit compiled for CZ Target Gateset (Optimized/Transpiled ckt):", optimized_circuit, "\n", sep="\n")

    qc_tr_depth = len(cirq.Circuit(optimized_circuit.all_operations()))
    qc_tr_size = count_gate_operations(optimized_circuit)    # total no. of gate operations (excluding barrier)
    qc_tr_count_ops = count_ops(optimized_circuit)
    # print(f"*** after transpile: 'qc_tr_depth' {qc_tr_depth} 'qc_tr_size' {qc_tr_size} 'qc_tr_count_ops' {qc_tr_count_ops}\n")
    
    logger.info(f'transpile_for_metrics - {round(time.time() - st, 5)} (ms)')
    if verbose_time: print(f"  *** transpile_for_metrics() time = {round(time.time() - st, 5)}")
    
    return qc_tr_depth, qc_tr_size, qc_tr_count_ops