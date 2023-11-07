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
# This module is derived from _common/ocean/execute.py
###########################
# Execute Module - Qatalyst
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
import importlib
import traceback
from collections import Counter
import logging
import numpy as np
import csv
from itertools import count
import json
import math

import qci_client

import HamiltonianCircuitProxy


logger = logging.getLogger(__name__)

# the currently selected provider_backend
backend = None 
device_name = None

# Execution options, passed to transpile method
backend_exec_options = None

# Cached transpiled circuit, used for parameterized execution
cached_circuits = {}

# embedding variables cached during execution
embedding_flag = True
embedding = None

# n_samples defaults to 1 for EQC, has no effect on CSample
n_samples = 1
# number of decimal places for time values (seconds)
time_precision = 5
# number of nanoseconds per second
ns = 1e9
##########################
# JOB MANAGEMENT VARIABLES 

# Configure a handler for processing circuits on completion
# user-supplied result handler
result_handler = None

# Print progress of execution
verbose = True

# Print additional time metrics for each stage of execution
verbose_time = False


######################################################################
# INITIALIZATION METHODS

# Initialize the execution module, with a custom result handler
def init_execution(handler):
    global result_handler
    result_handler = handler
    
    cached_circuits.clear()
    

# Set the backend for execution
def set_execution_target(backend_id='eqc1',
                provider_module_name=None, provider_name=None, provider_backend=None,
                hub=None, group=None, project=None, exec_options=None):
    """
    Used to run jobs on a real hardware
    :param backend_id:  device name. List of available devices depends on the provider
    :param group: NOT USED
    :param project: NOT USED
    :param provider_module_name: NOT USED
    :param provider_name:  NOT USED
    :provider_backend: Reference to instantiated QciClient

    set_execution_target(backend_id='eqc1', provider_backend=client)
    """
    
    global backend_exec_options, backend, device_name
    # create an informative device name
    device_name = backend_id
    backend = provider_backend
    metrics.set_plot_subtitle(f"Device = {device_name}")
    #metrics.set_properties( { "api":"ocean", "backend_id":backend_id } )
    
    # save execute options with backend
    backend_exec_options = exec_options
    print(f"... using backend_id = {backend_id}")
    print(f"... using device_name = {device_name}")

# not using this for now, but will enable if machinery requires it
# # Set the state of the transpilation flags
# def set_embedding_flag(embedding_flag = True):   
#     globals()['embedding_flag'] = embedding_flag
    
    
######################################################################
# CIRCUIT EXECUTION METHODS

# Submit circuit for execution
# Execute immediately if possible or put into the list of batched circuits
def submit_circuit(qc:HamiltonianCircuitProxy, group_id, circuit_id, shots=100, params=None):

    # create circuit object with submission time and circuit info
    circuit = { "qc": qc, "group": str(group_id), "circuit": str(circuit_id),
            "submit_time": time.time(), "shots": shots, "params": params }

    if verbose:
        print(f'... submit circuit - group={circuit["group"]} id={circuit["circuit"]}  params={circuit["params"]}')
    
    # logger doesn't like unicode, so just log the array values for now
    logger.info(f'Submitting circuit - group={circuit["group"]} id={circuit["circuit"]} params={str(circuit["params"])}')
    
    if device_name is not None:
        samples = execute_circuit(circuit)
    else:
        print(f"ERROR: No device_name specified, cannot execute program")

    return
    

# Launch execution of one job (circuit)
def execute_circuit(circuit):
    global backend, device_name
    logging.info('Entering execute_circuit')

    active_circuit = copy.copy(circuit)
    st = active_circuit["launch_time"] = time.time()
    active_circuit["pollcount"] = 0 
    
    shots = circuit["shots"]
    qc = circuit["qc"]
    # annealing_time=circuit["params"][0]
    
    sampleset = None

    global device_name
    try:
        logger.info(f"Executing on backend: {device_name}")
         
        # perform circuit execution on backend
        logger.info(f'Running trans_qc, shots={circuit["shots"]}')

        
        #***************************************
        # execute on Qatalyst
    
        # prepare the sampler with embedding or use a cached embedding
        ts = time.time() 
        
        # load the file
        H = qc.H
        print("Uploading file shape", H.shape)
        file_response = backend.upload_file(H, file_type="hamiltonian")
        file_id = file_response["file_id"]
        print("Got file_id", file_id)
        opt_exec_time = time.time() - ts
        
        # perform the annealing operation
        ts = time.time() 
        print("Building job request", device_name, n_samples)
        job_body = backend.build_job_body(hamiltonian_file_id=file_id,
                                            job_type="sample-hamiltonian",
                                            job_params={"sampler_type": device_name, 
                                                        "n_samples": n_samples})
        job_response = backend.process_job(job_body=job_body, job_type="sample-hamiltonian")
        job_metrics = job_response["job_info"]["metrics"]
        qatalyst_total_job_time_ns = job_metrics['time_ns']['wall']['total']

        # the total amount of time that the job spent in queue waiting to run
        qatalyst_total_queue_time_ns = job_metrics['time_ns']['wall']['queue']['total']

        device = device_name.replace("-", "_")
        device = device.replace("eqc", "dirac_")
        provider_name = "qphoton"
        # device_reference = f"{device}_device"
        # controller_name = f"{device}_controller"
        # Total time for the job inside the QPU subsystem
        qpu_total_time_ns = job_metrics['provider'][provider_name]['time_ns']['wall']['total']
        
        # Processing time for the job inside the QPU subsystem
        qpu_processing_time_ns = job_metrics['provider'][provider_name]['time_ns']['wall']['processing']['total']
        
        # Total queue time within the QPU subsystem
        qpu_queue_time_ns = job_metrics['provider'][provider_name]['time_ns']['wall']['queue']['total']
        
        elapsed_time = time.time() - ts
        # exec_time = (sampleset.info["timing"]["qpu_access_time"] / 1000000)
        exec_time = qpu_processing_time_ns / ns
        
        # opt_exec_time += (sampleset.info["timing"]["total_post_processing_time"] / 1000000)
        # opt_exec_time += (sampleset.info["timing"]["qpu_access_overhead_time"] / 1000000)
        opt_exec_time += (qpu_total_time_ns / ns - exec_time)
        opt_exec_time += qatalyst_total_queue_time_ns

        if verbose_time: print(json.dumps(metrics, indent=2))
        
        #if verbose: print(sampleset.info)
        results = job_response["results"]
        if verbose: print(results)

        elapsed_time = round(elapsed_time, time_precision)
        exec_time = round(exec_time, time_precision)
        opt_exec_time = round(opt_exec_time, time_precision)
        
        logger.info(f'Finished Running qci_client.execute() - elapsed, exec, opt time = {elapsed_time} {exec_time} {opt_exec_time} (sec)')
        if verbose_time: print(f"  ... qci_client.execute() elapsed, exec, opt time = {elapsed_time}, {exec_time}, {opt_exec_time} (sec)")
        
        metrics.store_metric(circuit["group"], circuit["circuit"], 'elapsed_time', elapsed_time)
        metrics.store_metric(circuit["group"], circuit["circuit"], 'exec_time', exec_time)
        metrics.store_metric(circuit["group"], circuit["circuit"], 'opt_exec_time', opt_exec_time)
    
        # this seems specific to MaxCut- dealing with that in the code for that
        # particular benchmark
        # def process_to_bitstring(cut_list):
        #     # DEVNOTE : Check if the mapping is correct
        #     # Convert 1 to 0 and -1 to 1
        #     cut_list = ['0' if i == 1 else '1' for i in cut_list]
        #     # (-(cut_array - 1)/2).astype('int32')
        #     return "".join(cut_list)
        
        # for Neal simulator (mimicker), compute counts from each record returned, one per shot
        samples = results["samples"]
        # rewrite Ising Hamiltonian as bits
        for i in range(len(samples)):
            for j in range(len(samples[i])):
                if samples[i][j] < 0:
                    samples[i][j] = 0
        logger.info("Converted samples from Ising Hamiltonian to Binary")
        bitstrings = [''.join(map(str, sol)) for sol in samples]
        result = dict(zip(bitstrings, results["counts"]))

        result_handler(circuit["qc"], result, circuit["group"], circuit["circuit"], circuit["shots"])
        logger.info("Handled result")
            
    except Exception as e:
        print(f'ERROR: Failed to execute {circuit["group"]} {circuit["circuit"]}')
        print(f"... exception = {e}")
        if verbose: print(traceback.format_exc())
        return

    # store circuit dimensional metrics

    qc_depth = 1
    qc_size = 1
    qc_xi = 0.5
    qc_n2q = 1
    
    qc_tr_depth = 1
    qc_tr_size = 1
    qc_tr_xi = 0.5
    qc_tr_n2q = 1
    
    metrics.store_metric(circuit["group"], circuit["circuit"], 'depth', qc_depth)
    metrics.store_metric(circuit["group"], circuit["circuit"], 'size', qc_size)
    metrics.store_metric(circuit["group"], circuit["circuit"], 'xi', qc_xi)
    metrics.store_metric(circuit["group"], circuit["circuit"], 'n2q', qc_n2q)

    metrics.store_metric(circuit["group"], circuit["circuit"], 'tr_depth', qc_tr_depth)
    metrics.store_metric(circuit["group"], circuit["circuit"], 'tr_size', qc_tr_size)
    metrics.store_metric(circuit["group"], circuit["circuit"], 'tr_xi', qc_tr_xi)
    metrics.store_metric(circuit["group"], circuit["circuit"], 'tr_n2q', qc_tr_n2q)

    
    #print(sampleset)
    logger.info("Stored metrics")
    
    return samples

# Get circuit metrics fom the circuit passed in
def get_circuit_metrics(qc):

    logger.info('Entering get_circuit_metrics')
    #print(qc)
    
    # obtain initial circuit size metrics
    qc_depth = qc.depth()
    qc_size = qc.size()
    qc_count_ops = qc.count_ops()
    qc_xi = 0
    qc_n2q = 0 
    
    # iterate over the ordereddict to determine xi (ratio of 2 qubit gates to one qubit gates)
    n1q = 0; n2q = 0
    if qc_count_ops != None:
        for key, value in qc_count_ops.items():
            if key == "measure": continue
            if key == "barrier": continue
            if key.startswith("c") or key.startswith("mc"):
                n2q += value
            else:
                n1q += value
        qc_xi = n2q / (n1q + n2q)
        qc_n2q = n2q
    
    return qc_depth, qc_size, qc_count_ops, qc_xi, qc_n2q
 

# Wait for all active and batched circuits to complete.
# Execute the user-supplied completion handler to allow user to 
# check if a group of circuits has been completed and report on results.
# Return when there are no more active circuits.
# This is used as a way to complete all groups of circuits and report results.

def finalize_execution(completion_handler=metrics.finalize_group, report_end=True):

    if verbose:
        print("... finalize_execution")
    
    for group in metrics.circuit_metrics:
        try:
            int(group)
        except ValueError:
            continue
        if verbose:
            print(f"... Completing metrics for group {group}")
        completion_handler(group)
    # indicate we are done collecting metrics (called once at end of app)
    if report_end:
        metrics.end_metrics()
       
    
###########################################################################
        
# Test circuit execution
def test_execution():
    pass


