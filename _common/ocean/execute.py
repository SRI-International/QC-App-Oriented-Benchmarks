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
import importlib
import traceback
from collections import Counter
import logging
import numpy as np
import csv
from itertools import count

from dwave.system.samplers import DWaveSampler
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from neal import SimulatedAnnealingSampler

import HamiltonianCircuitProxy


logger = logging.getLogger(__name__)

# the currently selected provider_backend
backend = None 

# Execution options, passed to transpile method
backend_exec_options = None

# Cached transpiled circuit, used for parameterized execution
cached_circuits = {}


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
def set_execution_target(backend_id='pegasus',
                provider_module_name=None, provider_name=None, provider_backend=None,
                hub=None, group=None, project=None, exec_options=None):
    """
    Used to run jobs on a real hardware
    :param backend_id:  device name. List of available devices depends on the provider
    :param group: used to load IBMQ accounts.
    :param project: used to load IBMQ accounts.
    :param provider_module_name: If using a provider other than IBMQ, the string of the module that contains the
    Provider class.  For example, for honeywell, this would be 'qiskit.providers.honeywell'
    :param provider_name:  If using a provider other than IBMQ, the name of the provider class.
    For example, for Honeywell, this would be 'Honeywell'.
    :provider_backend: a custom backend object created and passed in, use backend_id as identifier
    example usage.

    set_execution_target(backend_id='honeywell_device_1', provider_module_name='qiskit.providers.honeywell',
                        provider_name='Honeywell')
    """
    global backend

    authentication_error_msg = "No credentials for {0} backend found.  Using the simulator instead."
    
    # if a custom provider backend is given, use it ...
    if provider_backend != None:
        backend = provider_backend   

    # create an informative device name
    device_name = backend_id
    metrics.set_plot_subtitle(f"Device = {device_name}")
    #metrics.set_properties( { "api":"ocean", "backend_id":backend_id } )
    
    # save execute options with backend
    global backend_exec_options
    backend_exec_options = exec_options
    
    print(f"... using backend_id = {backend_id}")

# Set the state of the transpilation flags
def set_tranpilation_flags(do_transpile_metrics = True, do_transpile_for_execute = True):
    globals()['do_transpile_metrics'] = do_transpile_metrics
    globals()['do_transpile_for_execute'] = do_transpile_for_execute
    
    
######################################################################
# CIRCUIT EXECUTION METHODS

# Submit circuit for execution
# Execute immediately if possible or put into the list of batched circuits
def submit_circuit(qc:HamiltonianCircuitProxy, group_id, circuit_id, shots=100, params=None):

    # create circuit object with submission time and circuit info
    circuit = { "qc": qc, "group": str(group_id), "circuit": str(circuit_id),
            "submit_time": time.time(), "shots": shots, "params": params }

    if verbose:
        print(f'... submit circuit - group={circuit["group"]} id={circuit["circuit"]} shots={circuit["shots"]} params={circuit["params"]}')
    
    # logger doesn't like unicode, so just log the array values for now
    #logger.info(f'Submitting circuit - group={circuit["group"]} id={circuit["circuit"]} shots={circuit["shots"]} params={str(circuit["params"])}')
    logger.info(f'Submitting circuit - group={circuit["group"]} id={circuit["circuit"]} shots={circuit["shots"]} params={[param[1] for param in params.items()] if params else None}')
    
    # DEVNOTE: not sure why this is needed
    #backend["shots"] = shots
    #backend["group_id"] = group_id
    #backend["circuit_id"] = circuit_id

    if backend != None:
        execute_circuit(circuit)
    else:
        print(f"ERROR: No provider_backend specified, cannot execute program")

    return
    

# Launch execution of one job (circuit)
def execute_circuit(circuit):
        
    sampleset = None

    try:
        logger.info(f"Executing on backend: {backend['backend_id']}")
         
        # perform circuit execution on backend
        logger.info(f'Running trans_qc, shots={backend["shots"]}')
        st = time.time() 

        # DEVNOTE: may need to reset this somewhere!
        embedding = None
        
        '''
        x=0
        while x < 200:
                x = 1 if x==0 else x**2

                qpu = DWaveSampler(token=token, solver={'topology__type': backend["backend_id"]})

                if (embedding==None):
                    sampler = EmbeddingComposite(qpu)
                else:
                    sampler = FixedEmbeddingComposite(qpu, embedding=embedding)

                sampleset = sampler.sample_ising(circuit.qc.h, circuit.qc.J, circuit.qc.shots, annealing_time=x)
        '''
        
        if (embedding==None):
            sampler = EmbeddingComposite(backend)
        else:
            sampler = FixedEmbeddingComposite(backend, embedding=embedding)

        sampleset = sampler.sample_ising(circuit.qc.h, circuit.qc.J, circuit.qc.shots, annealing_time=x)
                
        logger.info(f'Finished Running - {round(time.time() - st, 5)} (ms)')
        if verbose_time: print(f"  *** ocean.sample() time = {round(time.time() - st, 5)}")
            
    except Exception as e:
        print(f'ERROR: Failed to execute {backend["group_id"]} {backend["circuit_id"]}')
        print(f"... exception = {e}")
        if verbose: print(traceback.format_exc())
        return

    # store circuit dimensional metrics
    metrics.store_metric(backend["group_id"], backend["circuit_id"], 'depth', qc_depth)
    metrics.store_metric(backend["group_id"], backend["circuit_id"], 'size', qc_size)
    metrics.store_metric(backend["group_id"], backend["circuit_id"], 'xi', qc_xi)
    metrics.store_metric(backend["group_id"], backend["circuit_id"], 'n2q', qc_n2q)

    metrics.store_metric(backend["group_id"], backend["circuit_id"], 'tr_depth', qc_tr_depth)
    metrics.store_metric(backend["group_id"], backend["circuit_id"], 'tr_size', qc_tr_size)
    metrics.store_metric(backend["group_id"], backend["circuit_id"], 'tr_xi', qc_tr_xi)
    metrics.store_metric(backend["group_id"], backend["circuit_id"], 'tr_n2q', qc_tr_n2q)

    return sampleset

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
    
    
###########################################################################
        
# Test circuit execution
def test_execution():
    pass


