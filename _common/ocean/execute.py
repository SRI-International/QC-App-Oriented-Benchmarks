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
import traceback
import logging
import json
import math

from dwave.system import EmbeddingComposite, FixedEmbeddingComposite

# this seems to be required for embedding to work

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

# The "num_sweeps" variable used by Neal simulator to generate reasonable distributions for testing.
# A value of 1 will provide the most random behavior.
num_sweeps = 10

##########################
# JOB MANAGEMENT VARIABLES 

# Configure a handler for processing circuits on completion
# user-supplied result handler
result_handler = None

# Print progress of execution
verbose = False

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
    global backend, device_name

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
def set_embedding_flag(embedding_flag = True):   
    globals()['embedding_flag'] = embedding_flag
    
    
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
    
    ''' DEVNOTE: doesn't work with our simple annealing time params
    logger.info(f'Submitting circuit - group={circuit["group"]} id={circuit["circuit"]} shots={circuit["shots"]} params={[param[1] for param in params.items()] if params else None}')
    '''
    
    if backend != None:
        execute_circuit(circuit)
    else:
        print("ERROR: No provider_backend specified, cannot execute program")

    return
    

# Launch execution of one job (circuit)
def execute_circuit(circuit):
    logging.info('Entering execute_circuit')

    active_circuit = copy.copy(circuit)
    st = active_circuit["launch_time"] = time.time()
    active_circuit["pollcount"] = 0 
    
    shots = circuit["shots"]
    qc = circuit["qc"]
    annealing_time=circuit["params"][0]
    
    sampleset = None

    try:
        logger.info(f"Executing on backend: {device_name}")
         
        # perform circuit execution on backend
        logger.info(f'Running trans_qc, shots={circuit["shots"]}')

        # this flag is true on the first iteration of a group
        if embedding_flag:
            globals()["embedding"] = None
            total_elapsed_time = 0
            total_exec_time = 0
            total_opt_time = 0     
        
        #***************************************
        # execute on D-Wave Neal simulator (not really simulator, for us it 'mimics' D-Wave behavior)
        if device_name == "pegasus":
            sampler = backend
            
            # for simulation purposes, add a little time for embedding
            ts = time.time() 
            
            #if (embedding_flag):
                #time.sleep(0.3)
                
            opt_exec_time = time.time() - ts
            
            # mimic the annealing operation
            ts = time.time() 
            
            num_sweeps = int(math.log(annealing_time, 2))
            
            sampleset = sampler.sample_ising(qc.h, qc.J, num_reads=shots, num_sweeps=num_sweeps, annealing_time=annealing_time)
            sampleset.resolve()
            
            elapsed_time = time.time() - ts  
            elapsed_time *= (annealing_time / 8)  # funky way to fake elapsed time
            exec_time = elapsed_time / 2        # faking exec time too
            opt_exec_time += elapsed_time / 8
            
        #***************************************
        # execute on D-Wave hardware
        else:
        
            # prepare the sampler with embedding or use a cached embedding
            ts = time.time() 
            
            if (embedding_flag):
                if verbose: print("... CREATE embedding")
                sampler = EmbeddingComposite(backend)
                            
            else:                
                if verbose: print("... USE embedding")
                sampler = FixedEmbeddingComposite(backend, embedding=embedding)

            opt_exec_time = time.time() - ts
            
            # perform the annealing operation
            ts = time.time() 
            
            sampleset = sampler.sample_ising(qc.h, qc.J, num_reads=shots, annealing_time=annealing_time)
            sampleset.resolve()
            
            elapsed_time = time.time() - ts
            exec_time = (sampleset.info["timing"]["qpu_access_time"] / 1000000)
            
            opt_exec_time += (sampleset.info["timing"]["total_post_processing_time"] / 1000000)
            opt_exec_time += (sampleset.info["timing"]["qpu_access_overhead_time"] / 1000000)

            # if embedding context is returned and we haven't already cached it, cache it here
            if embedding == None:
                if "embedding_context" in sampleset.info:
                    globals()["embedding"] = sampleset.info["embedding_context"]["embedding"]
            
            if verbose_time: print(json.dumps(sampleset.info["timing"], indent=2))
            
        #if verbose: print(sampleset.info)
        if verbose: print(sampleset.record)

        elapsed_time = round(elapsed_time, 5)
        exec_time = round(exec_time, 5)
        opt_exec_time = round(opt_exec_time, 5)
        
        logger.info(f'Finished Running ocean.execute() - elapsed, exec, opt time = {elapsed_time} {exec_time} {opt_exec_time} (sec)')
        if verbose_time: print(f"  ... ocean.execute() elapsed, exec, opt time = {elapsed_time}, {exec_time}, {opt_exec_time} (sec)")
        
        metrics.store_metric(circuit["group"], circuit["circuit"], 'elapsed_time', elapsed_time)
        metrics.store_metric(circuit["group"], circuit["circuit"], 'exec_time', exec_time)
        metrics.store_metric(circuit["group"], circuit["circuit"], 'opt_exec_time', opt_exec_time)
    
        def process_to_bitstring(cut_list):
            # DEVNOTE : Check if the mapping is correct
            # Convert 1 to 0 and -1 to 1
            cut_list = ['0' if i == 1 else '1' for i in cut_list]
            # (-(cut_array - 1)/2).astype('int32')
            return "".join(cut_list)
        
        # for Neal simulator (mimicker), compute counts from each record returned, one per shot
        if device_name == "pegasus":
            all_cuts = [elem[0].tolist() for elem in sampleset.record]
            all_cuts = [process_to_bitstring(cut) for cut in all_cuts]
            unique_cuts = list(set(all_cuts))
            cut_occurances = [all_cuts.count(cut) for cut in unique_cuts]
            result = { cut : count for (cut,count) in zip(unique_cuts, cut_occurances)}
            #print(result)
        
        # for hardware execution, generate counts by repeating records based on the number of times seen
        else:        
            all_cuts = [elem[0].tolist() for elem in sampleset.record]
            all_counts = [int(elem[2]) for elem in sampleset.record]
            all_cuts = [process_to_bitstring(cut) for cut in all_cuts]
            result = { cut : count for (cut,count) in zip(all_cuts, all_counts)}
            #print(result)

        result_handler(circuit["qc"], result, circuit["group"], circuit["circuit"], circuit["shots"])
            
    except Exception as e:
        print(f'ERROR: Failed to execute {circuit["group"]} {circuit["circuit"]}')
        print(f"... exception = {e}")
        if verbose: print(traceback.format_exc())
        return

    # store circuit dimensional metrics

    qc_depth = 20
    qc_size = 20
    qc_xi = 0.5
    qc_n2q = 12
    
    qc_tr_depth = 30
    qc_tr_size = 30
    qc_tr_xi = .5
    qc_tr_n2q = 12
    
    metrics.store_metric(circuit["group"], circuit["circuit"], 'depth', qc_depth)
    metrics.store_metric(circuit["group"], circuit["circuit"], 'size', qc_size)
    metrics.store_metric(circuit["group"], circuit["circuit"], 'xi', qc_xi)
    metrics.store_metric(circuit["group"], circuit["circuit"], 'n2q', qc_n2q)

    metrics.store_metric(circuit["group"], circuit["circuit"], 'tr_depth', qc_tr_depth)
    metrics.store_metric(circuit["group"], circuit["circuit"], 'tr_size', qc_tr_size)
    metrics.store_metric(circuit["group"], circuit["circuit"], 'tr_xi', qc_tr_xi)
    metrics.store_metric(circuit["group"], circuit["circuit"], 'tr_n2q', qc_tr_n2q)

    
    #print(sampleset)
    
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
 

# Wait for all active and batched circuits to complete.
# Execute the user-supplied completion handler to allow user to 
# check if a group of circuits has been completed and report on results.
# Return when there are no more active circuits.
# This is used as a way to complete all groups of circuits and report results.

def finalize_execution(completion_handler=metrics.finalize_group, report_end=True):

    if verbose:
        print("... finalize_execution")
    
    # indicate we are done collecting metrics (called once at end of app)
    if report_end:
        metrics.end_metrics()
       
    
###########################################################################
        
# Test circuit execution
def test_execution():
    pass


