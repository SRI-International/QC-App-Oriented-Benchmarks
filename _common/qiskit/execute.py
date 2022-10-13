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

from qiskit import execute, Aer, transpile
from qiskit import IBMQ
from qiskit.providers.jobstatus import JobStatus

# Noise
from qiskit.providers.aer.noise import NoiseModel, ReadoutError
from qiskit.providers.aer.noise import depolarizing_error, reset_error

logger = logging.getLogger(__name__)

# Use Aer qasm_simulator by default
backend = Aer.get_backend("qasm_simulator")  

# Execution options, passed to transpile method
backend_exec_options = None

# Cached transpiled circuit, used for parameterized execution
cached_circuits = {}

#####################
# DEFAULT NOISE MODEL 

# default noise model, can be overridden using set_noise_model
def default_noise_model():

    noise = NoiseModel()
    # Add depolarizing error to all single qubit gates with error rate 0.3%
    #                    and to all two qubit gates with error rate 3.0%
    depol_one_qb_error = 0.003
    depol_two_qb_error = 0.03
    noise.add_all_qubit_quantum_error(depolarizing_error(depol_one_qb_error, 1), ['rx', 'ry', 'rz'])
    noise.add_all_qubit_quantum_error(depolarizing_error(depol_two_qb_error, 2), ['cx'])

    # Add amplitude damping error to all single qubit gates with error rate 0.0%
    #                         and to all two qubit gates with error rate 0.0%
    amp_damp_one_qb_error = 0.0
    amp_damp_two_qb_error = 0.0
    noise.add_all_qubit_quantum_error(depolarizing_error(amp_damp_one_qb_error, 1), ['rx', 'ry', 'rz'])
    noise.add_all_qubit_quantum_error(depolarizing_error(amp_damp_two_qb_error, 2), ['cx'])

    # Add reset noise to all single qubit resets
    reset_to_zero_error = 0.005
    reset_to_one_error = 0.005
    noise.add_all_qubit_quantum_error(reset_error(reset_to_zero_error, reset_to_one_error),["reset"])

    # Add readout error
    p0given1_error = 0.000
    p1given0_error = 0.000
    error_meas = ReadoutError([[1 - p1given0_error, p1given0_error], [p0given1_error, 1 - p0given1_error]])
    noise.add_all_qubit_readout_error(error_meas)
    
    return noise

noise = default_noise_model()

##########################
# JOB MANAGEMENT VARIABLES 

# Create array of batched circuits and a dict of active circuits
batched_circuits = []
active_circuits = {}

# maximum number of active jobs
max_jobs_active = 5;

# Configure a handler for processing circuits on completion
# user-supplied result handler
result_handler = None

# job mode: False = wait, True = submit multiple jobs
job_mode = False

# Print progress of execution
verbose = False;

# Print additional time metrics for each stage of execution
verbose_time = False;

# Option to perform explicit transpile to collect depth metrics
do_transpile_metrics = True

# Option to perform transpilation prior to execution (disable to execute unmodified circuit)
do_transpile_for_execute = True

# Selection of basis gate set for transpilation
# Note: selector 1 is a hardware agnostic gate set
basis_selector = 1
basis_gates_array = [
    [],
    ['rx', 'ry', 'rz', 'cx'],       # a common basis set, default
    ['cx', 'rz', 'sx', 'x'],        # IBM default basis set
    ['rx', 'ry', 'rxx'],            # IonQ default basis set
    ['h', 'p', 'cx'],               # another common basis set
    ['u', 'cx']                     # general unitaries basis gates
]

######################################################################
# INITIALIZATION METHODS

# Initialize the execution module, with a custom result handler
def init_execution(handler):
    global batched_circuits, result_handler
    batched_circuits.clear()
    active_circuits.clear()
    result_handler = handler
    
    cached_circuits.clear()
    
    # On initialize, always set trnaspilation for metrics and execute to True
    set_tranpilation_flags(do_transpile_metrics=True, do_transpile_for_execute=True)
    

# Set the backend for execution
def set_execution_target(backend_id='qasm_simulator',
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
    
    # handle QASM simulator specially
    elif backend_id == 'qasm_simulator':
        backend = Aer.get_backend("qasm_simulator") 

    elif 'fake' in backend_id:
        backend = getattr(
            importlib.import_module(
                f'qiskit.providers.fake_provider.backends.{backend_id.split("_")[-1]}.{backend_id}'
            ),
            backend_id.title().replace('_', '')
        )
        backend = backend()
        logger.info(f'Set {backend = }')   

    # otherwise use the given backend_id to find the backend
    else:
        if provider_module_name and provider_name:
            # if provider_module and provider_name is provided, assume a custom provider
            provider = getattr(importlib.import_module(provider_module_name), provider_name)
            try:
                # not all custom providers have the .stored_account() method
                provider.load_account()
                backend = provider.get_backend(backend_id)
            except:
                print(authentication_error_msg.format(provider_name))
        else:
            # otherwise, assume IBMQ
            if IBMQ.stored_account():
                # load a stored account
                IBMQ.load_account()
                
                # then create backend from selected provider
                provider = IBMQ.get_provider(hub=hub, group=group, project=project)
                backend = provider.get_backend(backend_id)
            else:
                print(authentication_error_msg.format("IBMQ"))

    # create an informative device name
    device_name = backend_id
    metrics.set_plot_subtitle(f"Device = {device_name}")
    #metrics.set_properties( { "api":"qiskit", "backend_id":backend_id } )
    
    # save execute options with backend
    global backend_exec_options
    backend_exec_options = exec_options


# Set the state of the transpilation flags
def set_tranpilation_flags(do_transpile_metrics = True, do_transpile_for_execute = True):
    globals()['do_transpile_metrics'] = do_transpile_metrics
    globals()['do_transpile_for_execute'] = do_transpile_for_execute
    

def set_noise_model(noise_model = None):
    """
    See reference on NoiseModel here https://qiskit.org/documentation/stubs/qiskit.providers.aer.noise.NoiseModel.html

    Need to pass in a qiskit noise model object, i.e. for amplitude_damping_error:
    ```
        from qiskit.providers.aer.noise import amplitude_damping_error

        noise = NoiseModel()

        one_qb_error = 0.005
        noise.add_all_qubit_quantum_error(amplitude_damping_error(one_qb_error), ['u1', 'u2', 'u3'])

        two_qb_error = 0.05
        noise.add_all_qubit_quantum_error(amplitude_damping_error(two_qb_error).tensor(amplitude_damping_error(two_qb_error)), ['cx'])

        set_noise_model(noise_model=noise)
    ```

    Can also be used to remove noisy simulation by setting `noise_model = None`:
    ```
        set_noise_model()
    ```
    """
    
    global noise
    noise = noise_model

######################################################################
# CIRCUIT EXECUTION METHODS

# Submit circuit for execution
# Execute immediately if possible or put into the list of batched circuits
def submit_circuit(qc, group_id, circuit_id, shots=100, params=None):

    # create circuit object with submission time and circuit info
    circuit = { "qc": qc, "group": str(group_id), "circuit": str(circuit_id),
            "submit_time": time.time(), "shots": shots, "params": params }
            
    if verbose:
        print(f'... submit circuit - group={circuit["group"]} id={circuit["circuit"]} shots={circuit["shots"]} params={circuit["params"]}')

    '''
    if params != None: 
        for param in params.items(): print(f"{param}")
        print([param[1] for param in params.items()])
    '''
    
    # logger doesn't like unicode, so just log the array values for now
    #logger.info(f'Submitting circuit - group={circuit["group"]} id={circuit["circuit"]} shots={circuit["shots"]} params={str(circuit["params"])}')
    logger.info(f'Submitting circuit - group={circuit["group"]} id={circuit["circuit"]} shots={circuit["shots"]} params={[param[1] for param in params.items()] if params else None}')

    # immediately post the circuit for execution if active jobs < max
    if len(active_circuits) < max_jobs_active:
        execute_circuit(circuit)
    
    # or just add it to the batch list for execution after others complete
    else:
        batched_circuits.append(circuit)
        if verbose:
            print("  ... added circuit to batch")


# Launch execution of one job (circuit)
def execute_circuit(circuit):
    logging.info('Entering execute_circuit')

    active_circuit = copy.copy(circuit)
    active_circuit["launch_time"] = time.time()
    active_circuit["pollcount"] = 0 
    
    shots = circuit["shots"]
    
    qc = circuit["qc"]
    
    # do the decompose before obtaining circuit metrics so we expand subcircuits to 2 levels
    # Comment this out here; ideally we'd generalize it here, but it is intended only to 
    # 'flatten out' circuits with subcircuits; we do it in the benchmark code for now so
    # it only affects circuits with subcircuits (e.g. QFT, AE ...)
    # qc = qc.decompose()
    # qc = qc.decompose()
    
    # obtain initial circuit metrics
    qc_depth, qc_size, qc_count_ops, qc_xi, qc_n2q = get_circuit_metrics(qc)

    # default the normalized transpiled metrics to the same, in case exec fails
    qc_tr_depth = qc_depth
    qc_tr_size = qc_size
    qc_tr_count_ops = qc_count_ops
    qc_tr_xi = qc_xi; 
    qc_tr_n2q = qc_n2q
    #print(f"... before tp: {qc_depth} {qc_size} {qc_count_ops}")
    
    try:    
        # transpile the circuit to obtain size metrics using normalized basis
        if do_transpile_metrics:
            qc_tr_depth, qc_tr_size, qc_tr_count_ops, qc_tr_xi, qc_tr_n2q = transpile_for_metrics(qc)
            
        # use noise model from execution options if given for simulator
        this_noise = noise
        
        # make a clone of the backend options so we can remove elements that we use, then pass to .run()
        global backend_exec_options
        backend_exec_options_copy = copy.copy(backend_exec_options)

        '''
        # if 'executor' provided, perform all execution there and return
        if backend_exec_options_copy != None:
            executor = backend_exec_options_copy.pop("executor", None)     
            if executor:
                st = time.time()
                executor(circuit["qc"], shots=shots, backend=backend, result_handler=result_handler, **backend_exec_options_copy)            
                if verbose_time:
                    print(f"  *** executor() time = {time.time() - st}")   
                    
                continue with job
        '''
        
        # get noise model from options; used only in simulator for now
        if backend_exec_options_copy != None and "noise_model" in backend_exec_options_copy:
            this_noise = backend_exec_options_copy["noise_model"]
            #print(f"... using custom noise model: {this_noise}")
        
        # extract execution options if set
        if backend_exec_options_copy == None: backend_exec_options_copy = {}
        
        optimization_level = backend_exec_options_copy.pop("optimization_level", None)
        layout_method = backend_exec_options_copy.pop("layout_method", None)
        routing_method = backend_exec_options_copy.pop("routing_method", None)
        transpile_attempt_count = backend_exec_options_copy.pop("transpile_attempt_count", None)
        transformer = backend_exec_options_copy.pop("transformer", None)
        
        logger.info(f"Executing on backend: {backend.name()}")
            
        #************************************************
        # Initiate execution (with noise if specified and this is a simulator backend)
        if this_noise is not None and backend.name().endswith("qasm_simulator"):
            logger.info(f"Performing noisy simulation, shots = {shots}")
            
            simulation_circuits = circuit["qc"]

            # we already have the noise model, just need to remove it from the options
            # (only for simulator;  for other backends, it is treaded like keyword arg)
            dummy = backend_exec_options_copy.pop("noise_model", None)
                    
            # transpile and bind circuit with parameters; use cache if flagged   
            trans_qc = transpile_and_bind_circuit(circuit["qc"], circuit["params"], backend)
            simulation_circuits = trans_qc
                    
            # apply transformer pass if provided
            if transformer:
                logger.info("applying transformer to noisy simulator")
                simulation_circuits, shots = invoke_transformer(transformer,
                                    trans_qc, backend=backend, shots=shots)
            
            # for noisy simulator, use execute() which works; 
            # no need for transpile above unless there are options like transformer
            logger.info(f'Running circuit on noisy simulator, shots={shots}')
            st = time.time()
            
            ''' some circuits, like Grover's behave incorrectly if we use run()
            job = backend.run(simulation_circuits, shots=shots,
                noise_model=this_noise, basis_gates=this_noise.basis_gates,
                **backend_exec_options_copy)
            '''   
            job = execute(simulation_circuits, backend, shots=shots,
                noise_model=this_noise, basis_gates=this_noise.basis_gates,
                **backend_exec_options_copy)
                
            logger.info(f'Finished Running on noisy simulator - {round(time.time() - st, 5)} (ms)')
            if verbose_time: print(f"  *** qiskit.execute() time = {round(time.time() - st, 5)}")
        
        #************************************************
        # Initiate execution for all other backends and noiseless simulator
        else:            
 
            # if set, transpile many times and pick shortest circuit
            # DEVNOTE: this does not handle parameters yet, or optimizations
            if transpile_attempt_count:
                trans_qc = transpile_multiple_times(circuit["qc"], circuit["params"], backend,
                        transpile_attempt_count, 
                        optimization_level=None, layout_method=None, routing_method=None)
                        
            # transpile and bind circuit with parameters; use cache if flagged                       
            else:
                trans_qc = transpile_and_bind_circuit(circuit["qc"], circuit["params"], backend,
                        optimization_level=optimization_level,
                        layout_method=layout_method,
                        routing_method=routing_method)
            
            # apply transformer pass if provided
            if transformer:
                trans_qc, shots = invoke_transformer(transformer,
                        trans_qc, backend=backend, shots=shots)
            
            #*************************************
            # perform circuit execution on backend
            logger.info(f'Running trans_qc, shots={shots}')
            st = time.time() 
            
            job = backend.run(trans_qc, shots=shots, **backend_exec_options_copy)
            
            logger.info(f'Finished Running trans_qc - {round(time.time() - st, 5)} (ms)')
            if verbose_time: print(f"  *** qiskit.run() time = {round(time.time() - st, 5)}")
            
    except Exception as e:
        print(f'ERROR: Failed to execute circuit {active_circuit["group"]} {active_circuit["circuit"]}')
        print(f"... exception = {e}")
        if verbose: print(traceback.format_exc())
        return
    
    # print("Job status is ", job.status() )
    
    # put job into the active circuits with circuit info
    active_circuits[job] = active_circuit
    # print("... active_circuit = ", str(active_circuit))

    # store circuit dimensional metrics
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'depth', qc_depth)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'size', qc_size)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'xi', qc_xi)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'n2q', qc_n2q)

    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_depth', qc_tr_depth)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_size', qc_tr_size)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_xi', qc_tr_xi)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'tr_n2q', qc_tr_n2q)
    
    # return, so caller can do other things while waiting for jobs to complete

    # deprecated code ...
    '''
    # wait until job is complete
    job_wait_for_completion(job)

    ##############
    # Here we complete the job immediately 
    job_complete(job)
    '''
    if verbose:
        print(f"... executing job {job.job_id()}")

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
    
# Transpile the circuit to obtain normalized size metrics against a common basis gate set
def transpile_for_metrics(qc):

    logger.info('Entering transpile_for_metrics')
    #print("*** Before transpile ...")
    #print(qc)
    st = time.time()
    
    # use either the backend or one of the basis gate sets
    if basis_selector == 0:
        logger.info(f"Start transpile with {basis_selector = }")
        qc = transpile(qc, backend)
        logger.info(f"End transpile with {basis_selector = }")
    else:
        basis_gates = basis_gates_array[basis_selector]
        logger.info(f"Start transpile with basis_selector != 0")
        qc = transpile(qc, basis_gates=basis_gates, seed_transpiler=0)
        logger.info(f"End transpile with basis_selector != 0")
    
    #print(qc)
        
    qc_tr_depth = qc.depth()
    qc_tr_size = qc.size()
    qc_tr_count_ops = qc.count_ops()
    #print(f"*** after transpile: {qc_tr_depth} {qc_tr_size} {qc_tr_count_ops}")
    
    # iterate over the ordereddict to determine xi (ratio of 2 qubit gates to one qubit gates)
    n1q = 0; n2q = 0
    if qc_tr_count_ops != None:
        for key, value in qc_tr_count_ops.items():
            if key == "measure": continue
            if key == "barrier": continue
            if key.startswith("c"): n2q += value
            else: n1q += value
        qc_tr_xi = n2q / (n1q + n2q) 
        qc_tr_n2q = n2q   
    #print(f"... qc_tr_xi = {qc_tr_xi} {n1q} {n2q}")
    
    logger.info(f'transpile_for_metrics - {round(time.time() - st, 5)} (ms)')
    if verbose_time: print(f"  *** transpile_for_metrics() time = {round(time.time() - st, 5)}")
    
    return qc_tr_depth, qc_tr_size, qc_tr_count_ops, qc_tr_xi, qc_tr_n2q

           
# Return a transpiled and bound circuit
# Cache the transpiled, and use it if do_transpile_for_execute not set
# DEVNOTE: this approach does not permit passing of untranspiled circuit through
# DEVNOTE: currently this only caches a single circuit
def transpile_and_bind_circuit(circuit, params, backend,
                optimization_level=None, layout_method=None, routing_method=None):
                
    logger.info(f'transpile_and_bind_circuit()')
    st = time.time()
        
    if do_transpile_for_execute:
        logger.info('transpiling for execute')
        trans_qc = transpile(circuit, backend, 
                optimization_level=optimization_level, layout_method=layout_method, routing_method=routing_method) 
        
        # cache this transpiled circuit
        cached_circuits["last_circuit"] = trans_qc
    
    else:
        logger.info('use cached transpiled circuit for execute')
        if verbose_time: print(f"  ... using cached circuit, no transpile")

        ##trans_qc = circuit["qc"]
        
        # for now, use this cached transpiled circuit (should be separate flag to pass raw circuit)
        trans_qc = cached_circuits["last_circuit"]
    
    #print(trans_qc)
    #print(f"... trans_qc name = {trans_qc.name}")
    
    # obtain name of the transpiled or cached circuit
    trans_qc_name = trans_qc.name
        
    # if parameters provided, bind them to circuit
    if params != None:
        # Note: some loggers cannot handle unicode in param names, so only show the values
        #logger.info(f"Binding parameters to circuit: {str(params)}")
        logger.info(f"Binding parameters to circuit: {[param[1] for param in params.items()]}")
        if verbose_time: print(f"  ... binding parameters")
        
        trans_qc = trans_qc.bind_parameters(params)
        #print(trans_qc)
        
        # store original name in parameterized circuit, so it can be found with get_result()
        trans_qc.name = trans_qc_name
        #print(f"... trans_qc name = {trans_qc.name}")

    logger.info(f'transpile_and_bind_circuit - {trans_qc_name} {round(time.time() - st, 5)} (ms)')
    if verbose_time: print(f"  *** transpile_and_bind() time = {round(time.time() - st, 5)}")
    
    return trans_qc

# Transpile a circuit multiple times for optimal results
# DEVNOTE: this does not handle parameters yet
def transpile_multiple_times(circuit, params, backend, transpile_attempt_count, 
                optimization_level=None, layout_method=None, routing_method=None):
    
    logger.info(f"transpile_multiple_times({transpile_attempt_count})")
    st = time.time()
    
    # array of circuits that have been transpile
    trans_qc_list = [
        transpile(
            circuit, 
            backend, 
            optimization_level=optimization_level,
            layout_method=layout_method,
            routing_method=routing_method
        ) for _ in range(transpile_attempt_count)
    ]
    
    best_op_count = []
    for circ in trans_qc_list:
        # check if there are cx in transpiled circs
        if 'cx' in circ.count_ops().keys(): 
            # get number of operations
            best_op_count.append( circ.count_ops()['cx'] ) 
        # check if there are sx in transpiled circs
        elif 'sx' in circ.count_ops().keys(): 
            # get number of operations
            best_op_count.append( circ.count_ops()['sx'] ) 
            
    # print(f"{best_op_count = }")
    if best_op_count:
        # pick circuit with lowest number of operations
        best_idx = np.where(best_op_count == np.min(best_op_count))[0][0] 
        trans_qc = trans_qc_list[best_idx]
    else: # otherwise just pick the first in the list
        best_idx = 0
        trans_qc = trans_qc_list[0] 
        
    logger.info(f'transpile_multiple_times - {best_idx} {round(time.time() - st, 5)} (ms)')
    if verbose_time: print(f"  *** transpile_multiple_times() time = {round(time.time() - st, 5)}")
    
    return trans_qc


# Invoke a circuit transformer, returning modifed circuit (array) and modifed shots
def invoke_transformer(transformer, circuit, backend=backend, shots=100):

    logger.info('Invoking Transformer')
    st = time.time()
    
    # apply the transformer and get back either a single circuit or a list of circuits
    tr_circuit = transformer(circuit, backend=backend)

    # if transformer results in multiple circuits, divide shot count
    # results will be accumulated in job_complete
    # NOTE: this will need to set a flag to distinguish from multiple circuit execution 
    if isinstance(tr_circuit, list) and len(tr_circuit) > 1:
        shots = int(shots / len(tr_circuit))
    
    logger.info(f'Transformer - {round(time.time() - st, 5)} (ms)')
    if verbose_time:print(f"  *** transformer() time = {round(time.time() - st, 5)} (ms)")
        
    return tr_circuit, shots

    
###########################################################################

# Process a completed job
def job_complete(job):
    active_circuit = active_circuits[job]
    
    if verbose:
        print(f'\n... job complete - group={active_circuit["group"]} id={active_circuit["circuit"]} shots={active_circuit["shots"]}')

    logger.info(f'job complete - group={active_circuit["group"]} id={active_circuit["circuit"]} shots={active_circuit["shots"]}')

    # compute elapsed time for circuit; assume exec is same, unless obtained from result
    elapsed_time = time.time() - active_circuit["launch_time"]
    
    # report exec time as 0 unless valid measure returned
    exec_time = 0.0

    # get job result (DEVNOTE: this might be different for diff targets)
    result = None
        
    if job.status() == JobStatus.DONE:
        result = job.result()
        # print("... result = ", str(result))
        
        # get breakdown of execution time, if method exists 
        # this attribute not available for some providers;
        if "time_per_step" in dir(job) and callable(job.time_per_step):
            time_per_step = job.time_per_step()
            exec_creating_time = (time_per_step["VALIDATING"] - time_per_step["CREATING"]).total_seconds()
            exec_validating_time = (time_per_step["QUEUED"] - time_per_step["VALIDATING"]).total_seconds()
            exec_queued_time = (time_per_step["RUNNING"] - time_per_step["QUEUED"]).total_seconds()
            exec_running_time = (time_per_step["COMPLETED"] - time_per_step["RUNNING"]).total_seconds()
            
            metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_creating_time', exec_creating_time)
            metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_validating_time', exec_validating_time)
            metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_queued_time', exec_queued_time)
            metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_running_time', exec_running_time)
            
        else: 
            time_per_step = {}
            exec_creating_time = 0
            exec_validating_time = 0
            exec_queued_time = 0
            exec_running_time = 0
        
        #print("... time_per_step = ", str(time_per_step))
        if verbose:
            print(f"... exec times, creating = {exec_creating_time}, validating = {exec_validating_time}, queued = {exec_queued_time}, running = {exec_running_time}")        

        # counts = result.get_counts(qc)
        # print("Total counts are:", counts)
        
        # obtain timing info from the results object
        result_obj = result.to_dict()
        results_obj = result.to_dict()['results'][0]
        #print(f"result_obj = {result_obj}")
        #print(f"results_obj = {results_obj}")
        #print(f'shots = {results_obj["shots"]}')
        
        # get the actual shots and convert to int if it is a string
        actual_shots = 0
        for experiment in result_obj["results"]:
            actual_shots += experiment["shots"]
            
        if type(actual_shots) is str:
            actual_shots = int(actual_shots)
        
        if actual_shots != active_circuit["shots"]:
            print(f'WARNING: requested shots not equal to actual shots: {active_circuit["shots"]} != {actual_shots} ')
        
        if "time_taken" in result_obj:
            exec_time = result_obj["time_taken"]
        
        elif "time_taken" in results_obj:
            exec_time = results_obj["time_taken"]
    
    # remove from list of active circuits
    del active_circuits[job]

    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'elapsed_time', elapsed_time)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_time', exec_time)

    # If a result handler has been established, invoke it here with result object
    if result != None and result_handler:
    
        # The following computes the counts by summing them up, allowing for the case where
        # <result> contains results from multiple circuits
        # DEVNOTE: This will need to change; currently the only case where we have multiple result counts
        # is when using randomly_compile; later, there will be other cases
        if type(result.get_counts()) == list:
            total_counts = dict()
            for count in result.get_counts():
                total_counts = dict(Counter(total_counts) + Counter(count))
                
            # make a copy of the result object so we can return a modified version
            orig_result = result
            result = copy.copy(result) 

            # replace the results array with an array containing only the first results object
            # then populate other required fields
            results = copy.copy(result.results[0])
            results.header.name = active_circuit["qc"].name     # needed to identify the original circuit
            results.shots = actual_shots
            results.data.counts = total_counts
            result.results = [ results ]
            
        try:
            result_handler(active_circuit["qc"],
                            result,
                            active_circuit["group"],
                            active_circuit["circuit"],
                            active_circuit["shots"]
                            )
                        
        except Exception as e:
            print(f'ERROR: failed to execute result_handler for circuit {active_circuit["group"]} {active_circuit["circuit"]}')
            print(f"... exception = {e}")
            if verbose:
                print(traceback.format_exc())


# Process a job, whose status cannot be obtained
def job_status_failed(job):
    active_circuit = active_circuits[job]
    
    if verbose:
        print(f'\n... job status failed - group={active_circuit["group"]} id={active_circuit["circuit"]} shots={active_circuit["shots"]}')
    
    # compute elapsed time for circuit; assume exec is same, unless obtained from result
    elapsed_time = time.time() - active_circuit["launch_time"]
    
    # report exec time as 0 unless valid measure returned
    exec_time = 0.0
           
    # remove from list of active circuits
    del active_circuits[job]

    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'elapsed_time', elapsed_time)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_time', exec_time)

        
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
    logger.info('Entering throttle_execution')

    #if verbose:
        #print(f"... throttling execution, active={len(active_circuits)}, batched={len(batched_circuits)}")

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

# Wait for all active and batched circuits to complete.
# Execute the user-supplied completion handler to allow user to 
# check if a group of circuits has been completed and report on results.
# Return when there are no more active circuits.
# This is used as a way to complete all groups of circuits and report results.

def finalize_execution(completion_handler=metrics.finalize_group, report_end=True):

    #if verbose:
        #print("... finalize_execution")

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
    
    # indicate we are done collecting metrics (called once at end of app)
    if report_end:
        metrics.end_metrics()
    
    
# Check if any active jobs are complete - process if so
# Before returning, launch any batched jobs that will keep active circuits < max
# When any job completes, aggregate and report group metrics if all circuits in group are done
# then return, don't sleep

def check_jobs(completion_handler=None):
    
    for job, circuit in active_circuits.items():

        try:
            status = job.status()
            #print("Job status is ", status)
            
        except Exception as e:
            print(f'ERROR: Unable to retrieve job status for circuit {circuit["group"]} {circuit["circuit"]}')
            print(f"... job = {job.job_id()}  exception = {e}")
            
            # finish the job by removing from active list
            job_status_failed(job)
            
            break

        circuit["pollcount"] += 1
        
        # if job not complete, provide comfort ...
        if status == JobStatus.QUEUED:
            if verbose:
                if circuit["pollcount"] < 32 or (circuit["pollcount"] % 15 == 0):
                    print('.', end='')
            continue

        elif status == JobStatus.INITIALIZING:
            if verbose: print('i', end='')
            continue

        elif status == JobStatus.VALIDATING:
            if verbose: print('v', end='')
            continue

        elif status == JobStatus.RUNNING:
            if verbose: print('r', end='')
            continue

        # when complete, canceled, or failed, process the job
        if status == JobStatus.CANCELLED:
            print(f"... circuit execution cancelled.")

        if status == JobStatus.ERROR:
            print(f"... circuit execution failed.")
            if hasattr(job, "error_message"):
                print(f"    job = {job.job_id()}  {job.error_message()}")

        if status == JobStatus.DONE or status == JobStatus.CANCELLED or status == JobStatus.ERROR:
            #if verbose: print("Job status is ", job.status() )
            
            active_circuit = active_circuits[job]
            group = active_circuit["group"]
            
            # process the job and its result data
            job_complete(job)
            
            # call completion handler with the group id
            if completion_handler != None:
                completion_handler(group)
                
            break

    # if not at maximum jobs and there are jobs in batch, then execute another
    if len(active_circuits) < max_jobs_active and len(batched_circuits) > 0:

        # pop the first circuit in the batch and launch execution
        circuit = batched_circuits.pop(0)
        if verbose:
            print(f'... pop and submit circuit - group={circuit["group"]} id={circuit["circuit"]} shots={circuit["shots"]}')
            
        execute_circuit(circuit)  
        

# Test circuit execution
def test_execution():
    pass


########################################
# DEPRECATED METHODS

# these methods are retained for reference and in case needed later

# Wait for active and batched circuits to complete
# This is used as a way to complete a group of circuits and report
# results before continuing to create more circuits.
# Deprecated: maintained for compatibility until all circuits are modified
# to use throttle_execution(0 and finalize_execution()

def execute_circuits():

    # deprecated code ...
    '''
    for batched_circuit in batched_circuits:
        execute_circuit(batched_circuit)
    batched_circuits.clear()
    '''
    
    # wait for all jobs to complete
    wait_for_completion()


# Wait for all active and batched jobs to complete
# deprecated version, with no completion handler

def wait_for_completion():

    if verbose:
        print("... waiting for completion")

    # check and sleep if not complete
    done = False
    pollcount = 0
    while not done:
    
        # check if any jobs complete
        check_jobs()

        # return only when all jobs complete
        if len(active_circuits) < 1:
            break
            
        # delay a bit, increasing the delay periodically 
        sleeptime = 0.25
        if pollcount > 6: sleeptime = 0.5
        if pollcount > 60: sleeptime = 1.0
        time.sleep(sleeptime)
        
        pollcount += 1
    
    if verbose:
        if pollcount > 0: print("")


# Wait for a single job to complete, return when done
# (replaced by wait_for_completion, which handle multiple jobs)

def job_wait_for_completion(job):
        
    done=False
    pollcount = 0
    while not done:
        status = job.status()
        #print("Job status is ", status)
        
        if status == JobStatus.DONE:
            break
            
        if status == JobStatus.CANCELLED:
            break
            
        if status == JobStatus.ERROR:
            break
        
        if status == JobStatus.QUEUED:
            if verbose:
                if pollcount < 44 or (pollcount % 15 == 0):
                    print('.', end='')
        
        elif status == JobStatus.INITIALIZING:
            if verbose: print('i', end='')
            
        elif status == JobStatus.VALIDATING:
            if verbose: print('v', end='')
            
        elif status == JobStatus.RUNNING:
            if verbose: print('r', end='')
            
        pollcount += 1
        
        # delay a bit, increasing the delay periodically 
        sleeptime = 0.25
        if pollcount > 8: sleeptime = 0.5
        if pollcount > 100: sleeptime = 1.0
        time.sleep(sleeptime)

    if pollcount > 0:
        if verbose: print("")

    #if verbose: print("Job status is ", job.status() )
    
    if job.status() == JobStatus.CANCELLED:
        print(f"\n... circuit execution cancelled.")
            
    if job.status() == JobStatus.ERROR:
        print(f"\n... circuit execution failed.")



