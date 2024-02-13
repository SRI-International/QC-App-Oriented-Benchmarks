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

import os
import time
import copy
import importlib
import traceback
from collections import Counter
import logging
import numpy as np

from datetime import datetime, timedelta
from qiskit import execute, Aer, transpile
from qiskit.providers.jobstatus import JobStatus

# QED-C modules
import metrics

# Noise
from qiskit.providers.aer.noise import NoiseModel, ReadoutError
from qiskit.providers.aer.noise import depolarizing_error, reset_error


##########################
# JOB MANAGEMENT VARIABLES 

# these are defined globally currently, but will become class variables later

#### these variables are currently accessed as globals from user code

# maximum number of active jobs
max_jobs_active = 3

# job mode: False = wait, True = submit multiple jobs
job_mode = False

# Print progress of execution
verbose = False

# Print additional time metrics for each stage of execution
verbose_time = False

#### the following variables are accessed only through functions ...

# Specify whether to execute using sessions (and currently using Sampler only)
use_sessions = False

# Session counter for use in creating session name
session_count = 0

# internal session variables: users do not access
session = None
sampler = None

# Use the IBM Quantum Platform system; default is to use the new IBM Cloud
use_ibm_quantum_platform = False

# IBM Quantum Service save here if created
service = None

# Azure Quantum Provider saved here if created
azure_provider = None

# logger for this module
logger = logging.getLogger(__name__)

# Use Aer qasm_simulator by default
backend = Aer.get_backend("qasm_simulator")  

# Execution options, passed to transpile method
backend_exec_options = None

# Create array of batched circuits and a dict of active circuits
batched_circuits = []
active_circuits = {}

# Cached transpiled circuit, used for parameterized execution
cached_circuits = {}

# Configure a handler for processing circuits on completion
# user-supplied result handler
result_handler = None

# Option to compute normalized depth during execution (can disable to reduce overhead in large circuits)
use_normalized_depth = True

# Option to perform explicit transpile to collect depth metrics
# (disabled after first circuit in iterative algorithms)
do_transpile_metrics = True

# Option to perform transpilation prior to execution
# (disabled after first circuit in iterative algorithms)
do_transpile_for_execute = True

# Intercept function to post-process results
result_processor = None
width_processor = None

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


#######################
# SUPPORTING CLASSES

# class BenchmarkResult is made for sessions runs. This is because
# qiskit primitive job result instances don't have a get_counts method 
# like backend results do. As such, a get counts method is calculated
# from the quasi distributions and shots taken.
class BenchmarkResult(object):

    def __init__(self, qiskit_result):
        super().__init__()
        self.qiskit_result = qiskit_result
        self.metadata = qiskit_result.metadata

    def get_counts(self, qc=0):
        counts= self.qiskit_result.quasi_dists[0].binary_probabilities()
        for key in counts.keys():
            counts[key] = int(counts[key] * self.qiskit_result.metadata[0]['shots'])        
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
       
#####################
# DEFAULT NOISE MODEL 

# default noise model, can be overridden using set_noise_model()
def default_noise_model():

    noise = NoiseModel()
    
    # Add depolarizing error to all single qubit gates with error rate 0.05%
    #                    and to all two qubit gates with error rate 0.5%
    depol_one_qb_error = 0.0005
    depol_two_qb_error = 0.005
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
    
    # assign a quantum volume (measured using the values below)
    noise.QV = 2048
    
    return noise

noise = default_noise_model()


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
                hub=None, group=None, project=None, exec_options=None,
                context=None):
    """
    Used to run jobs on a real hardware
    :param backend_id:  device name. List of available devices depends on the provider
    :param hub: hub identifier, currently "ibm-q" for IBM Quantum, "azure-quantum" for Azure Quantum 
    :param group: group identifier, used with IBM-Q accounts.
    :param project: project identifier, used with IBM-Q accounts.
    :param provider_module_name: If using a provider other than IBM-Q, the string of the module that contains the Provider class.  For example, for Quantinuum, this would be 'qiskit.providers.quantinuum'
    :param provider_name:  If using a provider other than IBMQ, the name of the provider class.
    :param context: context for execution, used to create session names
    For example, for Quantinuum, this would be 'Quantinuum'.
    :provider_backend: a custom backend object created and passed in, use backend_id as identifier
    example usage.

    set_execution_target(backend_id='quantinuum_device_1', provider_module_name='qiskit.providers.quantinuum',
                        provider_name='Quantinuum')
    """
    global backend
    global session
    global use_sessions
    global session_count
    authentication_error_msg = "No credentials for {0} backend found. Using the simulator instead."

    # if a custom provider backend is given, use it ...
    # Note: in this case, the backend_id is an identifier that shows up in plots
    if provider_backend != None:
        backend = provider_backend
        
        # The hub variable is used to identify an Azure Quantum backend
        if hub == "azure-quantum":
            from azure.quantum.job.session import Session, SessionJobFailurePolicy 
            
            # increment session counter
            session_count += 1
            
            # create session name
            if context is not None: session_name = context
            else: session_name = f"QED-C Benchmark Session {session_count}"
            
            if verbose:
                print(f"... creating session {session_name} on Azure backend {backend_id}")
                
            # open a session on the backend
            session = backend.open_session(name=session_name,
                    job_failure_policy=SessionJobFailurePolicy.CONTINUE)
                    
            backend.latest_session = session
            
    # handle QASM simulator specially
    elif backend_id == 'qasm_simulator':
        backend = Aer.get_backend("qasm_simulator") 

    # handle Statevector simulator specially
    elif backend_id == 'statevector_simulator':
        backend = Aer.get_backend("statevector_simulator")
        
    # handle 'fake' backends here
    elif 'fake' in backend_id:
        backend = getattr(
            importlib.import_module(
                f'qiskit.providers.fake_provider.backends.{backend_id.split("_")[-1]}.{backend_id}'
            ),
            backend_id.title().replace('_', '')
        )
        backend = backend()
        logger.info(f'Set {backend = }')   

    # otherwise use the given providername or backend_id to find the backend
    else:
        global service
        global sampler
    
        # if provider_module name and provider_name are provided, obtain a custom provider
        if provider_module_name and provider_name:  
            provider = getattr(importlib.import_module(provider_module_name), provider_name)
            try:
                # try, since not all custom providers have the .stored_account() method
                provider.load_account()
                backend = provider.get_backend(backend_id)
            except:
                print(authentication_error_msg.format(provider_name))
        
        # If hub variable indicates an Azure Quantum backend
        elif hub == "azure-quantum":
            global azure_provider
            from azure.quantum.qiskit import AzureQuantumProvider
            from azure.quantum.job.session import Session, SessionJobFailurePolicy
            
            # create an Azure Provider only the first time it is needed
            if azure_provider is None:
                if verbose:
                    print("... creating Azure provider")
                azure_provider = AzureQuantumProvider(
                     resource_id = os.getenv("AZURE_QUANTUM_RESOURCE_ID"),
                     location = os.getenv("AZURE_QUANTUM_LOCATION") 
                )  
                # List the available backends in the workspace
                if verbose:
                    print("    Available backends:")
                    for backend in azure_provider.backends():
                        print(f"      {backend}")

            # increment session counter
            session_count += 1
            
            # create session name
            if context is not None: session_name = context
            else: session_name = f"QED-C Benchmark Session {session_count}"
            
            if verbose:
                print(f"... creating Azure backend {backend_id} and session {session_name}")
                
            # then find backend from the backend_id
            # we should cache this and only change if backend_id changes
            backend = azure_provider.get_backend(backend_id)
 
            # open a session on the backend
            session = backend.open_session(name=session_name,
                    job_failure_policy=SessionJobFailurePolicy.CONTINUE)
                    
            backend.latest_session = session
            
        ###############################
        # if using IBM Quantum Platform, assume the backend_id is given only
        elif use_ibm_quantum_platform: 
        
            from qiskit import IBMQ
            if IBMQ.stored_account():
            
                # load a stored account
                IBMQ.load_account()
                
                # set use_sessions in provided by user - NOTE: this will modify the global setting
                this_use_sessions = exec_options.get("use_sessions", None)
                if this_use_sessions != None:
                    use_sessions = this_use_sessions

                # if use sessions, setup runtime service, Session, and Sampler
                if use_sessions:
                    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
                    
                    service = QiskitRuntimeService()
                    session_count += 1
                    
                    backend = service.backend(backend_id)
                    session = Session(service=service, backend=backend_id)
                    
                    # get Sampler resilience level and transpiler optimization level from exec_options
                    options = Options()
                    options.resilience_level = exec_options.get("resilience_level", 1)
                    options.optimization_level = exec_options.get("optimization_level", 3)
                    
                    # special handling for ibmq_qasm_simulator to set noise model
                    if backend_id == "ibmq_qasm_simulator":
                        this_noise = noise
                        # get noise model from options; used only in simulator for now
                        if "noise_model" in exec_options:
                            this_noise = exec_options.get("noise_model", None)
                            if verbose:
                                print(f"... using custom noise model: {this_noise}")
                        # attach to backend if not None
                        if this_noise != None:
                            options.simulator = {"noise_model": this_noise}
                            metrics.QV = this_noise.QV
                            if verbose:
                                print(f"... setting noise model, QV={this_noise.QV} on {backend_id}")
                        
                    if verbose:
                        print(f"... execute using Sampler on backend_id {backend_id} with options = {options}")
                    
                    # create the Qiskit Sampler with these options
                    sampler = Sampler(session=session, options=options)
                    
                # otherwise, use provider and old style backend
                # for IBM, create backend from IBMQ provider and given backend_id
                else: 
                    if verbose:
                        print(f"... execute using Circuit Runner on backend_id {backend_id}")
                        
                    provider = IBMQ.get_provider(hub=hub, group=group, project=project)
                    backend = provider.get_backend(backend_id)
            else:
                print(authentication_error_msg.format("IBMQ"))

        ###############################
        # otherwise, assume the backend_id is given only and assume it is IBM Cloud device
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
            
            # create the Runtime Service object
            service = QiskitRuntimeService()
            
            # obtain a backend from the service
            backend = service.backend(backend_id)
               
            # DEVNOTE: here we assume if the sessions flag is set, we use Sampler
            # however, we may want to add a use_sampler option so that we can separate these
            
            # set use_sessions if provided by user - NOTE: this will modify the global setting
            this_use_sessions = exec_options.get("use_sessions", None)
            if this_use_sessions != None:
                use_sessions = this_use_sessions

            # if use sessions, setup runtime service, Session, and Sampler
            if use_sessions:
            
                if verbose:
                    print("... using sessions")
                
                session = Session(service=service, backend=backend_id)
                
                # get Sampler resilience level and transpiler optimization level from exec_options
                options = Options()
                options.resilience_level = exec_options.get("resilience_level", 1)
                options.optimization_level = exec_options.get("optimization_level", 3)
                
                # special handling for ibmq_qasm_simulator to set noise model
                if backend_id == "ibmq_qasm_simulator":
                    this_noise = noise
                    
                    # get noise model from options; used only in simulators for now
                    if "noise_model" in exec_options:
                        this_noise = exec_options.get("noise_model", None)
                        if verbose:
                            print(f"... using custom noise model: {this_noise}")
                            
                    # attach to backend if not None
                    if this_noise != None:
                        options.simulator = {"noise_model": this_noise}
                        metrics.QV = this_noise.QV
                        if verbose:
                            print(f"... setting noise model, QV={this_noise.QV} on {backend_id}")
                    
                if verbose:
                    print(f"... execute using Sampler on backend_id {backend_id} with options = {options}")
                
                # create the Qiskit Sampler with these options
                sampler = Sampler(session=session, options=options)
                
            # otherwise, use the circuit runner in the submit_circuit method, without sessions
            else: 
                if verbose:
                    print("... not using sessions")
                    print(f"... execute using Circuit Runner on backend_id {backend_id}")


    # create an informative device name for plots
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


# set flag to control use of sessions
def set_use_sessions(val = False):
    global use_sessions
    use_sessions = val


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
    qc_tr_xi = qc_xi 
    qc_tr_n2q = qc_n2q
    #print(f"... before tp: {qc_depth} {qc_size} {qc_count_ops}")
    
    backend_name = get_backend_name(backend)
    
    try:    
        # transpile the circuit to obtain size metrics using normalized basis
        if do_transpile_metrics and use_normalized_depth:
            qc_tr_depth, qc_tr_size, qc_tr_count_ops, qc_tr_xi, qc_tr_n2q = transpile_for_metrics(qc)
            
            # we want to ignore elapsed time contribution of transpile for metrics (normalized depth)
            active_circuit["launch_time"] = time.time()
            
        # use noise model from execution options if given for simulator
        this_noise = noise
        
        # make a clone of the backend options so we can remove elements that we use, then pass to .run()
        global backend_exec_options
        backend_exec_options_copy = copy.copy(backend_exec_options)
        
        # get noise model from options; used only in simulator for now
        if backend_exec_options_copy != None and "noise_model" in backend_exec_options_copy:
            this_noise = backend_exec_options_copy["noise_model"]
            #print(f"... using custom noise model: {this_noise}")
        
        # extract execution options if set
        if backend_exec_options_copy == None: backend_exec_options_copy = {}
        
        # used in Sampler setup, here remove it for execution
        this_use_sessions = backend_exec_options_copy.pop("use_sessions", None)
        resilience_level = backend_exec_options_copy.pop("resilience_level", None)
        
        # standard Qiskit transpiler options
        optimization_level = backend_exec_options_copy.pop("optimization_level", None)
        layout_method = backend_exec_options_copy.pop("layout_method", None)
        routing_method = backend_exec_options_copy.pop("routing_method", None)
        
        # option to transpile multiple times to find best one
        transpile_attempt_count = backend_exec_options_copy.pop("transpile_attempt_count", None)
        
        # gneeralized transformer method, custom to user
        transformer = backend_exec_options_copy.pop("transformer", None)
        
        global result_processor, width_processor
        postprocessors = backend_exec_options_copy.pop("postprocessor", None)
        if postprocessors:
            result_processor, width_processor = postprocessors
        
        ##############
        # if 'executor' is provided, perform all execution there and return
        # the executor returns a result object that implements get_counts(qc)
        executor = None
        if backend_exec_options_copy != None:
            executor = backend_exec_options_copy.pop("executor", None) 
        
        # NOTE: the executor does not perform any other optional processing
        # Also, the result_handler is called before elapsed_time processing which is not correct
        if executor:
            st = time.time()
            
            # invoke custom executor function with backend options
            qc = circuit["qc"]
            result = executor(qc, backend_name, backend, shots=shots, **backend_exec_options_copy)
            
            if verbose_time:
                print(f"  *** executor() time = {round(time.time() - st,4)}")
            
            # create a pseudo-job to perform metrics processing upon return
            job = Job()
            
            # store the result object on the job for processing in job_complete
            job.executor_result = result  
        
        ##############        
        # normal execution processing is performed here
        else:       
            logger.info(f"Executing on backend: {backend_name}")

            #************************************************
            # Initiate execution (with noise if specified and this is a simulator backend)
            if this_noise is not None and not use_sessions and backend_name.endswith("qasm_simulator"):
                logger.info(f"Performing noisy simulation, shots = {shots}")
                
                # if the noise model has associated QV value, copy it to metrics module for plotting
                if hasattr(this_noise, "QV"):
                    metrics.QV = this_noise.QV
                       
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

                # Indicate number of qubits about to be executed
                if width_processor:
                    width_processor(qc)
                
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
                
                # Indicate number of qubits about to be executed
                if width_processor:
                    width_processor(qc)

                # to execute on Aer state vector simulator, need to remove measurements
                if backend_name.lower() == "statevector_simulator":
                    trans_qc = trans_qc.remove_final_measurements(inplace=False)
                            
                #*************************************
                # perform circuit execution on backend
                logger.info(f'Running trans_qc, shots={shots}')
                st = time.time() 

                if use_sessions:
                    job = sampler.run(trans_qc, shots=shots, **backend_exec_options_copy)
                else:
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

    # also store the job_id for future reference
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'job_id', job.job_id())
    
    if verbose:
        print(f"  ... executing job {job.job_id()}")
    
    # special handling when only runnng one job at a time: wait for result here
    # so the status check called later immediately returns done and avoids polling
    if max_jobs_active <= 1:
        wait_on_job_result(job, active_circuit)

    # return, so caller can do other things while waiting for jobs to complete    

# Utility function to obtain name of backend
# This is needed because some backends support backend.name and others backend.name()
def get_backend_name(backend):
    if callable(backend.name):
        name = backend.name()
    else:
        name = backend.name
    return name

# block and wait for the job result to be returned
# handle network timeouts by doing up to 40 retries once every 15 seconds
def wait_on_job_result(job, active_circuit):
    retry_count = 0
    result = None
    while retry_count < 40:
        try:
            retry_count += 1
            result = job.result()
            break
                         
        except Exception:
            print(f'... error occurred during job.result() for circuit {active_circuit["group"]} {active_circuit["circuit"]} -- retry {retry_count}')
            if verbose: print(traceback.format_exc())
            time.sleep(15)
            continue
    
    if result == None:
        print(f'ERROR: during job.result() for circuit {active_circuit["group"]} {active_circuit["circuit"]}')
        raise ValueError("Failed to execute job")
    else:
        #print(f"... job.result() is done, with result data, continuing")
        pass
    
# Check and return job_status
# handle network timeouts by doing up to 40 retries once every 15 seconds
def get_job_status(job, active_circuit):
    retry_count = 0
    status = None
    while retry_count < 3:
        try:
            retry_count += 1
            #print(f"... calling job.status()")
            status = job.status()
            break
                         
        except Exception:
            print(f'... error occurred during job.status() for circuit {active_circuit["group"]} {active_circuit["circuit"]} -- retry {retry_count}')
            if verbose: print(traceback.format_exc())
            time.sleep(15)
            continue
    
    if status == None:
        print(f'ERROR: during job.status() for circuit {active_circuit["group"]} {active_circuit["circuit"]}')
        raise ValueError("Failed to get job status")
    else:
        #print(f"... job.result() is done, with result data, continuing")
        pass
        
    return status
 
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
        logger.info("Start transpile with basis_selector != 0")
        qc = transpile(qc, basis_gates=basis_gates, seed_transpiler=0)
        logger.info("End transpile with basis_selector != 0")
    
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
# Cache the transpiled circuit, and use it if do_transpile_for_execute not set
# DEVNOTE: this approach does not permit passing of untranspiled circuit through
# DEVNOTE: currently this only caches a single circuit
def transpile_and_bind_circuit(circuit, params, backend,
                optimization_level=None, layout_method=None, routing_method=None):
                
    logger.info('transpile_and_bind_circuit()')
    st = time.time()
        
    if do_transpile_for_execute:
        logger.info('transpiling for execute')
        trans_qc = transpile(circuit, backend, 
                optimization_level=optimization_level, layout_method=layout_method, routing_method=routing_method) 
        
        # cache this transpiled circuit
        cached_circuits["last_circuit"] = trans_qc
    
    else:
        logger.info('use cached transpiled circuit for execute')
        if verbose_time: print("  ... using cached circuit, no transpile")

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
        if verbose_time: print("  ... binding parameters")
        
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
    
    # store these initial time measures now, in case job had error
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'elapsed_time', elapsed_time)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_time', exec_time)

    ###### executor completion
    
    # If job has the 'local_job' attr, execution was done by the executor, and all work is done
    # we process it here, as the subsequent processing has too much time-specific detail
    if hasattr(job, 'local_job'):
    
        # get the result object directly from the pseudo-job object
        result = job.result()
        
        if hasattr(result, 'exec_time'):
            exec_time = result.exec_time
            
        # assume the exec time is the elapsed time, since we don't have more detail
        metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_time', exec_time)
        
        # invoke the result handler with the result object
        if result != None and result_handler:
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
        
        # remove from list of active circuits
        del active_circuits[job]
    
        return      
    
    ###### normal completion
    
    # get job result (DEVNOTE: this might be different for diff targets)
    result = None
        
    if job.status() == JobStatus.DONE:
        result = job.result()
        # print("... result = ", str(result))

        # for Azure Quantum, need to obtain execution time from sessions object
        # Since there may be multiple jobs, need to find the one that matches the current job_id
        if azure_provider is not None and session is not None:
            details = job._azure_job.details
            
            # print("... session_job.details = ", details)
            exec_time = (details.end_execution_time - details.begin_execution_time).total_seconds()
            
            # DEVNOTE: startup time is not currently used or stored
            # it seesm to include queue time, so it has no added value over the elapsed time we currently store
            startup_time = (details.begin_execution_time - details.creation_time).total_seconds()
            
        # counts = result.get_counts(qc)
        # print("Total counts are:", counts)
        
        # if we are using sessions, structure of result object is different;
        # use a BenchmarkResult object to hold session result and provide a get_counts()
        # that returns counts to the benchmarks in the same form as without sessions
        if use_sessions:
            result = BenchmarkResult(result)
            #counts = result.get_counts()
            
            actual_shots = result.metadata[0]['shots']
            result_obj = result.metadata[0]
            results_obj = result.metadata[0]
        else:
            result_obj = result.to_dict()
            results_obj = result.to_dict()['results'][0]
            
            # get the actual shots and convert to int if it is a string
            # DEVNOTE: this summation currently applies only to randomized compiling 
            # and may cause problems with other use cases (needs review)
            actual_shots = 0
            for experiment in result_obj["results"]:
                actual_shots += experiment["shots"]

        #print(f"result_obj = {result_obj}")
        #print(f"results_obj = {results_obj}")
        #print(f'shots = {results_obj["shots"]}')

        # convert actual_shots to int if it is a string
        if type(actual_shots) is str:
            actual_shots = int(actual_shots)
        
        # check for mismatch of requested shots and actual shots
        if actual_shots != active_circuit["shots"]:
            print(f'WARNING: requested shots not equal to actual shots: {active_circuit["shots"]} != {actual_shots} ')
            
            # allow processing to continue, but use the requested shot count
            actual_shots = active_circuit["shots"]
            
        # obtain timing info from the results object
        # the data has been seen to come from both places below
        if "time_taken" in result_obj:
            exec_time = result_obj["time_taken"]
        
        elif "time_taken" in results_obj:
            exec_time = results_obj["time_taken"]
        
        # override the initial value with exec_time returned from successful execution
        metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_time', exec_time)
        
        # process additional detailed step times, if they exist (this might override exec_time too)
        process_step_times(job, result, active_circuit)
    
    # remove from list of active circuits
    del active_circuits[job]

    # If a result handler has been established, invoke it here with result object
    if result != None and result_handler:
    
        # invoke a result processor if specified in exec_options
        if result_processor:
            logger.info('result_processor(...)')
            result = result_processor(result)
    
        # The following computes the counts by summing them up, allowing for the case where
        # <result> contains results from multiple circuits
        # DEVNOTE: This will need to change; currently the only case where we have multiple result counts
        # is when using randomly_compile; later, there will be other cases
        if not use_sessions and type(result.get_counts()) == list:
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

# Process detailed step times, if they exist
def process_step_times(job, result, active_circuit):
    #print("... processing step times")
    
    exec_creating_time = 0
    exec_validating_time = 0
    exec_queued_time = 0
    exec_running_time = 0
        
    # get breakdown of execution time, if method exists 
    # this attribute not available for some providers and only for circuit-runner model;
    if not use_sessions and "time_per_step" in dir(job) and callable(job.time_per_step):
        time_per_step = job.time_per_step()
        if verbose:
            print(f"... job.time_per_step() = {time_per_step}")
        
        creating_time = time_per_step.get("CREATING")
        validating_time = time_per_step.get("VALIDATING")
        queued_time = time_per_step.get("QUEUED")
        running_time = time_per_step.get("RUNNING")
        completed_time = time_per_step.get("COMPLETED")
        
        # for testing, since hard to reproduce on some systems
        #running_time = None
        
        # make these all slightly non-zero so averaging code is triggered (> 0.001 required)
        exec_creating_time = 0.001
        exec_validating_time = 0.001
        exec_queued_time = 0.001
        exec_running_time = 0.001
        
        # this is note used
        exec_quantum_classical_time = 0.001

        # compute the detailed time metrics
        if validating_time and creating_time:
            exec_creating_time = (validating_time - creating_time).total_seconds()
        if queued_time and validating_time:
            exec_validating_time = (queued_time - validating_time).total_seconds()
        if running_time and queued_time:
            exec_queued_time = (running_time - queued_time).total_seconds()
        if completed_time and running_time:
            exec_running_time = (completed_time - running_time).total_seconds()

    # when sessions and sampler used, we obtain metrics differently
    if use_sessions:
        job_timestamps = job.metrics()['timestamps']
        job_metrics = job.metrics()
        # print(f"... usage = {job_metrics['usage']} {job_metrics['executions']}")
        
        if verbose:
            print(f"... job.metrics() = {job.metrics()}")
            print(f"... job.result().metadata[0] = {result.metadata[0]}")

        # occasionally, these metrics come back as None, so try to use them
        try:
            created_time = datetime.strptime(job_timestamps['created'][11:-1],"%H:%M:%S.%f")
            created_time_delta = timedelta(hours=created_time.hour, minutes=created_time.minute, seconds=created_time.second, microseconds = created_time.microsecond)
            finished_time = datetime.strptime(job_timestamps['finished'][11:-1],"%H:%M:%S.%f")
            finished_time_delta = timedelta(hours=finished_time.hour, minutes=finished_time.minute, seconds=finished_time.second, microseconds = finished_time.microsecond)
            running_time = datetime.strptime(job_timestamps['running'][11:-1],"%H:%M:%S.%f")
            running_time_delta = timedelta(hours=running_time.hour, minutes=running_time.minute, seconds=running_time.second, microseconds = running_time.microsecond)
            
            # compute the total seconds for creating and running the circuit
            exec_creating_time = (running_time_delta - created_time_delta).total_seconds()
            exec_running_time = (finished_time_delta - running_time_delta).total_seconds()
            
            # these do not seem to be avaiable
            exec_validating_time = 0.001
            exec_queued_time = 0.001
            
            # when using sessions, the 'running_time' is the 'quantum exec time' - override it here.
            metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_time', exec_running_time)
            
            # use the data in usage field if it is returned, usually by hardware
            # here we use the executions field to indicate we are on hardware
            if "usage" in job_metrics and "executions" in job_metrics:
                if job_metrics['executions'] > 0:
                    exec_time = job_metrics['usage']['quantum_seconds'] 
                    
                    # and use this one as it seems to be valid for this case
                    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_time', exec_time)
            
            # DEVNOTE: we do not compute this yet
            # exec_quantum_classical_time = job.metrics()['bss']
            # metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_quantum_classical_time', exec_quantum_classical_time)     
        
        except Exception as e:
            if verbose:
                print(f'WARNING: incomplete time metrics for circuit {active_circuit["group"]} {active_circuit["circuit"]}')
                print(f"... job = {job.job_id()}  exception = {e}")

    # In metrics, we use > 0.001 to indicate valid data; need to floor these values to 0.001
    exec_creating_time = max(0.001, exec_creating_time)
    exec_validating_time = max(0.001, exec_validating_time)
    exec_queued_time = max(0.001, exec_queued_time)
    exec_running_time = max(0.001, exec_running_time)

    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_creating_time', exec_creating_time)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_validating_time', 0.001)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_queued_time', 0.001)
    metrics.store_metric(active_circuit["group"], active_circuit["circuit"], 'exec_running_time', exec_running_time)

    #print("... time_per_step = ", str(time_per_step))
    if verbose:
        print(f"... computed exec times: queued = {exec_queued_time}, creating/transpiling = {exec_creating_time}, validating = {exec_validating_time}, running = {exec_running_time}") 

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
        
    # also, close any active session at end of the app
    global session
    if report_end and session != None:
        if verbose:
            print(f"... closing active session: {session_count}\n")
        
        session.close()
        session = None
        

# Check if any active jobs are complete - process if so
# Before returning, launch any batched jobs that will keep active circuits < max
# When any job completes, aggregate and report group metrics if all circuits in group are done
# then return, don't sleep

def check_jobs(completion_handler=None):
    
    for job, circuit in active_circuits.items():

        try:
            #status = job.status()
            status = get_job_status(job, circuit)   # use this version, robust to network failure 
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
            print("... circuit execution cancelled.")

        if status == JobStatus.ERROR:
            print("... circuit execution failed.")
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
        print("\n... circuit execution cancelled.")
            
    if job.status() == JobStatus.ERROR:
        print("\n... circuit execution failed.")



