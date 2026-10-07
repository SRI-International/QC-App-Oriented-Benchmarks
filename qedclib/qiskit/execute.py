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
import threading
from collections import Counter
import logging
import numpy as np

from datetime import datetime, timedelta
from qiskit import QuantumCircuit, transpile
from qiskit.providers.jobstatus import JobStatus
from qiskit.primitives import StatevectorSampler
from qiskit_aer import Aer

# Noise Model imports
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_aer.noise import depolarizing_error, reset_error

# QED-C modules
from qedclib import metrics
from qedclib import qcb_mpi as mpi

##########################
# JOB MANAGEMENT VARIABLES 

# these are defined globally currently, but will become class variables later

#### these variables are currently accessed as globals from user code

# Print progress of execution
verbose = False

# Cancel flag — set by request_cancel() to interrupt execution between batches/circuits
cancel_requested = False

# Parallel execution flag — when True, execute_circuits() will attempt to run
# circuits in parallel by mapping them onto disjoint qubit regions of the QPU.
# Set via: execute.parallel_execution = True (or via CLI --parallel_mode)
parallel_execution = False

def request_cancel():
    """Request cancellation of the current execution."""
    global cancel_requested
    cancel_requested = True

# Print additional time metrics for each stage of execution
verbose_time = False

# Timing decomposition — set by execute_circuits/execute_circuits_parallel after each call.
# Callers (e.g. hamlib) can read these to report meaningful timing breakdowns.
last_transpile_time = 0.0    # seconds spent in transpilation
last_exec_time = 0.0         # QPU execution time (from execution_spans, or sampler wall-clock)
last_elapsed_time = 0.0      # total wall-clock of the execute call

#### the following variables are accessed only through functions ...

# Specify whether to execute using sessions (and currently using Sampler only)
use_sessions = False

# Session counter for use in creating session name
session_count = 0

# internal session variables: users do not access
session = None
sampler = None

# M3 mitigation
use_m3 = False
m3_mitigation = {}
m3_cache = {}

# Use the IBM Quantum Platform system; default is to use the IBM Cloud
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

# Cached transpiled circuit, used for parameterized execution
cached_circuits = {}

# Configure a handler for processing circuits on completion
# user-supplied result handler
result_handler = None

# Option to compute normalized depth during execution (can disable to reduce overhead in large circuits)
use_normalized_depth = True

# Auto-warmup: execute a tiny circuit on first call to execute_circuits() to prime the backend
auto_warmup = True
_warmup_done = False

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

# class ExecutionResult is a normalized result wrapper for quantum circuit execution.
# It accepts either a Qiskit PrimitiveResult (from sampler), a raw counts dict,
# or a list of counts dicts. get_counts() always returns:
#   - dict for a single circuit
#   - list[dict] for multiple circuits
# This normalization allows benchmark code to process results uniformly
# without knowing which execution path was used.
class ExecutionResult:

    def __init__(self, source):
        super().__init__()
        self._counts = None
        self.metadata = None
        self.native_result = source  # preserve original result for vendor-specific access

        if isinstance(source, dict):
            self._counts = source
        elif isinstance(source, list):
            self._counts = self._normalize(source)
        elif hasattr(source, 'get_counts'):
            # Native Qiskit Result object (from backend.run().result())
            counts = source.get_counts()
            if isinstance(counts, list):
                self._counts = self._normalize(counts)
            else:
                self._counts = counts
        else:
            # Qiskit PrimitiveResult from sampler (sessions or immediate)
            self._extract_from_qiskit(source)

    def _extract_from_qiskit(self, result):
        """Extract counts from a Qiskit PrimitiveResult object."""
        self.metadata = result.metadata
        count_array = []
        for pub_result in result:
            # join_data() merges all classical registers into one
            bitvals = pub_result.join_data()
            count_array.append(bitvals.get_counts())
        self._counts = self._normalize(count_array)

    def _normalize(self, counts):
        """Normalize counts: single-element list unwraps to dict."""
        if isinstance(counts, list):
            if len(counts) == 0:
                return {}
            elif len(counts) == 1:
                return counts[0]
            else:
                return counts
        return counts

    def set_counts(self, counts):
        self._counts = counts

    def get_counts(self, qc=None):
        # Support integer index into the counts list (matches native Qiskit Result API)
        if isinstance(qc, int) and isinstance(self._counts, list):
            return self._counts[qc]
        return self._counts

# Backward compatibility aliases
BenchmarkResult = ExecutionResult
BenchmarkResult2 = ExecutionResult
ExecResult = ExecutionResult

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
    global result_handler
    result_handler = handler
    
    cached_circuits.clear()
    
    # On initialize, always set transpilation for metrics and execute to True
    set_transpilation_flags(do_transpile_metrics=True, do_transpile_for_execute=True)
    

# Set the backend for execution
def set_execution_target(backend_id='qasm_simulator',
                provider_module_name=None, provider_name=None, provider_backend=None,
                hub=None, group=None, project=None, exec_options=None,
                context=None, gpus_per_circuit=None):
    # DEVNOTE: gpus_per_circuit is accepted but ignored for Qiskit.
    # It is a CUDA-Q concept (statevector sharing across GPUs).
    # Remove this parameter when gpus_per_circuit moves into exec_options.
    """
    Set the backend execution target.
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
    global sampler
    global session
    global use_ibm_quantum_platform
    global use_sessions
    global session_count
    global use_m3
    authentication_error_msg = "ERROR: Failed to connect to backend '{0}'. Check credentials and backend name."

    # default to qasm_simulator if None passed in
    if backend_id == None:
        backend_id="qasm_simulator"

    if exec_options is None:
        exec_options = {}

    # Reset session state from any previous backend (prevents stale IBM sampler/session
    # from being used when switching to a different backend like IonQ or qasm_simulator)
    sampler = None
    session = None

    # set M3 options
    use_m3 = exec_options.get("use_m3", False)

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
    
    elif backend_id == "statevector_sampler":
        from qiskit.primitives import StatevectorSampler
        sampler = StatevectorSampler()  # does not support mid-circuit measurement

    elif backend_id == "aer_sampler":
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        sampler = AerSampler()  # support mid-circuit measurement
        backend = AerSimulator()

    # handle 'fake' backends here
    elif 'fake' in backend_id:
        backend = getattr(
            importlib.import_module(
                f'qiskit_ibm_runtime.fake_provider.backends.{backend_id.split("_")[-1]}.{backend_id}'
            ),
            backend_id.title().replace('_', '')
        )
        backend = backend()
        logger.info(f'Set {backend = }')   

    # otherwise use the given providername or backend_id to find the backend
    else:
        global service
    
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
        # otherwise, assume the backend_id is given only and assume it is IBM Cloud device
        else:
            # need to import `Session` here to avoid the collision with
            # `azure.quantum.job.session.Session`

            from qiskit_ibm_runtime import (
                QiskitRuntimeService,
                SamplerOptions,
                SamplerV2,
                Batch,
                Session,
            )

            # set use_ibm_quantum_platform if provided by user - NOTE: this will modify the global setting
            use_ibm_quantum_platform = exec_options.get("use_ibm_quantum_platform", use_ibm_quantum_platform)

            if use_ibm_quantum_platform:
                channel = "ibm_quantum"
                instance = f"{hub}/{group}/{project}"
            else:
                channel = "ibm_cloud"
                instance = f"{hub or ''}{group or ''}{project or ''}"
            print(f"... using Qiskit Runtime {channel=} {instance=}")

            backend_name = backend_id
            primitive_name = "sampler"

            # Use token from environment if available, otherwise fall back to saved account
            ibm_token = os.environ.get("IBM_API_TOKEN", None)
            if instance and not ibm_token:
                print("  WARNING: IBM_INSTANCE is set but IBM_API_TOKEN is not — using saved token.")
                print("  If using a different account, set both IBM_API_TOKEN and IBM_INSTANCE.")

            try:
                service = QiskitRuntimeService(channel=channel, token=ibm_token, instance=instance)
                backend = service.backend(backend_name)

                # DEVNOTE : If dynamic circuit is enabled in the exec_options, then If Else Operation
                # is imported so that the Hardware device supports the feature.
                ######@@@@@@@@@@@@###########
                if exec_options.get("dynamic_circuit"):
                    from qiskit.circuit import IfElseOp
                    backend.target.add_instruction(IfElseOp, name="if_else")
                ######@@@@@@@@@@@@###########

            except Exception as ex:
                if "403" in str(ex) or "Forbidden" in str(ex):
                    print(f"ERROR: IBM credentials rejected (403 Forbidden) for {backend_id}.")
                    print("  Check that IBM_API_TOKEN and IBM_INSTANCE match the same account.")
                    print("  Also verify saved credentials: QiskitRuntimeService.saved_accounts(channel='ibm_cloud')")
                    raise RuntimeError(f"IBM authentication failed for {backend_id} — 403 Forbidden") from None
                print(authentication_error_msg.format(backend_id))
                raise ex
            print(f"... using {backend=} {primitive_name=}")

            # DEVNOTE: here we assume if the sessions flag is set, we use Sampler
            # however, we may want to add a use_sampler option so that we can separate these
            
            # set use_sessions if provided by user - NOTE: this will modify the global setting
            this_use_sessions = exec_options.get("use_sessions", None)
            if this_use_sessions != None:
                use_sessions = this_use_sessions

            # if use sessions, setup runtime service, Session, and Sampler
            if use_sessions:
                if verbose:
                    print("... using session")
                if session is None:
                    session = Session(backend=backend)
            # otherwise, use Sampler in Batch mode
            else:
                if verbose:
                    print("... using batch")
                if session is None:
                    session = Batch(backend=backend)

            # set Sampler options
            options_dict = exec_options.get("sampler_options", None)
            options = SamplerOptions(**options_dict if options_dict else {})
            print(f"... execute using Sampler on {backend_name=} with {options=}")

            sampler = SamplerV2(session if session else backend, options=options)  
            
            # DEVNOTE : If dynamic circuit is enabled in the exec_options, then it has to enable the
            # sampler option with the execution_path to gen3-experimental. This feature was updated in
            # the IBM Qiskit hardware update around the month July 2025 with the depreciation of the 
            # dynamic circits feature on IBM Sherbrooke. 
            ######@@@@@@@@@@@@###########
            if exec_options.get("dynamic_circuit"):
                sampler = SamplerV2(backend)
                sampler.options.experimental = {"execution_path" : "gen3-experimental"}
            ######@@@@@@@@@@@@###########

    # create an informative device name for plots
    device_name = backend_id
    metrics.set_plot_subtitle(f"Device = {device_name}")
    #metrics.set_properties( { "api":"qiskit", "backend_id":backend_id } )
    
    # save execute options with backend
    global backend_exec_options
    backend_exec_options = exec_options

    # Warmup here so the transpiler startup cost (~3-4s on first call) is not
    # included in execution timing. If set_execution_target() is not called,
    # warmup happens lazily on first execute_circuits() call instead.
    _do_warmup()


def _do_warmup():
    """Run a small transpile+execute to prime Qiskit's transpiler and Aer backend.
    Only runs once — subsequent calls are no-ops via the _warmup_done flag."""
    global _warmup_done
    if not auto_warmup or _warmup_done:
        return
    _warmup_done = True
    try:
        from qiskit import QuantumCircuit as QC
        wc = QC(2, 2); wc.h(0); wc.cx(0, 1); wc.measure([0, 1], [0, 1])
        wc = transpile(wc, backend)
        backend.run([wc, wc], shots=100).result()
        if verbose:
            print("... warmup circuit executed")
    except Exception:
        pass  # warmup is best-effort


# Set the state of the transpilation flags
def set_transpilation_flags(do_transpile_metrics = True, do_transpile_for_execute = True):
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

def close_session():
    # close any active session at end of the app
    global session
    if session is not None:
        if verbose:
            print(f"... closing active session: {session_count}\n")
        
        session.close()
        session = None
        
        
######################################################################
# CIRCUIT METADATA METHODS
  
# compute circuit properties (depth, etc) and store to active circuit object
def compute_and_store_circuit_info(
        qc: QuantumCircuit,
        group_id: str,
        circuit_id: str,
        do_transpile_metrics: bool = True,
        use_normalized_depth: bool = True
    ):

    if qc is None:
        return

    metrics_values = compute_circuit_metrics(qc, do_transpile_metrics, use_normalized_depth)
    store_circuit_metrics(group_id, circuit_id, metrics_values)


def compute_all_circuit_metrics(circuits, do_transpile_metrics=True, use_normalized_depth=True):
    """Compute and store circuit metrics for all circuits in a nested dict.

    Args:
        circuits: nested dict {group: {circuit_id: qc}} from get_circuits()
        do_transpile_metrics: if True, compute transpiled depth metrics (expensive)
        use_normalized_depth: if True, use normalized basis for depth (requires transpile)
    """
    for group_id in circuits:
        if not isinstance(circuits[group_id], dict):
            continue
        for circuit_id in circuits[group_id]:
            compute_and_store_circuit_info(circuits[group_id][circuit_id],
                                          str(group_id), str(circuit_id),
                                          do_transpile_metrics, use_normalized_depth)


# Compute circuit metrics (algorithmic + optionally normalized) and return as a tuple.
# Does not store to metrics table — caller decides when to store.
def compute_circuit_metrics(qc, do_transpile_metrics=True, use_normalized_depth=True):

    if qc is None:
        return (0, 0, 0, 0, 0, 0, 0, 0)

    # obtain initial (algorithmic) circuit metrics
    qc_depth, qc_size, qc_count_ops, qc_xi, qc_n2q = get_circuit_metrics(qc)

    # default the normalized transpiled metrics to the same, in case transpile is skipped or fails
    qc_tr_depth = qc_depth
    qc_tr_size = qc_size
    qc_tr_xi = qc_xi
    qc_tr_n2q = qc_n2q

    try:
        # transpile the circuit to obtain size metrics using normalized basis
        if do_transpile_metrics and use_normalized_depth:
            qc_tr_depth, qc_tr_size, qc_tr_count_ops, qc_tr_xi, qc_tr_n2q = transpile_for_metrics(qc)

    except Exception as e:
        print(f'ERROR: Failed to transpile circuit for metrics')
        print(f"... exception = {e}")
        if verbose: print(traceback.format_exc())

    return (qc_depth, qc_size, qc_xi, qc_n2q,
            qc_tr_depth, qc_tr_size, qc_tr_xi, qc_tr_n2q)


# Store precomputed circuit metrics to the metrics table.
# metrics_values is the 8-tuple returned by compute_circuit_metrics().
def store_circuit_metrics(group_id, circuit_id, metrics_values):

    (qc_depth, qc_size, qc_xi, qc_n2q,
     qc_tr_depth, qc_tr_size, qc_tr_xi, qc_tr_n2q) = metrics_values

    metrics.store_metric(group_id, circuit_id, 'depth', qc_depth)
    metrics.store_metric(group_id, circuit_id, 'size', qc_size)
    metrics.store_metric(group_id, circuit_id, 'xi', qc_xi)
    metrics.store_metric(group_id, circuit_id, 'n2q', qc_n2q)

    metrics.store_metric(group_id, circuit_id, 'tr_depth', qc_tr_depth)
    metrics.store_metric(group_id, circuit_id, 'tr_size', qc_tr_size)
    metrics.store_metric(group_id, circuit_id, 'tr_xi', qc_tr_xi)
    metrics.store_metric(group_id, circuit_id, 'tr_n2q', qc_tr_n2q)


# Utility function to obtain name of backend
# This is needed because some backends support backend.name and others backend.name()
def get_backend_name(backend):
    if callable(backend.name):
        name = backend.name()
    else:
        name = backend.name
    return name

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
        qc_xi = round(n2q / (n1q + n2q), 3)
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
        qc = transpile(qc, backend, seed_transpiler=0)
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
        qc_tr_xi = round(n2q / (n1q + n2q), 3)
        qc_tr_n2q = n2q
    #print(f"... qc_tr_xi = {qc_tr_xi} {n1q} {n2q}")
    
    logger.info(f'transpile_for_metrics - {round(time.time() - st, 5)} (ms)')
    if verbose_time: print(f"  *** transpile_for_metrics() time = {round(time.time() - st, 5)}")
    
    return qc_tr_depth, qc_tr_size, qc_tr_count_ops, qc_tr_xi, qc_tr_n2q

           
# Return a transpiled and bound circuit
# Cache the transpiled circuit, and use it if do_transpile_for_execute not set
# DEVNOTE: this approach does not permit passing of untranspiled circuit through
# DEVNOTE: currently this only caches a single circuit
def transpile_and_bind_circuit(circuit, params, backend, basis_gates=None,
                optimization_level=None, layout_method=None, routing_method=None,
                seed_transpiler=0):
    logger.info('transpile_and_bind_circuit()')
    st = time.time()
        
    if do_transpile_for_execute:
        logger.info('transpiling for execute')
        

        # DEVNOTE CAUTION: There are two ways you can transpile circuits. If exec_options has the dynamical_decoupling
        # enabled, then the passmanager is imported to transpile the circuit. This mechanism allows the dynamical decoupling effect 
        # to be manually added with the choice of your own dynamical decoupling sequence and duration. If dynamical decoupling is not
        # enabled then just the regular transpile function is used.
        if not backend_exec_options.get("dynamical_decoupling"):
            trans_qc = transpile(circuit, backend, basis_gates=basis_gates,
                    optimization_level=optimization_level, layout_method=layout_method, routing_method=routing_method,
                    seed_transpiler=seed_transpiler)
        else:
        ######@@@@@@@@@@@@###########
            from qiskit.transpiler import generate_preset_pass_manager
            from qiskit.transpiler.passmanager import PassManager
            from qiskit.circuit import IfElseOp
            from qiskit.circuit.library import XGate
            from qiskit_ibm_runtime.transpiler.passes.scheduling import PadDynamicalDecoupling
            from qiskit_ibm_runtime.transpiler.passes.scheduling import DynamicCircuitInstructionDurations
            from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis
            from qiskit_ibm_runtime.transpiler.passes.scheduling import PadDelay

            dd_sequence = [XGate(), XGate()]
            durations = DynamicCircuitInstructionDurations.from_backend(backend)
            pm = generate_preset_pass_manager(optimization_level=1, target=backend.target) 
            pm.scheduling = PassManager([ALAPScheduleAnalysis(durations),PadDynamicalDecoupling(durations, dd_sequence),])
            trans_qc = pm.run(circuit)
        ######@@@@@@@@@@@@###########
        
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
        
        trans_qc = trans_qc.assign_parameters(params)
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
            routing_method=routing_method,
            seed_transpiler=seed,
        ) for seed in range(transpile_attempt_count)
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


# Test circuit execution
def test_execution():
    pass

###########################################################################
# NEW ARRAY-BASED EXECUTION PATH
#
# These functions implement a cleaner execution model that operates on
# arrays of circuits. They are designed to coexist with the existing
# execute_circuit/submit_circuit/job_complete path above, which remains
# untouched for backward compatibility.
#
# API Layering:
#   Level 1 (primitives): execute_circuits, process_circuit_results,
#       compute_circuit_metrics, store_circuit_metrics, ExecutionResult
#   Level 2 (convenience): submit_circuits
#   Level 3 (benchmark): calls submit_circuits or Level 1 directly
###########################################################################

def submit_circuits(circuits, num_shots=100, max_batch_size=None, batch_by_group=False, params=None):
    """
    Execute a dict of circuit arrays and store execution metrics.

    Assumes circuit depth metrics have already been computed (via compute_and_store_circuit_info)
    if desired. This function handles execution and timing only.

    Automatically calls metrics.init_metrics() if not already initialized.

    Args:
        circuits: nested dict {group: {circuit_id: qc}} from get_circuits()
        num_shots: shots per circuit
        max_batch_size: max circuits per batch (None = no limit)
        batch_by_group: if True, batch boundaries align with group changes.
                        If False (default), batch by max_batch_size only.
    """

    # Auto-initialize metrics if not yet done
    if metrics.start_time == 0:
        metrics.init_metrics()

    # Flatten circuits dict to ordered list for execution
    circuits_info = []
    for group_id in circuits:
        if not isinstance(circuits[group_id], dict):
            continue
        for circuit_id in circuits[group_id]:
            circuits_info.append({
                "qc": circuits[group_id][circuit_id],
                "group": str(group_id),
                "circuit": str(circuit_id),
                "shots": num_shots
            })

    if len(circuits_info) == 0:
        print("WARNING: No circuits to execute")
        return

    if verbose:
        print(f"... submit_circuits({len(circuits_info)} circuits, max_batch={max_batch_size}, by_group={batch_by_group})")

    # Synchronize MPI ranks before execution begins
    mpi.barrier()

    if batch_by_group:
        # Group circuits by their "group" key, execute one group at a time
        from itertools import groupby
        for group_key, group_iter in groupby(circuits_info, key=lambda ci: ci["group"]):
            batch = list(group_iter)
            _execute_batch(batch, num_shots, max_batch_size, params=params)
    else:
        # Batch by max_batch_size regardless of group boundaries
        _execute_batch(circuits_info, num_shots, max_batch_size, params=params)


def _normalize_params(params):
    """Convert params to list-of-dicts format.
    Accepts: list of dicts, or tuple of (names_list, values_2d_array).
    Returns: list of dicts mapping param names/objects to values.
    """
    if params is None:
        return None
    if isinstance(params, tuple) and len(params) == 2:
        names, values_list = params
        return [dict(zip(names, vals)) for vals in values_list]
    return params


def _bind_params_to_circuits(circuit, params):
    """Expand a single parameterized circuit into bound circuits using params.

    Each entry in params is a dict mapping parameter identifiers to values.
    Keys can be string names or native Parameter/ParameterVector objects.
    String names are looked up from circuit.parameters by name.

    Returns: list of fully-bound circuits.
    """
    param_dicts = _normalize_params(params)
    if param_dicts is None:
        return None

    # Build lookup for string-name keys (skip if keys are already Parameter objects)
    first_key = next(iter(param_dicts[0]))
    if isinstance(first_key, str):
        param_lookup = {p.name: p for p in circuit.parameters}
    else:
        param_lookup = None

    bound_circuits = []
    for pd in param_dicts:
        if param_lookup is not None:
            qiskit_dict = {param_lookup[name]: val for name, val in pd.items()}
        else:
            qiskit_dict = pd
        bound_circuits.append(circuit.assign_parameters(qiskit_dict))

    return bound_circuits


def execute_circuits(circuits, num_shots=100, wait=True, gpus_per_circuit=None, params=None):
    """
    Execute an array of circuits. Pure execution -- no metrics,
    no result_handler, no dict knowledge.

    Always takes an array. Always returns (job_id, result).
    Uses the backend/sampler/noise configured by set_execution_target().
    Processes exec_options for noise_model, executor, and transpilation settings.

    When parallel_execution is True, routes to execute_circuits_parallel()
    in execute_parallel.py, which maps circuits onto disjoint qubit regions.

    Args:
        circuits: list of QuantumCircuit objects
        num_shots: shots per circuit
        wait: if True (default), block until results are ready.
              if False, return immediately with result=None.
        gpus_per_circuit: accepted for API compatibility with cudaq (ignored here)
        params: parameter bindings for repeated execution of a single circuit.
            None (default): circuits are already fully bound.
            List of dicts: [{name: value, ...}, ...] one dict per execution.
            Tuple of (names, values): (["name0", "name1"], [[v0, v1], ...])
            Keys can be string names or native Parameter/ParameterVector objects.

    Returns:
        (job_id, result) tuple:
        - job_id: identifier for the job (serializable)
        - result: ExecutionResult with get_counts() -> list of dicts,
          or None if wait=False
    """

    # Expand parameterized circuit into bound circuits
    if params is not None:
        circuits = _bind_params_to_circuits(circuits[0], params)

    # Route to parallel execution if enabled (implementation in execute_parallel.py)
    if parallel_execution and circuits and len(circuits) > 1:
        from execute_parallel import execute_circuits_parallel
        return execute_circuits_parallel(circuits, num_shots)

    if verbose:
        print(f"... execute_circuits({len(circuits)}, {num_shots}, wait={wait})")

    # Lazy warmup fallback — in case set_execution_target() was not called
    _do_warmup()

    # Extract exec_options from the global set by set_execution_target()
    opts = copy.copy(backend_exec_options) if backend_exec_options else {}

    # Extract noise model — exec_options can override the module-level noise
    # Values: None = no noise, "default" = built-in default model, or a NoiseModel object
    this_noise = opts.pop("noise_model", noise)
    if this_noise == "default":
        this_noise = default_noise_model()

    # Extract executor callback
    executor = opts.pop("executor", None)

    # Extract transpilation options
    optimization_level = opts.pop("optimization_level", None)
    layout_method = opts.pop("layout_method", None)
    routing_method = opts.pop("routing_method", None)

    # Pop options that are handled elsewhere (don't pass to backend.run)
    opts.pop("use_sessions", None)
    opts.pop("resilience_level", None)
    opts.pop("sampler_options", None)
    opts.pop("use_ibm_quantum_platform", None)
    opts.pop("dynamic_circuit", None)
    opts.pop("transpile_attempt_count", None)
    opts.pop("transformer", None)
    opts.pop("postprocessor", None)
    opts.pop("use_m3", None)

    backend_name = get_backend_name(backend) if backend else "unknown"

    ##########
    # Executor path — custom execution callback, per-circuit
    if executor:
        pseudo_job = Job()
        counts_array = []
        per_circuit_times = []
        for qc in circuits:
            ts_circ = time.time()
            result = executor(qc, backend_name, backend, shots=num_shots, **opts)
            per_circuit_times.append(time.time() - ts_circ)
            if hasattr(result, 'get_counts'):
                counts_array.append(result.get_counts())
            else:
                counts_array.append(result)
        exec_result = ExecutionResult(counts_array)
        exec_result._per_circuit_times = per_circuit_times
        return (pseudo_job.job_id(), exec_result)

    ##########
    # Standard execution paths — always pass array

    global last_transpile_time, last_exec_time, last_elapsed_time
    last_transpile_time = 0.0
    last_exec_time = 0.0
    last_elapsed_time = 0.0
    ts_execute = time.time()

    try:
        # Execute on appropriate path
        if sampler is not None:
            # Sampler path (IBM Runtime, StatevectorSampler, AerSampler)
            from qiskit.primitives import StatevectorSampler as _SvSampler
            if isinstance(sampler, _SvSampler):
                # StatevectorSampler is pure math — no device constraints, skip transpilation
                trans_qcs = circuits if isinstance(circuits, list) else [circuits]
                last_transpile_time = 0.0
            else:
                ts_transpile = time.time()
                trans_qcs = transpile(circuits, backend,
                    optimization_level=optimization_level,
                    layout_method=layout_method,
                    routing_method=routing_method)
                last_transpile_time = time.time() - ts_transpile

            # set job tags if SamplerV2 on IBM Quantum Platform (max 8 tags allowed)
            if hasattr(sampler, "options") and hasattr(sampler.options, "environment"):
                job_tags = [qc.name for qc in circuits if hasattr(qc, 'name')][:8]
                sampler.options.environment.job_tags = job_tags

            job = sampler.run(trans_qcs, shots=num_shots)

        elif this_noise is not None and not sampler and backend_name.endswith("qasm_simulator"):
            # Noisy simulator path — transpile to noise model's basis gates only;
            # don't pass backend alongside basis_gates (Qiskit 2.x warning)
            ts_transpile = time.time()
            trans_qcs = transpile(circuits,
                basis_gates=this_noise.basis_gates)
            last_transpile_time = time.time() - ts_transpile

            # Copy noise model QV to metrics if available
            if hasattr(this_noise, "QV"):
                metrics.QV = this_noise.QV

            # Remove noise_model from opts if present (passed explicitly to backend.run)
            opts.pop("noise_model", None)

            job = backend.run(trans_qcs, shots=num_shots,
                noise_model=this_noise, basis_gates=this_noise.basis_gates,
                **opts)

        else:
            # All other backends and noiseless simulator
            ts_transpile = time.time()
            trans_qcs = transpile(circuits, backend,
                optimization_level=optimization_level,
                layout_method=layout_method,
                routing_method=routing_method)
            last_transpile_time = time.time() - ts_transpile

            # Statevector simulator: remove final measurements
            if backend_name.lower() == "statevector_simulator":
                trans_qcs = [qc.remove_final_measurements(inplace=False) for qc in trans_qcs]

            job = backend.run(trans_qcs, shots=num_shots, **opts)

    except KeyboardInterrupt:
        raise  # always let Ctrl-C through
    except Exception as e:
        print(f'ERROR: Failed to execute circuits')
        print(f"... exception = {e}")
        print(traceback.format_exc())  # always print traceback for execution errors
        # Return empty result with pseudo job_id
        pseudo_job = Job()
        return (pseudo_job.job_id(), ExecutionResult([{} for _ in circuits]))

    # Extract job_id before waiting
    try:
        job_id = job.job_id()
    except:
        job_id = Job.unique_job_id  # fallback if job doesn't support job_id()

    # If not waiting, return job_id with no result
    if not wait:
        return (job_id, None)

    # Wait for result using threaded approach: job.result() runs in a background
    # thread with retry logic, while main thread prints comfort dots and checks
    # job status for early error/cancel detection. Result is returned instantly
    # via threading.Event — no polling delay.
    is_local_simulator = 'simulator' in backend_name.lower()
    raw_result = wait_for_result_threaded(job, job_id, circuits, is_local_simulator)

    # Wrap result: ExecutionResult for sampler path, native Result for backend path
    if raw_result is not None:
        if sampler is not None:
            result = ExecutionResult(raw_result)
        else:
            result = raw_result
    else:
        result = ExecutionResult([{} for _ in circuits])

    # Extract per-circuit timing from result and attach (cudaq pattern)
    try:
        per_circuit_times = _extract_per_circuit_times(raw_result, len(circuits))
        if per_circuit_times:
            result._per_circuit_times = per_circuit_times
            last_exec_time = sum(per_circuit_times)
    except Exception:
        pass  # timing extraction is best-effort

    # If no execution_spans timing, fall back to 0 (unknown)
    if last_exec_time == 0.0 and raw_result is not None:
        # No execution_spans available (e.g. simulator) — leave as 0
        pass

    # Attach job object for hardware step timing in process_circuit_results
    try:
        result._job = job
    except Exception:
        pass

    last_elapsed_time = time.time() - ts_execute

    if verbose:
        print(f"... execute_circuits complete, job_id={job_id}")

    return (job_id, result)


def execute_circuit_groups(circuit_groups, num_shots_list=None, num_shots=None):
    """
    Execute groups of circuits, each group with its own shot count.

    Each group is a list of QuantumCircuits. Groups may have different numbers
    of circuits and different shot counts. When parallel_execution is True,
    routes to execute_circuit_groups_parallel() which will compose circuits
    from multiple groups onto disjoint qubit regions for parallel execution.

    Args:
        circuit_groups: list of lists of QuantumCircuit objects
        num_shots_list: list of ints, one per group (shot count for each group).
                        If None, uses num_shots for all groups.
        num_shots: default shot count if num_shots_list is not provided.
                   Ignored if num_shots_list is given. Defaults to 100.

    Returns:
        (job_id, group_results) tuple:
        - job_id: identifier for the job
        - group_results: list of ExecutionResult, one per group
    """
    # Handle shot count args
    if num_shots_list is None:
        if num_shots is None:
            num_shots = 100
        num_shots_list = [num_shots] * len(circuit_groups)

    if len(num_shots_list) != len(circuit_groups):
        raise ValueError(f"num_shots_list length ({len(num_shots_list)}) must match "
                         f"circuit_groups length ({len(circuit_groups)})")

    if verbose:
        group_sizes = [len(g) for g in circuit_groups]
        print(f"... execute_circuit_groups: {len(circuit_groups)} groups, "
              f"sizes={group_sizes}, shots={num_shots_list}")

    # Determine max circuit width across all groups
    max_width = 0
    for group in circuit_groups:
        for qc in group:
            w = qc.num_qubits if hasattr(qc, 'num_qubits') else 0
            if w > max_width:
                max_width = w

    # Check available qubits on the backend
    available_qubits = None
    try:
        if backend is not None:
            if hasattr(backend, 'num_qubits'):
                available_qubits = backend.num_qubits
            elif hasattr(backend, 'configuration'):
                available_qubits = backend.configuration().n_qubits
    except Exception:
        pass

    if available_qubits is not None and max_width > 0:
        if available_qubits < max_width:
            print(f"ERROR: circuit width ({max_width} qubits) exceeds device capacity "
                  f"({available_qubits} qubits)")
            pseudo_job = Job()
            return (pseudo_job.job_id(), [ExecutionResult([{}] * len(g)) for g in circuit_groups])
        elif available_qubits < 2 * max_width:
            if verbose:
                print(f"... WARNING: device has {available_qubits} qubits but parallel "
                      f"requires >= {2 * max_width} (2x max circuit width {max_width}), "
                      f"executing sequentially")

    # Route to parallel execution if enabled
    if parallel_execution and len(circuit_groups) > 1:
        from execute_parallel import execute_circuit_groups_parallel
        return execute_circuit_groups_parallel(circuit_groups, num_shots_list)

    # Sequential: execute each group independently
    group_results = []
    last_job_id = None
    for circuits, shots in zip(circuit_groups, num_shots_list):
        job_id, result = execute_circuits(circuits, num_shots=shots)
        last_job_id = job_id
        group_results.append(result)

    if verbose:
        print(f"... execute_circuit_groups complete, {len(group_results)} groups")

    return (last_job_id, group_results)


def _execute_batch(circuits_info, num_shots, max_batch_size, params=None):
    """Internal: execute circuits_info in chunks of max_batch_size."""
    global cancel_requested
    cancel_requested = False
    batch_size = max_batch_size or len(circuits_info)
    for i in range(0, len(circuits_info), batch_size):
        if cancel_requested:
            print("\n... execution cancelled by user")
            break
        batch = circuits_info[i:i + batch_size]
        circuits = [ci["qc"] for ci in batch]
        ts = time.time()
        job_id, results = execute_circuits(circuits, num_shots, params=params)
        elapsed_time = time.time() - ts
        if results is not None:
            process_circuit_results(batch, results, job_id=job_id, elapsed_time=elapsed_time)
        else:
            print(f'WARNING: No results for batch of {len(batch)} circuits (job {job_id}) -- skipping')


def process_circuit_results(circuits_info, results, job_id=None, elapsed_time=None, num_shots=None):
    """
    Map batch results back to individual circuits. For each circuit:
    wraps counts in ExecutionResult, calls result_handler, stores timing
    and job_id.

    Replaces the need for CountsWrapper / BenchmarkResult2 classes
    used in modularized notebooks.

    Args:
        circuits_info: either:
            - list of dicts with keys "qc", "group", "circuit", "shots"
            - nested dict {group: {circuit_id: qc}} from get_circuits()
        results: result object with get_counts() returning list or dict.
                 Can be from execute_circuits() or user's own execution.
        job_id: job identifier (stored per circuit in metrics)
        elapsed_time: wall-clock seconds for the batch (stored per circuit)
        num_shots: shots per circuit (used when circuits_info is a nested dict)
    """

    # If circuits_info is a nested dict {group: {circuit_id: qc}}, flatten it
    if isinstance(circuits_info, dict):
        flat_info = []
        for group_id, group_circuits in circuits_info.items():
            if isinstance(group_circuits, dict):
                for circuit_id, qc in group_circuits.items():
                    flat_info.append({
                        "qc": qc, "group": str(group_id),
                        "circuit": str(circuit_id), "shots": num_shots or 0
                    })
        circuits_info = flat_info

    # Guard against None or missing results (e.g. job timeout or cancellation)
    if results is None:
        print("WARNING: No results to process (execution may have failed)")
        return

    logger.info(f'process_circuit_results({len(circuits_info)}, job_id={job_id})')

    if verbose:
        print(f"... process_circuit_results({len(circuits_info)}, job_id={job_id})")

    # Extract per-circuit counts from batch result
    counts_list = results.get_counts()
    if isinstance(counts_list, dict):
        counts_list = [counts_list]  # single-element array was unwrapped by ExecutionResult

    # Validate result count matches circuit count
    if len(counts_list) != len(circuits_info):
        print(f'WARNING: result count mismatch — expected {len(circuits_info)}, got {len(counts_list)}')
        # Pad with empty dicts so all circuits get processed (with empty results)
        while len(counts_list) < len(circuits_info):
            counts_list.append({})

    # Compute per-circuit timing from the results object and elapsed wall-clock time
    num_in_batch = len(counts_list)
    per_circuit_times, per_circuit_elapsed, batch_exec_time = \
        _compute_circuit_timing(results, elapsed_time, num_in_batch)

    # Process each circuit's result
    for idx, (ci, counts) in enumerate(zip(circuits_info, counts_list)):

        # Store timing metrics: per-circuit if available, otherwise batch time
        if per_circuit_times and idx < len(per_circuit_times):
            metrics.store_metric(ci["group"], ci["circuit"], 'exec_time', per_circuit_times[idx])
            if per_circuit_elapsed:
                metrics.store_metric(ci["group"], ci["circuit"], 'elapsed_time', per_circuit_elapsed[idx])
            elif elapsed_time is not None:
                metrics.store_metric(ci["group"], ci["circuit"], 'elapsed_time', elapsed_time / num_in_batch)
        else:
            # No per-circuit timing — divide batch elapsed evenly
            if elapsed_time is not None:
                metrics.store_metric(ci["group"], ci["circuit"], 'elapsed_time', elapsed_time / num_in_batch)
            # exec_time: use backend-reported time if available, otherwise elapsed as fallback
            metrics.store_metric(ci["group"], ci["circuit"], 'exec_time',
                                 batch_exec_time / num_in_batch)

        # Store job_id for tracking/retrieval
        if job_id is not None:
            metrics.store_metric(ci["group"], ci["circuit"], 'job_id', job_id)

        # Validate shot count matches request
        actual_shots = sum(counts.values()) if counts else 0
        if actual_shots > 0 and ci["shots"] > 0 and actual_shots != ci["shots"]:
            if verbose:
                print(f'WARNING: circuit {ci["group"]}/{ci["circuit"]}: requested {ci["shots"]} shots, got {actual_shots}')

        # Wrap individual counts in ExecutionResult for result_handler
        circuit_result = ExecutionResult(counts)

        # Call the benchmark's result handler (computes fidelity etc.)
        # Skip if counts are empty (cancelled/failed job) to avoid division-by-zero in handlers
        if result_handler and counts:
            if verbose:
                sample = dict(list(counts.items())[:3])
                print(f"... [debug pcr] circuit {ci['group']}/{ci['circuit']}: "
                      f"counts_type={type(counts).__name__}, "
                      f"sample={sample}")
            try:
                result_handler(ci["qc"], circuit_result,
                              ci["group"], ci["circuit"], ci["shots"])
            except Exception as e:
                print(f'ERROR: failed in result_handler for circuit {ci["group"]} {ci["circuit"]}')
                print(f"... exception = {e}")
                if verbose: print(traceback.format_exc())
        elif result_handler and not counts:
            print(f'WARNING: empty results for circuit {ci["group"]}/{ci["circuit"]} — skipping result_handler')

    # Process hardware step times if job object is available
    job = getattr(results, '_job', None)
    if job is not None and not hasattr(job, 'local_job'):
        _process_step_times_batch(job, circuits_info)

def _extract_per_circuit_times(raw_result, num_circuits):
    """
    Extract per-circuit execution times from a Qiskit result object.

    Returns a list of floats (one per circuit), or None if per-circuit
    timing is not available. Ported from the old job_complete() timing logic.

    Timing sources (in priority order):
    1. Per-experiment time_taken from result.to_dict()["results"][i] (simulators)
    2. execution_spans from PrimitiveResult metadata (IBM hardware via sampler)
    3. Total time_taken / num_circuits (evenly divided fallback)
    4. None (caller falls back to elapsed_time)
    """
    if num_circuits == 0:
        return None

    # Path A: Native Qiskit Result (non-sampler) — has to_dict()
    if hasattr(raw_result, 'to_dict'):
        try:
            result_dict = raw_result.to_dict()
            #print(f"... in to_dict branch: {raw_result}", flush=True)

            if verbose_time:
                print(f"... _extract_per_circuit_times: to_dict() path, {num_circuits} circuits")
                top_keys = list(result_dict.keys())
                print(f"... result_dict keys: {top_keys}")
                has_tt = "time_taken" in result_dict
                print(f"... result_dict has time_taken: {has_tt} {'= ' + str(result_dict['time_taken']) if has_tt else ''}")
                if "results" in result_dict and len(result_dict["results"]) > 0:
                    exp0_keys = list(result_dict["results"][0].keys())
                    print(f"... results[0] keys: {exp0_keys}")
                    has_exp_tt = "time_taken" in result_dict["results"][0]
                    print(f"... results[0] has time_taken: {has_exp_tt} {'= ' + str(result_dict['results'][0]['time_taken']) if has_exp_tt else ''}")

            # Try per-experiment time_taken first (best data for simulators)
            if "results" in result_dict:
                experiments = result_dict["results"]
                per_times = []
                for exp in experiments:
                    if "time_taken" in exp and exp["time_taken"] > 0:
                        per_times.append(exp["time_taken"])
                    else:
                        break
                if len(per_times) == num_circuits:
                    if verbose_time:
                        print(f"... per-experiment times: {per_times[:5]}{'...' if len(per_times) > 5 else ''}")
                    return per_times

            # Fall back to total time_taken divided evenly
            if "time_taken" in result_dict and result_dict["time_taken"] > 0:
                avg = result_dict["time_taken"] / num_circuits
                if verbose_time:
                    print(f"... using total time_taken / {num_circuits} = {avg}")
                return [avg] * num_circuits
        except Exception as e:
            if verbose_time:
                print(f"... to_dict() extraction failed: {e}")

        if verbose_time:
            print(f"... no timing extracted from to_dict() path")
        return None

    # Path B: PrimitiveResult (sampler) — iterable over PubResults
    # The old code submitted one circuit at a time, so top-level metadata was per-circuit.
    # Now we submit batches, so we need per-pub metadata for individual circuit timing.
    if hasattr(raw_result, 'metadata'):
    
        #print(f"... in metadata branch: {raw_result}", flush=True)
        #execution_spans = raw_result.metadata['execution']['execution_spans']
        #print(f"... exec info: {execution_spans}", flush=True)

        if verbose_time:
            print(f"... _extract_per_circuit_times: PrimitiveResult path, {num_circuits} circuits")
            print(f"... top-level metadata keys: {list(raw_result.metadata.keys()) if isinstance(raw_result.metadata, dict) else type(raw_result.metadata)}")
            try:
                first_pub = raw_result[0]
                print(f"... pub[0].metadata type: {type(first_pub.metadata)}")
                if isinstance(first_pub.metadata, dict):
                    print(f"... pub[0].metadata keys: {list(first_pub.metadata.keys())}")
                    if 'execution' in first_pub.metadata:
                        print(f"... pub[0].metadata['execution']: {first_pub.metadata['execution']}")
                else:
                    print(f"... pub[0].metadata = {first_pub.metadata}")
            except Exception as e:
                print(f"... could not inspect pub[0].metadata: {e}")

        # Try per-pub metadata first (each PubResult may have its own timing)
        # PrimitiveResult is iterable: result[i] is PubResult with its own .metadata
        try:
            per_times = []
            for pub_result in raw_result:
                pub_meta = pub_result.metadata
                if isinstance(pub_meta, dict):
                    if 'execution' in pub_meta:
                        try:
                            # DEVNOTE: pre-2.5 code commented out (deprecate later)
                            #spans = pub_meta['execution']['execution_spans']['__value__']['spans']
                            #per_times.append(spans[0].duration)
                            spans = pub_meta['execution']['execution_spans']
                            per_times.append(spans.duration)
                            continue
                        except (KeyError, TypeError, AttributeError, IndexError):
                            pass
                    if 'time_taken' in pub_meta:
                        per_times.append(pub_meta['time_taken'])
                        continue
                # This pub has no timing — abort per-pub extraction
                if verbose_time:
                    print(f"... pub has no timing info, aborting per-pub extraction")
                per_times = []
                break
            if len(per_times) == num_circuits:
                if verbose_time:
                    print(f"... per-pub times extracted: {per_times}")
                return per_times
        except (TypeError, IndexError) as e:
            if verbose_time:
                print(f"... per-pub iteration failed: {e}")

        # Fall back to top-level metadata (batch-level timing, divided evenly)
        if isinstance(raw_result.metadata, dict):
            metadata = raw_result.metadata

            # Try execution_spans (IBM hardware)
            try:
                if 'execution' in metadata:       
                    spans = metadata['execution']['execution_spans']
                    
                    duration = 0.0
                    if hasattr(spans, 'duration'):
                        duration = spans.duration
                        
                    # DEVNOTE: handle pre-2.5 version of Qiskit (deprecate later)
                    else:
                        spans = spans['__value__']['spans']
                        duration = spans[0].duration

                    avg = duration / num_circuits
                    if verbose_time:
                        print(f"... duration: {duration}", flush=True)
                        print(f"... single span, dividing evenly: {avg}")
                        
                    return [avg] * num_circuits
          
            except (KeyError, TypeError, AttributeError, IndexError):
                pass

            # Try top-level time_taken divided evenly (last resort before None)
            if "time_taken" in metadata:
                try:
                    avg = metadata["time_taken"] / num_circuits
                    if verbose_time:
                        print(f"... using top-level time_taken / {num_circuits} = {avg}")
                    return [avg] * num_circuits
                except (TypeError, ZeroDivisionError):
                    pass

        if verbose_time:
            print(f"... no timing extracted from PrimitiveResult")

    return None


def _compute_circuit_timing(results, elapsed_time, num_in_batch):
    """
    Compute per-circuit execution and elapsed times from a batch result.

    Uses per-circuit times from _extract_per_circuit_times (attached to results
    as _per_circuit_times) if available. Falls back to batch-level time_taken
    from the result object, or elapsed_time as last resort.

    Distributes elapsed_time overhead proportionally by exec_time so that
    larger circuits absorb more overhead.

    Args:
        results: ExecutionResult or native result object
        elapsed_time: wall-clock seconds for the batch (or None)
        num_in_batch: number of circuits in the batch

    Returns:
        (per_circuit_times, per_circuit_elapsed, batch_exec_time):
        - per_circuit_times: list of exec times per circuit, or None
        - per_circuit_elapsed: list of elapsed times per circuit, or None
        - batch_exec_time: total batch exec time (fallback when no per-circuit)
    """

    # Extract per-circuit timing (attached by execute_circuits)
    per_circuit_times = getattr(results, '_per_circuit_times', None)

    # Executor/local_job path: if the result has exec_time attached (from a custom executor),
    # use it as the batch exec time when no per-circuit times are available
    if per_circuit_times is None and hasattr(results, 'exec_time'):
        batch_exec_time = results.exec_time
        return per_circuit_times, None, batch_exec_time

    # Extract batch-level exec_time as fallback (when per-circuit times not available)
    batch_exec_time = 0.0
    if per_circuit_times is None:
        try:
            native = getattr(results, 'native_result', results)

            if hasattr(native, 'to_dict'):
                # Native Qiskit Result — timing in result dict
                result_dict = native.to_dict()
                if "time_taken" in result_dict:
                    batch_exec_time = result_dict["time_taken"]
                elif "results" in result_dict and len(result_dict["results"]) > 0:
                    if "time_taken" in result_dict["results"][0]:
                        batch_exec_time = result_dict["results"][0]["time_taken"]
            elif hasattr(native, 'metadata') and isinstance(native.metadata, dict):
                if "time_taken" in native.metadata:
                    batch_exec_time = native.metadata["time_taken"]
        except Exception as e:
            if verbose:
                print(f"... could not extract exec_time: {e}")

    # Track whether batch_exec_time came from the backend or is just a fallback
    exec_time_from_backend = batch_exec_time > 0.0

    # Fall back to elapsed_time for exec_time only if backend provided no timing
    if not exec_time_from_backend and elapsed_time is not None:
        batch_exec_time = elapsed_time

    # Compute per-circuit elapsed times by distributing overhead proportionally.
    # Overhead = elapsed_time - total_exec_time (transpilation, submission, queuing, etc.)
    # Each circuit gets overhead proportional to its share of total exec time,
    # so larger circuits absorb more overhead (they likely had more prep time too).
    per_circuit_elapsed = None
    if per_circuit_times and elapsed_time is not None:
        total_exec = sum(per_circuit_times)
        if total_exec > 0:
            overhead = elapsed_time - total_exec
            if overhead > 0:
                per_circuit_elapsed = [
                    et + overhead * (et / total_exec) for et in per_circuit_times
                ]
            else:
                # No overhead (or negative due to timing granularity) — elapsed = exec
                per_circuit_elapsed = list(per_circuit_times)
        else:
            # All exec times are zero — divide elapsed evenly
            avg_elapsed = elapsed_time / num_in_batch
            per_circuit_elapsed = [avg_elapsed] * num_in_batch

    return per_circuit_times, per_circuit_elapsed, batch_exec_time


def _process_step_times_batch(job, circuits_info):
    """
    Extract detailed step timing from IBM hardware jobs and store per circuit.
    Ported from the old process_step_times() function, adapted for batch processing.

    This may override exec_time with quantum_seconds for real hardware.
    Only called when job is an IBM backend job (supports time_per_step or metrics).
    """
    if job is None:
        return

    exec_creating_time = 0
    exec_validating_time = 0
    exec_queued_time = 0
    exec_running_time = 0

    # get breakdown of execution time, if method exists
    # this attribute not available for some providers
    if "time_per_step" in dir(job) and callable(job.time_per_step):
        try:
            time_per_step = job.time_per_step()
            if verbose:
                print(f"... job.time_per_step() = {time_per_step}")

            creating_time = time_per_step.get("CREATING")
            validating_time = time_per_step.get("VALIDATING")
            queued_time = time_per_step.get("QUEUED")
            running_time = time_per_step.get("RUNNING")
            completed_time = time_per_step.get("COMPLETED")

            # make these all slightly non-zero so averaging code is triggered (> 0.001 required)
            exec_creating_time = 0.001
            exec_validating_time = 0.001
            exec_queued_time = 0.001
            exec_running_time = 0.001

            # compute the detailed time metrics
            if validating_time and creating_time:
                exec_creating_time = (validating_time - creating_time).total_seconds()
            if queued_time and validating_time:
                exec_validating_time = (queued_time - validating_time).total_seconds()
            if running_time and queued_time:
                exec_queued_time = (running_time - queued_time).total_seconds()
            if completed_time and running_time:
                exec_running_time = (completed_time - running_time).total_seconds()
        except Exception:
            pass

    # Sampler timing from job.metrics() — timestamps and usage data.
    #
    # NOTE: We extract these values for verbose reporting but do NOT overwrite exec_time.
    # The execution_spans duration from _extract_per_circuit_times (called earlier in
    # execute_circuits) provides more precise quantum execution time than what we get here.
    # Testing on IBM backends (May 2026) showed:
    #   - execution_spans: 2.20s (precise, fractional seconds)
    #   - quantum_seconds: 3s (same value rounded up to the nearest second)
    #   - timestamps running→finished: 4.9s (includes server-side overhead)
    # The execution_spans value best represents actual quantum execution time.
    # For larger circuits or error-mitigated workloads, these values may diverge further
    # and this decision may need to be revisited.
    if sampler is not None:
        try:
            job_metrics = job.metrics()
            job_timestamps = job_metrics['timestamps']

            created_time = datetime.strptime(job_timestamps['created'][11:-1],"%H:%M:%S.%f")
            created_time_delta = timedelta(hours=created_time.hour, minutes=created_time.minute, seconds=created_time.second, microseconds=created_time.microsecond)
            finished_time = datetime.strptime(job_timestamps['finished'][11:-1],"%H:%M:%S.%f")
            finished_time_delta = timedelta(hours=finished_time.hour, minutes=finished_time.minute, seconds=finished_time.second, microseconds=finished_time.microsecond)
            running_time = datetime.strptime(job_timestamps['running'][11:-1],"%H:%M:%S.%f")
            running_time_delta = timedelta(hours=running_time.hour, minutes=running_time.minute, seconds=running_time.second, microseconds=running_time.microsecond)

            # compute the total seconds for creating and running the circuit
            exec_creating_time = (running_time_delta - created_time_delta).total_seconds()
            exec_running_time = (finished_time_delta - running_time_delta).total_seconds()

            # these do not seem to be available
            exec_validating_time = 0.001
            exec_queued_time = 0.001

            # Extract quantum_seconds from usage data if available
            sampler_quantum_seconds = None
            if "usage" in job_metrics:
                sampler_quantum_seconds = job_metrics['usage'].get('quantum_seconds')

            # Report sampler timing in verbose mode for comparison with execution_spans
            if verbose:
                print(f"... sampler timing: creating={exec_creating_time:.3f}s, "
                      f"running={exec_running_time:.3f}s")
                if sampler_quantum_seconds is not None:
                    print(f"... sampler quantum_seconds={sampler_quantum_seconds}s "
                          f"(rounded up from execution_spans)")

            # NOT overwriting exec_time here — execution_spans from
            # _extract_per_circuit_times is more precise (see note above)

        except Exception as e:
            if verbose:
                print(f'WARNING: incomplete sampler time metrics for batch')
                print(f"... exception = {e}")

    # In metrics, we use > 0.001 to indicate valid data; need to floor these values to 0.001
    exec_creating_time = max(0.001, exec_creating_time)
    exec_validating_time = max(0.001, exec_validating_time)
    exec_queued_time = max(0.001, exec_queued_time)
    exec_running_time = max(0.001, exec_running_time)

    # Store step breakdown for all circuits in the batch
    for ci in circuits_info:
        metrics.store_metric(ci["group"], ci["circuit"], 'exec_creating_time', exec_creating_time)
        metrics.store_metric(ci["group"], ci["circuit"], 'exec_validating_time', 0.001)
        metrics.store_metric(ci["group"], ci["circuit"], 'exec_queued_time', 0.001)
        metrics.store_metric(ci["group"], ci["circuit"], 'exec_running_time', exec_running_time)

    if verbose:
        print(f"... computed exec times: queued = {exec_queued_time}, creating/transpiling = {exec_creating_time}, validating = {exec_validating_time}, running = {exec_running_time}")



###########################################################################
# JOB RESULT WAITING AND STATUS CHECKING
# These functions handle waiting for job completion with retry logic,
# comfort dots, and early error/cancel detection using a background thread.
###########################################################################


def wait_for_result_threaded(job, job_id, circuits, is_local_simulator=False):
    """
    Wait for job completion with comfort dots and retry logic.

    For local simulators: calls job.result() directly (no thread overhead).
    For hardware: runs job.result() in a background thread so main thread
    can print comfort dots and check for errors.

    Returns: raw_result (or None if all retries exhausted)
    """

    # Fast path: simulators return immediately, no need for threading.
    # Also skip threading if job doesn't support status polling — no comfort dots possible.
    if is_local_simulator or not hasattr(job, 'status'):
        return _wait_on_job_result(job, job_id)

    # Hardware path: thread for job.result(), main thread for comfort dots

    # Shared state between main thread and worker thread.
    # These are arrays (not plain variables) so the worker function can
    # modify the contents — like passing a pointer in C.
    result_holder = [None]       # [0] = raw_result when done
    error_holder = [None]        # [0] = exception if all retries failed
    done_event = threading.Event()
    stop_event = threading.Event()   # signals worker to stop retrying

    def worker():
        """Background thread: call job.result() with retry logic."""
        try:
            result_holder[0] = _wait_on_job_result(job, job_id, stop_event)
        except Exception as e:
            error_holder[0] = e
        finally:
            done_event.set()     # instantly wakes the main thread

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Main thread: print comfort dots and check for errors while waiting.
    # done_event.wait(timeout=N) returns immediately when the event is set,
    # so there is ZERO delay between job.result() completing and us getting it.
    # The timeout only controls how often we print dots / check status.
    pollcount = 0
    while not done_event.is_set():

        # Wait for result — returns instantly if thread finished
        if done_event.wait(timeout=2.0):
            break   # result is ready

        # Check if cancellation was requested
        if cancel_requested:
            print(f'\n... cancelling job {job_id}')
            stop_event.set()  # stop background thread retries
            try:
                job.cancel()
            except Exception:
                pass  # best effort
            return None

        # Still waiting — check job status (with retry, robust to network errors)
        pollcount += 1
        status = _get_job_status(job, job_id)

        if status is None:
            # get_job_status already printed the error — bail out
            return None

        # Error: don't wait for thread, return immediately
        if status == JobStatus.ERROR or status == 'ERROR':
            print(f'\nERROR: job execution failed: {job_id}')
            stop_event.set()  # stop background thread retries
            if hasattr(job, 'error_message'):
                print(f"... {job.error_message()}")
            else:
                try:
                    _ = job.result()
                except Exception as ex:
                    print(f"... {ex}")
                    if verbose:
                        print(traceback.format_exc())
            return None

        # Cancelled: don't wait for thread, return immediately
        if status == JobStatus.CANCELLED or status == 'CANCELLED':
            print(f'\nWARNING: Job {job_id} was cancelled')
            stop_event.set()  # stop background thread retries
            close_session()   # session is dead on IBM's side, clear local ref
            return None

        # Comfort dots showing job state (matches old check_jobs pattern)
        if verbose:
            if status == JobStatus.QUEUED or status == 'QUEUED':
                if pollcount < 32 or pollcount % 15 == 0:
                    print('.', end='', flush=True)
            elif status == JobStatus.INITIALIZING or status == 'INITIALIZING':
                print('i', end='', flush=True)
            elif status == JobStatus.VALIDATING or status == 'VALIDATING':
                print('v', end='', flush=True)
            elif status == JobStatus.RUNNING or status == 'RUNNING':
                print('r', end='', flush=True)
            elif status == JobStatus.DONE or status == 'DONE':
                pass  # thread should be finishing momentarily

    if verbose and pollcount > 0:
        print()  # newline after dots

    # Thread is done — check for errors
    if error_holder[0] is not None:
        print(f'ERROR: Failed to get result for job {job_id} after all retries')
        if verbose: print(f"... exception = {error_holder[0]}")
        return None

    return result_holder[0]


def _wait_on_job_result(job, job_id, stop_event=None):
    """
    Call job.result() with retry logic for network errors.
    Reuses the existing wait_on_job_result pattern (40 retries, 15s apart).

    This runs in the background thread for hardware, or directly for simulators.
    If stop_event is provided and gets set, stops retrying immediately.
    """
    max_retries = 40
    retry_interval = 15

    result = None
    for retry_count in range(1, max_retries + 1):
        # Check if we've been told to stop (e.g., job was cancelled)
        if stop_event is not None and stop_event.is_set():
            return None

        try:
            result = job.result()
            break
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            ex_str = str(ex)
            # Don't retry on fatal errors (bad backend, auth failure, cancelled, etc.)
            fatal_keywords = ["403", "Forbidden", "not found", "authentication",
                              "not supported by the target", "IBMInputValueError",
                              "cancelled", "canceled"]
            if any(kw.lower() in ex_str.lower() for kw in fatal_keywords):
                print(f'ERROR: Fatal error for job {job_id} — {ex}')
                raise
            print(f'... error during job.result() for job {job_id} — retry {retry_count}/{max_retries}')
            if verbose: print(traceback.format_exc())
            if retry_count < max_retries:
                # Sleep in short intervals so we can check stop_event
                for _ in range(retry_interval):
                    if stop_event is not None and stop_event.is_set():
                        return None
                    time.sleep(1)

    if result is None:
        print(f'ERROR: Failed to get result for job {job_id} after {max_retries} retries')
        raise ValueError(f"Failed to get result for job {job_id}")

    return result


def _get_job_status(job, job_id):
    """
    Check job status with retry logic for network errors.
    Reuses the existing get_job_status pattern (3 retries, 15s apart).
    """
    max_retries = 3

    for retry_count in range(1, max_retries + 1):
        try:
            status = job.status()
            return status
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f'... error during job.status() for job {job_id} — retry {retry_count}/{max_retries}')
            if verbose: print(traceback.format_exc())
            if retry_count < max_retries:
                time.sleep(15)

    # All retries failed — return None (caller decides what to do)
    print(f'ERROR: Failed to get status for job {job_id} after {max_retries} retries')
    return None
