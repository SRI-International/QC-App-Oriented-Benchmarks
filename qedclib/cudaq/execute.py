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

from qedclib import qcb_mpi as mpi
from qedclib import metrics

# import the CUDA-Q package
import cudaq

verbose = False

# Timing decomposition — set by execute_circuits after each call.
# Callers (e.g. hamlib) can read these to report meaningful timing breakdowns.
last_transpile_time = 0.0    # cudaq has minimal transpilation
last_exec_time = 0.0         # wall-clock of cudaq.sample() calls
last_elapsed_time = 0.0      # total wall-clock of the execute call

# Cancel flag — set by request_cancel() to interrupt execution between circuits
cancel_requested = False

def request_cancel():
    """Request cancellation of the current execution."""
    global cancel_requested
    cancel_requested = True

# Auto-warmup: execute a tiny circuit on first call to execute_circuits() to prime the JIT
auto_warmup = True
_warmup_done = False

# Parallel execution flag — when True, execute_circuits() will distribute
# circuits across MPI ranks (equivalent to gpus_per_circuit=1).
# Provides a consistent API with the Qiskit execute module.
# Requires MPI with >1 rank to have any effect.
parallel_execution = False

_parallel_warning_shown = False

# If we have performed setup for multi-QPU / parallel circuits.
_hybrid_initialized = False


###################################################################
# PARALLEL EXECUTION SUPPORT (EXPERIMENTAL)
#
# Simple MPI-based parallel execution of independent circuits.
# Each MPI rank executes a subset of circuits on its assigned GPU.
#
# REQUIREMENTS:
# - Launch with: mpiexec -np N python -m mpi4py script.py -gpc 1
# - GPU binding must be set externally (Slurm --gpus-per-task=1 or CUDA_VISIBLE_DEVICES)
# - Each rank should see exactly ONE GPU
#
# NOTE: This is DIFFERENT from mgpu mode (which pools GPUs for one large circuit).
# When gpus_per_circuit=1, we override mgpu and use single-GPU per rank.
###################################################################

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


def _execute_parallel_mpi(circuits: list, num_shots: int) -> list:
    """
    Execute circuits in parallel using MPI rank distribution.

    Each MPI rank processes a contiguous block of circuits.
    GPU assignment is handled externally (Slurm/CUDA_VISIBLE_DEVICES).
    Results are gathered to rank 0.

    IMPORTANT: This overrides mgpu mode and sets single-GPU target per rank.
    """
    if not mpi.enabled():
        print("... MPI not enabled, using sequential")
        return None  # Signal to fall back to sequential

    rank = mpi.rank
    size = mpi.size

    if size < 2:
        if rank == 0 and verbose:
            print("... Only 1 MPI rank, using sequential")
        return None  # Signal to fall back to sequential

    # Skip MPI distribution for single-circuit arrays (no benefit, just overhead)
    if len(circuits) < 2:
        return None  # Fall back to sequential

    # Override mgpu mode - each rank uses single GPU
    # This is critical: mgpu pools all GPUs for ONE circuit, we want the opposite
    if rank == 0 and verbose:
        print(f"... MPI parallel: {size} ranks, {len(circuits)} circuits")
        print(f"... Setting single-GPU target (overriding mgpu)")

    cudaq.set_target("nvidia", option="fp32")

    # Synchronize before distribution
    mpi.barrier()

    # Calculate this rank's block of circuits
    num_circuits = len(circuits)
    block_indices = _get_block_indices(num_circuits, size)
    my_start, my_end = block_indices[rank]
    my_circuits = circuits[my_start:my_end]

    if rank == 0 and verbose:
        print(f"... Distribution: {[(s, e-s) for s, e in block_indices]} circuits per rank")

    # Execute this rank's circuits
    local_results = []
    for circuit in my_circuits:
        try:
            kernel, params = circuit[0], circuit[1]
            if noise is None:
                counts = cudaq.sample(kernel, *params, shots_count=num_shots)
            else:
                counts = cudaq.sample(kernel, *params, shots_count=num_shots, noise_model=noise)
            # Convert cudaq result to dict
            local_results.append({k: v for k, v in counts.items()})
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f'ERROR: Rank {rank} failed to execute circuit')
            print(f"... exception = {e}")
            local_results.append({})

    # Gather all results to rank 0
    all_results = mpi.gather(local_results)

    if rank == 0:
        # Flatten gathered results in correct order
        flattened = []
        for rank_results in all_results:
            flattened.extend(rank_results)
        if verbose:
            print(f"... Gathered {len(flattened)} results from {size} ranks")
        return flattened
    else:
        # Non-leader ranks return empty - caller should check mpi.leader()
        return []

def _execute_parallel_hybrid(circuits: list, num_shots: int, gpus_per_circuit: int) -> list:
    """
    Execute circuits in parallel using MPI subcommunicators.

    QPU topology (subcommunicators, GPU assignment) is initialized once by
    mpi.init_qpus() at benchmark startup. This function only performs the
    cudaq target setup on the first call, then dispatches circuits to QPU groups.

    Requires: mpi.init_qpus(gpus_per_circuit) already called.
    """
    global _hybrid_initialized

    world_rank = mpi.rank
    qpu_id = mpi.qpu_id
    num_qpus = mpi.num_qpus
    is_leader = mpi.is_qpu_leader
    leaders_comm = mpi.leaders_comm

    # One-time: point cudaq at the QPU subcommunicator.
    if not _hybrid_initialized:
        qpu_handle = mpi.get_qpu_handle()
        cudaq.set_target("nvidia", option="mgpu", comm=qpu_handle)
        cudaq.mpi.set_communicator(qpu_handle)
        _hybrid_initialized = True

    mpi.barrier()

    block_indices = _get_block_indices(len(circuits), num_qpus)
    my_start, my_end = block_indices[qpu_id]
    my_circuits = circuits[my_start:my_end]

    if world_rank == 0:
        print(f"... MPI hybrid: {mpi.size} ranks, "
              f"{gpus_per_circuit} GPUs/circuit, {num_qpus} QPUs, "
              f"{len(circuits)} circuits")
        print(f"... Distribution: {[(s, e-s) for s, e in block_indices]} circuits per QPU")

    local_results = []
    for circuit in my_circuits:
        kernel, params = circuit[0], circuit[1]
        counts = (
            cudaq.sample(kernel, *params, shots_count=num_shots) if noise is None
            else cudaq.sample(kernel, *params, shots_count=num_shots, noise_model=noise))
        local_results.append({k: v for k, v in counts.items()})

    if is_leader:
        all_results = leaders_comm.gather(local_results, root=0)
    else:
        all_results = None

    if world_rank == 0:
        flattened = [r for block in all_results for r in block]
        print(f"... Gathered {len(flattened)} results from {num_qpus} QPU leaders")
        return flattened
    return []

def _execute_groups_parallel_mpi(circuit_groups, num_shots_list):
    """
    Distribute circuit groups across MPI ranks for parallel execution.

    Each rank gets a contiguous block of groups and executes them sequentially.
    Within each group, all circuits run with that group's shot count.
    Results are gathered to rank 0 in original group order.

    Args:
        circuit_groups: list of lists of [kernel, [args]] tuples
        num_shots_list: list of ints, one per group

    Returns:
        (job_id, group_results) tuple:
        - job_id: identifier for the job
        - group_results: list of ExecutionResult, one per group (on rank 0)
                         empty list on non-leader ranks
    """
    rank = mpi.rank
    size = mpi.size

    # Override mgpu mode — each rank uses single GPU
    if rank == 0 and verbose:
        print(f"... MPI group-parallel: {size} ranks, {len(circuit_groups)} groups")

    cudaq.set_target("nvidia", option="fp32")
    mpi.barrier()

    # Distribute groups across ranks
    num_groups = len(circuit_groups)
    block_indices = _get_block_indices(num_groups, size)
    my_start, my_end = block_indices[rank]

    if rank == 0 and verbose:
        print(f"... Distribution: {[(s, e-s) for s, e in block_indices]} groups per rank")

    # Execute this rank's groups
    local_group_results = []
    for g_idx in range(my_start, my_end):
        circuits = circuit_groups[g_idx]
        shots = num_shots_list[g_idx]

        # Execute all circuits in this group
        counts_array = []
        for circuit in circuits:
            try:
                kernel, params = circuit[0], circuit[1]
                if noise is None:
                    result = cudaq.sample(kernel, *params, shots_count=shots)
                else:
                    result = cudaq.sample(kernel, *params, shots_count=shots, noise_model=noise)
                counts_array.append({k: v for k, v in result.items()})
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f'ERROR: Rank {rank} failed on group {g_idx}')
                print(f"... exception = {e}")
                counts_array.append({})

        local_group_results.append(counts_array)

    # Gather all group results to rank 0
    all_group_results = mpi.gather(local_group_results)

    pseudo_job = Job()

    if rank == 0:
        # Flatten: all_group_results is list-of-lists (one per rank), each containing
        # that rank's group results. Reconstruct in original group order.
        ordered_results = []
        for rank_results in all_group_results:
            for group_counts in rank_results:
                ordered_results.append(ExecutionResult(group_counts))

        if verbose:
            print(f"... Gathered {len(ordered_results)} group results from {size} ranks")

        return (pseudo_job.job_id(), ordered_results)
    else:
        # Non-leader ranks return empty results
        return (pseudo_job.job_id(), [ExecutionResult([]) for _ in circuit_groups])


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

# class ExecutionResult is a normalized result wrapper for quantum circuit execution.
# It accepts either a cudaq SampleResult (dict-like), a raw counts dict,
# or a list of counts dicts. get_counts() always returns:
#   - dict (or dict-like SampleResult) for a single circuit
#   - list[dict] for multiple circuits
# This normalization allows benchmark code to process results uniformly
# without knowing which execution path was used.
class ExecutionResult:

    def __init__(self, source):
        super().__init__()
        self._counts = None
        self.native_result = source  # preserve original result for vendor-specific access

        if isinstance(source, dict):
            self._counts = source
        elif isinstance(source, list):
            self._counts = self._normalize(source)
        else:
            # cudaq SampleResult (dict-like) — pass through
            self._counts = source

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
        context=None, gpus_per_circuit=None):
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
    
    # When hybrid mode (gpus_per_circuit > 1) is used, skip cudaq.set_target here.
    hybrid_mode = gpus_per_circuit is not None and gpus_per_circuit > 1
    if not hybrid_mode:
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

    # Handle noise_model in exec_options (same pattern as qiskit)
    # Values: None = no noise, "default" = built-in depolarization model, or a cudaq.NoiseModel object
    global noise
    if exec_options is not None and isinstance(exec_options, dict):
        if "noise_model" in exec_options:
            nm = exec_options["noise_model"]
            if nm == "default":
                set_default_noise_model()
            else:
                noise = nm
            if verbose:
                print(f"  ... noise model set from exec_options: {noise}")

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
    _execute_batched_circuits()
    
    
# Launch execution of all batched circuits
def _execute_batched_circuits ():
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
    try:
        if verbose: print(f"... during exec, noise model is: {noise}")
        if noise is None:
            if verbose: print("... executing without noise")
            result = cudaq.sample(circuit[0], *circuit[1], shots_count=num_shots)
        else:
            if verbose: print("... executing WITH noise")
            result = cudaq.sample(circuit[0], *circuit[1], shots_count=num_shots, noise_model=noise)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f'ERROR: Failed to execute circuit ({batched_circuit["group"]}/{batched_circuit["circuit"]})')
        print(f"... exception = {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise

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

    # compute and store circuit dimensional metrics
    circuit_metrics = compute_circuit_metrics(circuit)
    store_circuit_metrics(active_circuit["group"], active_circuit["circuit"], circuit_metrics)

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
        
        qc_xi = round(two_qubit_gates / max(total_gates, 1), 3)
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


# Compute circuit metrics and return as a tuple (matching qiskit signature).
# Does not store to metrics table — caller decides when to store.
# For cudaq, there is no transpile-based normalization, so tr_* = algorithmic.
def compute_circuit_metrics(qc):

    qc_depth, qc_size, qc_count_ops, qc_xi, qc_n2q = 0, 0, 0, 0, 0
    try:
        qc_size = qc[1][0]   # the first item after the kernel is always num_qubits
        qc_depth, qc_size, qc_count_ops, qc_xi, qc_n2q = get_circuit_metrics(qc, qc_size)
    except Exception as ex:
        print(f"ERROR attempting to compute circuit metrics")
        print(ex)

    # cudaq has no transpile-based normalization; tr_* = algorithmic
    return (qc_depth, qc_size, qc_xi, qc_n2q,
            qc_depth, qc_size, qc_xi, qc_n2q)


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


# Compute and store circuit metrics in one call (convenience function).
def compute_and_store_circuit_info(qc, group_id, circuit_id):

    metrics_values = compute_circuit_metrics(qc)
    store_circuit_metrics(group_id, circuit_id, metrics_values)


def compute_all_circuit_metrics(circuits, do_transpile_metrics=True, use_normalized_depth=True):
    """Compute and store circuit metrics for all circuits in a nested dict.

    Args:
        circuits: nested dict {group: {circuit_id: qc}} from get_circuits()
        do_transpile_metrics: accepted for API parity with qiskit (cudaq has no transpile)
        use_normalized_depth: accepted for API parity with qiskit (cudaq has no transpile)
    """
    for group_id in circuits:
        if not isinstance(circuits[group_id], dict):
            continue
        for circuit_id in circuits[group_id]:
            compute_and_store_circuit_info(circuits[group_id][circuit_id],
                                          str(group_id), str(circuit_id))


# Placeholder for API parity with qiskit — cudaq has no transpile-based normalization.
def compute_and_store_normalized_depth(qc, group_id, circuit_id):
    pass  # tr_* already set by compute_and_store_circuit_info


#########################################################################

# klunky way to know the last group executed 
last_group = None 

# Process a completed job
def job_complete (job):
    active_circuit = active_circuits[job]
        
    # get job result (DEVNOTE: this might be different for diff targets)
    cq_result = job.result()
    
    # create a normalized result object to return to the caller
    result = ExecutionResult(cq_result)
    
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
    _execute_batched_circuits()

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
        sleeptime = 0.2
        if pollcount > 6: sleeptime = 0.5
        if pollcount > 60: sleeptime = 5.0
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
    _execute_batched_circuits()
    
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
        sleeptime = 0.2
        if pollcount > 6: sleeptime = 0.5
        if pollcount > 60: sleeptime = 5.0
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
    # compute circuit metrics (not stored here — execute_circuit_immed is a low-level function)
    _circuit_metrics = compute_circuit_metrics(circuit)

    # get job result
    cq_result = result

    # create a normalized result object to return to the caller
    result = ExecutionResult(cq_result)
    
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


def execute_circuits(circuits, num_shots=100, wait=True, gpus_per_circuit=None):
    """
    Execute an array of circuits. Pure execution — no metrics,
    no result_handler, no dict knowledge.

    Always takes an array. Always returns (job_id, result).
    Cudaq circuits are [kernel, [args]] tuples.

    Args:
        circuits: list of [kernel, [args]] tuples
        num_shots: shots per circuit
        wait: if True (default), block until results are ready.
              if False, return immediately with result=None (not yet supported for cudaq).
        gpus_per_circuit: Number of GPUs to pool per circuit.
            None = use all available GPUs together (mgpu if MPI, single GPU if not).
            1 = each GPU runs one circuit independently (max parallelism, requires MPI).
            M = M GPUs pool per circuit, P/M circuits in parallel (requires MPI, mpi.size % M == 0).

    Returns:
        (job_id, result) tuple:
        - job_id: identifier for the job (serializable)
        - result: ExecutionResult with get_counts() → list of dicts,
          or None if wait=False
    """

    global _warmup_done

    if verbose:
        print(f"... execute_circuits({len(circuits)}, {num_shots}, wait={wait}, gpus_per_circuit={gpus_per_circuit})")

    # Auto-warmup: run a tiny circuit to prime the JIT compiler on first execution
    if auto_warmup and not _warmup_done:
        _warmup_done = True
        try:
            @cudaq.kernel
            def _warmup_kernel():
                q = cudaq.qubit()
                h(q)
                mz(q)
            cudaq.sample(_warmup_kernel, shots_count=1)
            if verbose:
                print("... warmup circuit executed")
        except Exception:
            pass  # warmup is best-effort

    # Reset timing decomposition
    global last_transpile_time, last_exec_time, last_elapsed_time
    last_transpile_time = 0.0
    last_exec_time = 0.0
    last_elapsed_time = 0.0
    ts_execute = time.time()

    # Handle empty case
    if not circuits or len(circuits) == 0:
        pseudo_job = Job()
        return (pseudo_job.job_id(), ExecutionResult([]))

    counts_array = None
    per_circuit_times = []

    # Warn once if parallel requested but not available
    global _parallel_warning_shown
    if parallel_execution and (not mpi.enabled() or mpi.size < 2):
        if not _parallel_warning_shown:
            print(f"... WARNING: parallel_execution=True but only 1 GPU available (MPI not enabled or single rank)")
            print(f"... executing sequentially. Launch with mpiexec -np N for parallel execution.")
            _parallel_warning_shown = True

    # MPI parallel execution (gpus_per_circuit or parallel_execution flag)
    if (gpus_per_circuit is not None or parallel_execution) and mpi.enabled() and mpi.size > 1:
        if gpus_per_circuit == 1 or (parallel_execution and gpus_per_circuit is None):
            # Mode 3: each GPU runs one circuit independently (max parallelism)
            counts_array = _execute_parallel_mpi(circuits, num_shots)
        elif gpus_per_circuit > 1 and mpi.size % gpus_per_circuit == 0:
            # Mode 4: hybrid — M GPUs pool per circuit, P/M circuits in parallel
            counts_array = _execute_parallel_hybrid(circuits, num_shots, gpus_per_circuit)
        elif gpus_per_circuit > 1:
            if mpi.rank == 0:
                print(f"... WARNING: mpi.size ({mpi.size}) not divisible by "
                      f"gpus_per_circuit ({gpus_per_circuit}), using default execution")

    # Default: sequential execution (single GPU or mgpu mode)
    if counts_array is None:
        global cancel_requested
        cancel_requested = False
        counts_array = []
        per_circuit_times = []
        for circuit in circuits:
            if cancel_requested:
                print("\n... execution cancelled by user")
                break
            try:
                ts_circ = time.time()
                if noise is None:
                    result = cudaq.sample(circuit[0], *circuit[1], shots_count=num_shots)
                else:
                    result = cudaq.sample(circuit[0], *circuit[1], shots_count=num_shots, noise_model=noise)
                per_circuit_times.append(time.time() - ts_circ)

                # Convert cudaq SampleResult to dict for counts array
                counts = {key: val for key, val in result.items()}
                counts_array.append(counts)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f'ERROR: Failed to execute circuit')
                print(f"... exception = {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                counts_array.append({})
                per_circuit_times.append(0.0)

    # Create pseudo-Job for job_id consistency
    pseudo_job = Job()
    job_id = pseudo_job.job_id()

    # If not waiting, return job_id with no result (cudaq is always synchronous for now)
    if not wait:
        return (job_id, None)

    # Wrap in ExecutionResult
    results = ExecutionResult(counts_array)

    # Attach per-circuit timing if available (sequential execution only)
    if per_circuit_times:
        results._per_circuit_times = per_circuit_times
        last_exec_time = sum(per_circuit_times)

    last_elapsed_time = time.time() - ts_execute

    if verbose:
        print(f"... execute_circuits complete, job_id={job_id}")

    return (job_id, results)


def execute_circuit_groups(circuit_groups, num_shots_list=None, num_shots=None):
    """
    Execute groups of circuits, each group with its own shot count.

    Each group is a list of [kernel, [args]] tuples. Groups may have different
    numbers of circuits and different shot counts. When parallel_execution is
    True and MPI is available with >1 rank, groups are distributed across ranks.

    Args:
        circuit_groups: list of lists of [kernel, [args]] tuples
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

    # MPI group-level distribution: distribute groups across ranks
    if parallel_execution and mpi.enabled() and mpi.size > 1 and len(circuit_groups) > 1:
        return _execute_groups_parallel_mpi(circuit_groups, num_shots_list)

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


def process_circuit_results(circuits_info, results, job_id=None, elapsed_time=None, num_shots=None):
    """
    Map batch results back to individual circuits. For each circuit:
    wraps counts in ExecutionResult, calls result_handler, stores timing
    and job_id.

    Args:
        circuits_info: either:
            - list of dicts with keys "qc", "group", "circuit", "shots"
            - nested dict {group: {circuit_id: qc}} from get_circuits()
        results: result object with get_counts() returning list or dict.
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

    if verbose:
        print(f"... process_circuit_results({len(circuits_info)}, job_id={job_id})")

    # Extract per-circuit counts from batch result
    counts_list = results.get_counts()
    if isinstance(counts_list, dict):
        counts_list = [counts_list]  # single-element array was unwrapped

    # Validate result count matches circuit count
    if len(counts_list) != len(circuits_info):
        print(f'WARNING: result count mismatch — expected {len(circuits_info)}, got {len(counts_list)}')
        while len(counts_list) < len(circuits_info):
            counts_list.append({})

    # Use per-circuit times if available (from sequential execution), otherwise batch time
    per_circuit_times = getattr(results, '_per_circuit_times', None)

    # Process each circuit's result
    for idx, (ci, counts) in enumerate(zip(circuits_info, counts_list)):

        # Store timing metrics: per-circuit if available, otherwise batch elapsed_time
        if per_circuit_times and idx < len(per_circuit_times):
            circ_time = per_circuit_times[idx]
            metrics.store_metric(ci["group"], ci["circuit"], 'elapsed_time', circ_time)
            metrics.store_metric(ci["group"], ci["circuit"], 'exec_time', circ_time)
        else:
            if elapsed_time is not None:
                metrics.store_metric(ci["group"], ci["circuit"], 'elapsed_time', elapsed_time)
            metrics.store_metric(ci["group"], ci["circuit"], 'exec_time',
                                 elapsed_time if elapsed_time is not None else 0.0)

        # Store job_id for tracking/retrieval
        if job_id is not None:
            metrics.store_metric(ci["group"], ci["circuit"], 'job_id', job_id)

        # Wrap individual counts in ExecutionResult for result_handler
        circuit_result = ExecutionResult(counts)

        # Call the benchmark's result handler (computes fidelity etc.)
        if result_handler:
            try:
                result_handler(ci["qc"], circuit_result,
                              ci["group"], ci["circuit"], ci["shots"])
            except Exception as e:
                print(f'ERROR: failed in result_handler for circuit {ci["group"]} {ci["circuit"]}')
                print(f"... exception = {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()


def submit_circuits(circuits, num_shots=100, max_batch_size=None, batch_by_group=False,
                    gpus_per_circuit=None):
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
        gpus_per_circuit: passed through to execute_circuits for MPI support
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
            _execute_batch(batch, num_shots, max_batch_size, gpus_per_circuit)
    else:
        # Batch by max_batch_size regardless of group boundaries
        _execute_batch(circuits_info, num_shots, max_batch_size, gpus_per_circuit)


def _execute_batch(circuits_info, num_shots, max_batch_size, gpus_per_circuit=None):
    """Internal: execute circuits_info in chunks of max_batch_size."""
    batch_size = max_batch_size or len(circuits_info)
    for i in range(0, len(circuits_info), batch_size):
        batch = circuits_info[i:i + batch_size]
        circuits = [ci["qc"] for ci in batch]
        ts = time.time()
        job_id, results = execute_circuits(circuits, num_shots, gpus_per_circuit=gpus_per_circuit)
        elapsed_time = time.time() - ts
        if results is not None:
            process_circuit_results(batch, results, job_id=job_id, elapsed_time=elapsed_time)
        else:
            print(f'WARNING: No results for batch of {len(batch)} circuits (job {job_id}) — skipping')


