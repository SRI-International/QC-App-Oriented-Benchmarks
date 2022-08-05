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
############################
# Execute Module - Fire Opal
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

from qiskit import execute, Aer
from qiskit.qobj import QobjExperimentHeader
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

from qctrl import Qctrl

# Use Aer qasm_simulator by default
simulator_backend = Aer.get_backend("qasm_simulator")
use_real_device = False
device_backend_name = None
device_credentials = None
qctrl = None

# Initialize circuit execution module
# Create array of batched circuits and a dict of active circuits
# Configure a handler for processing circuits on completion

batched_circuits = []
active_circuits = []
result_handler = None


# Special object class to hold job information and used as a dict key
class Job:
    pass


# Initialize the execution module, with a custom result handler
def init_execution(handler):
    global batched_circuits, result_handler
    batched_circuits.clear()
    active_circuits.clear()
    result_handler = handler


# Set the backend for execution
def set_execution_target(backend_id="simulator", credentials=None):
    """
    Used to run jobs on a real hardware
    :param backend_id:  device name. List of available devices depends on the provider
    :credentials:  credentials for accessing a real device

    example usage.
    set_execution_target(backend_id)
    """
    global use_real_device, device_backend_name, qctrl, device_credentials

    if backend_id.startswith("fire_opal_"):
        possible_backend_name = backend_id.split("fire_opal_")[1]
        if possible_backend_name in [
            "ibmq_jakarta", "ibm_lagos", "ibmq_guadalupe",
        ]:
            device_backend_name = possible_backend_name
            device_credentials = credentials
            use_real_device = True
            qctrl = Qctrl()
        else:
            print(f"ERROR: Cannot use {possible_backend_name} as a backend device.")
    else:
        raise ValueError(f"ERROR: Unknown backend_id: {backend_id}")

    # create an informative device name
    device_name = backend_id
    metrics.set_plot_subtitle(f"Device = {device_name}")


# Submit circuit for execution
# This version executes immediately and calls the result handler
def submit_circuit(qc, group_id, circuit_id, shots=100):
    # store circuit in array with submission time and circuit info
    batched_circuits.append(
        {
            "qc": qc,
            "group": str(group_id),
            "circuit": str(circuit_id),
            "submit_time": time.time(),
            "shots": shots,
        }
    )
    # print("... submit circuit - ", str(batched_circuits[len(batched_circuits) - 1]))


# Launch execution of all batched circuits
def execute_circuits():
    # list of QuantumCircuits to run
    circuits = []

    # use the same number of shots for all circuits
    max_shots = max(batched_circuit["shots"] for batched_circuit in batched_circuits)

    for batched_circuit in batched_circuits:
        batched_circuit["shots"] = max_shots
        active_circuit = copy.copy(batched_circuit)
        active_circuit["launch_time"] = time.time()

        # Store circuit dimensional metrics
        store_circuit_metrics(active_circuit)

        # add circuit to list of QuantumCircuits to run
        circuits.append(batched_circuit["qc"])

        # add circuit to active circuits
        active_circuits.append(active_circuit)

    # Initiate execution
    if use_real_device:
        counts_list = qctrl.functions.calculate_fire(
            circuits=[circuit.qasm() for circuit in circuits],
            shot_count=max_shots,
            credentials=device_credentials,
            backend_override=device_backend_name
        ).results["results"]
    else:
        counts_list = execute(
            circuits, simulator_backend, shots=max_shots
        ).result().get_counts()

    for (counts, active_circuit) in zip(counts_list, active_circuits):
        job = Job()
        job.result = counts_to_result(counts, active_circuit["qc"].name)

        # Here we complete the job immediately
        job_complete(job, active_circuit)

    active_circuits.clear()
    batched_circuits.clear()


# Convert counts dictionary to Qiskit Result
def counts_to_result(counts, name):
    experiment_result = ExperimentResult(
        data=ExperimentResultData(counts=counts),
        shots=sum(counts.values()),
        success=True,
        header=QobjExperimentHeader(name=name)
    )
    return Result(
        backend_name=None,
        backend_version=None,
        job_id=None,
        qobj_id=None,
        success=True,
        results=[experiment_result]
    )


# Store circuit dimensional metrics
def store_circuit_metrics(active_circuit):
    qc = active_circuit["qc"]

    # obtain initial circuit size metrics
    qc_depth = qc.depth()
    qc_size = qc.size()
    qc_count_ops = qc.count_ops()
    qc_xi = 0

    # iterate over the ordereddict to determine xi (ratio of 2 qubit gates to one qubit gates)
    n1q, n2q = 0, 0
    if qc_count_ops is not None:
        for key, value in qc_count_ops.items():
            if key == "measure":
                continue
            if key == "barrier":
                continue
            if key.startswith("c") or key.startswith("mc"):
                n2q += value
            else:
                n1q += value
        qc_xi = n2q / (n1q + n2q)

    metrics.store_metric(
        active_circuit["group"], active_circuit["circuit"], "depth", qc_depth
    )
    metrics.store_metric(
        active_circuit["group"], active_circuit["circuit"], "size", qc_size
    )
    metrics.store_metric(
        active_circuit["group"], active_circuit["circuit"], "xi", qc_xi
    )

    # default the transpiled metrics to the same
    metrics.store_metric(
        active_circuit["group"], active_circuit["circuit"], "tr_depth", qc_depth
    )
    metrics.store_metric(
        active_circuit["group"], active_circuit["circuit"], "tr_size", qc_size
    )
    metrics.store_metric(
        active_circuit["group"], active_circuit["circuit"], "tr_xi", qc_xi
    )
    metrics.store_metric(
        active_circuit["group"], active_circuit["circuit"], "tr_n2q", n2q
    )


# Process a completed job
def job_complete(job, active_circuit):
    # get job result (DEVNOTE: this might be different for diff targets)
    result = job.result
    # print("... result = ", str(result))

    # get measurement array and shot count
    counts = result.get_counts()
    actual_shots = sum(counts.values())
    # print(f"actual_shots = {actual_shots}")

    if actual_shots != active_circuit["shots"]:
        print(f"WARNING: requested shots not equal to actual shots: {actual_shots}")

    metrics.store_metric(
        active_circuit["group"],
        active_circuit["circuit"],
        "elapsed_time",
        time.time() - active_circuit["submit_time"],
    )

    metrics.store_metric(
        active_circuit["group"],
        active_circuit["circuit"],
        "exec_time",
        time.time() - active_circuit["launch_time"],
    )

    # If a handler has been established, invoke it here with result object
    if result_handler:
        result_handler(
            active_circuit["qc"],
            result,
            active_circuit["group"],
            active_circuit["circuit"],
        )
