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
###############################################################################
# Parallel Execution Module - Qiskit (EXPERIMENTAL)
#
# Maps multiple circuits onto disjoint qubit regions of a single QPU
# for parallel execution, then decomposes results back to per-circuit counts.
#
# This module is loaded lazily from execute.py when parallel_execution is True.
# It accesses execute module state (sampler, backend, noise, etc.) via
# "import execute as ex".
###############################################################################

def execute_circuits_parallel(circuits, num_shots):
    """
    Execute circuits in parallel by mapping onto disjoint qubit regions of the QPU.

    When the real implementation is in place, this function will:
    1. Compose multiple circuits onto disjoint qubit subsets of a single large circuit
    2. Submit the composed circuit as one job
    3. Decompose the combined results back to per-circuit counts

    Currently a stub that calls back to execute_circuits() sequentially.
    Replace the stub section below with the real implementation
    (e.g., Qiskit ParallelExperiment or QuantumCircuit.compose() + initial_layout).

    Args:
        circuits: list of QuantumCircuit objects
        num_shots: shots per circuit (all circuits share the same shot count)

    Returns:
        (job_id, ExecutionResult) tuple, same as execute_circuits()
    """
    import execute as ex

    if ex.verbose:
        print(f"... execute_circuits_parallel: {len(circuits)} circuits, {num_shots} shots")

    #######################################################################
    # STUB: replace this section with the real parallel implementation.
    # The code below simply calls execute_circuits() sequentially.
    # It temporarily disables parallel_execution to avoid recursion
    # back into this function.
    #######################################################################
    print(f">>> execute_circuits_parallel [qiskit]: {len(circuits)} circuits, {num_shots} shots")
    print(f"... [STUB] parallel qubit mapping not yet implemented, executing sequentially")

    ex.parallel_execution = False
    try:
        result = ex.execute_circuits(circuits, num_shots)
    finally:
        ex.parallel_execution = True

    return result


def execute_circuit_groups_parallel(circuit_groups, num_shots_list):
    """
    Execute circuit groups in parallel by mapping circuits from multiple groups
    onto disjoint qubit regions of a single QPU.

    When the real implementation is in place, this function will:
    1. Determine qubit allocation per group (based on max circuit width in group)
    2. For groups with the same shot count: compose one circuit from each group
       onto disjoint qubit regions and submit as a single job
    3. For groups with different shot counts: batch same-shot groups together,
       execute each batch as above
    4. Decompose combined results back to per-group, per-circuit counts

    Qubit allocation uses the widest circuit in each group to determine that
    group's region. Narrower circuits within a group use a subset of the
    allocated qubits.

    Currently a stub that calls back to execute_circuit_groups() sequentially.
    Replace the stub section below with the real implementation.

    Args:
        circuit_groups: list of lists of QuantumCircuit objects
        num_shots_list: list of ints, one per group

    Returns:
        (job_id, group_results) tuple:
        - job_id: identifier for the job
        - group_results: list of ExecutionResult, one per group
    """
    import execute as ex

    if ex.verbose:
        group_sizes = [len(g) for g in circuit_groups]
        print(f"... execute_circuit_groups_parallel: {len(circuit_groups)} groups, "
              f"sizes={group_sizes}, shots={num_shots_list}")

    #######################################################################
    # STUB: replace this section with the real parallel implementation.
    # The code below calls execute_circuit_groups() sequentially.
    # It temporarily disables parallel_execution to avoid recursion
    # back into this function.
    #######################################################################
    print(f">>> execute_circuit_groups_parallel [qiskit]: {len(circuit_groups)} groups")
    print(f"... [STUB] parallel group execution not yet implemented, executing sequentially")

    ex.parallel_execution = False
    try:
        result = ex.execute_circuit_groups(circuit_groups, num_shots_list=num_shots_list)
    finally:
        ex.parallel_execution = True

    return result
