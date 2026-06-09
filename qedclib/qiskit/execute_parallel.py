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

def _remove_measurements(circuit):
    """
    Return a copy of `circuit` with measurement/barrier operations removed.

    This is needed before passing QED-C circuits into the parallel-experiment
    mapper, because the mapper may call circuit.inverse(). Qiskit cannot invert
    measurement operations, since measurements are non-unitary.

    We still create a classical register with the same size as the quantum
    register because some of the original parallel-experiment code assumes
    circuit.cregs[0].size exists and uses it as the logical circuit size.
    Final measurements are added back later by CircuitExperiment.measure_all().
    """
    from qiskit import QuantumCircuit

    # Keep both quantum and classical registers.
    # The circuit is measurement-free, but the classical register preserves
    # compatibility with legacy code that reads circuit.cregs[0].size.
    clean = QuantumCircuit(circuit.num_qubits, circuit.num_qubits, name=circuit.name)

    for inst in circuit.data:
        op = inst.operation

        # Measurements cannot be inverted; barriers are scheduling/visual markers
        # and are not needed for partitioning or mapping.
        if op.name in ("measure", "barrier"):
            continue

        # Re-map the original instruction's qubits onto the new clean circuit.
        qargs = [clean.qubits[circuit.find_bit(q).index] for q in inst.qubits]

        # Copy the operation so the new circuit is independent of the original.
        clean.append(op.copy(), qargs)

    return clean

def _run_qiskit_parallel_experiment(circuits, num_shots):
    """
    Execute a batch of QED-C generated Qiskit circuits using Qiskit's
    ParallelExperiment framework.

    QED-C normally executes a list of circuits sequentially or through its own
    batching path. This function replaces that execution step with the custom
    multiprogramming flow:

        QED-C circuits
            -> remove measurements
            -> partition hardware
            -> compute logical-to-physical mappings
            -> wrap each circuit as a Qiskit Experiment
            -> combine them with ParallelExperiment
            -> run once on the selected backend
            -> return per-circuit counts

    The returned value is a list of count dictionaries, one dictionary per input
    circuit. The caller later converts this list into QED-C's ExecutionResult.
    """
    import os
    import sys
    import networkx as nx

    import execute as ex

    # Locate the embedded parallel_experiment package inside qedclib/qiskit/.
    # The original parallel-experiment code was copied in as:
    #
    #   qedclib/qiskit/parallel_experiment/
    #       HA/
    #       src/
    #
    # These paths are inserted into sys.path so that the original imports inside
    # the parallel-experiment source continue to work without rewriting that
    # entire codebase into a Python package.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pe_root = os.path.join(this_dir, "parallel_experiment")
    pe_src = os.path.join(pe_root, "src")

    if pe_root not in sys.path:
        sys.path.insert(0, pe_root)
    if pe_src not in sys.path:
        sys.path.insert(0, pe_src)

    # Qiskit Experiments is used to combine multiple component experiments into
    # one composite experiment. Each component gets its own physical qubit set.
    from qiskit_experiments.framework import ParallelExperiment

    # When QED-C is using qasm_simulator, use ibmq_manhattan as the
    # hardware model for partitioning/mapping, but run the experiment on
    # an unconstrained GenericBackendV2 simulator.
    from qiskit.providers.fake_provider import GenericBackendV2


    # Imports from the existing parallel_experiment/src code.
    # These provide the hardware model, error matrices, hardware partitioning,
    # initial mapping, idle-qubit cleanup, and experiment wrapper.
    from hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
    from hardware.distance_matrx import (
        get_distance_matrix_cnot_error_cost,
        get_qubit_readout_error,
    )
    from partition_process.qubit_partition import (
        partition_hardware_heuristic,
        largest_circuit_logical_degree,
        hardware_qubit_physical_degree,
    )
    from find_initial_mapping import (
        get_simultaneous_partition,
        compute_initial_mapping,
    )
    from run_parallel_experiment import (
        strip_idle_qubits,
        CircuitExperiment,
        extract_physical_qubits,
    )

    # Get the backend selected by QED-C.
    # If QED-C is using qasm_simulator, it does not provide a real coupling map,
    # so we use the existing ibmq_manhattan model for mapping and a compatible
    # fake BackendV2 for running the ParallelExperiment.
    #
    # If QED-C is using a real/fake backend with topology, we use that backend
    # directly so the partitioning/mapping code sees the backend's actual
    # coupling map.
    backend = ex.backend
    backend_name = ex.get_backend_name(backend)

    print(f"... parallel experiment using backend = {backend_name}")

    if backend_name == "qasm_simulator":
        hardware = IBMQHardwareArchitecture("ibmq_manhattan")

        run_backend = GenericBackendV2(
            num_qubits=hardware.qubit_number,
            coupling_map=None,
            basis_gates=["id", "rz", "sx", "x", "cx"],
            seed=42,
        )
    else:
        hardware = IBMQHardwareArchitecture(backend)
        run_backend = backend

    # Convert the hardware coupling graph into the directed graph format
    # expected by the partitioning code.
    hardware_graph = nx.DiGraph(hardware._coupling_graph)

    # Error/readout information is used by the partitioner to prefer better
    # physical qubit regions.
    cnot_error_matrix = get_distance_matrix_cnot_error_cost(hardware)
    readout_error = get_qubit_readout_error(hardware)

    # Quick capacity check: the requested circuits must fit on the available
    # hardware qubits if they are to be run simultaneously.
    total_qubits = sum(c.num_qubits for c in circuits)
    if total_qubits > hardware.qubit_number:
        raise RuntimeError("Circuits do not fit on the selected hardware model.")

    # QED-C circuits usually already contain final measurements. The mapping code
    # may invert circuits internally, and measurements cannot be inverted, so we
    # remove measurements before partitioning/mapping. CircuitExperiment adds
    # final measurements back later.
    circuit_list = [_remove_measurements(c) for c in circuits]

    # Compute physical-qubit connectivity information and logical-qubit degree
    # information. These are used to match highly connected logical circuits to
    # suitable physical qubit regions.
    qubit_physical_degree, largest_physical_degree = hardware_qubit_physical_degree(
        hardware
    )
    largest_logical_degrees = [
        largest_circuit_logical_degree(c) for c in circuit_list
    ]

    # Select disjoint hardware partitions for all circuits.
    # Each accepted circuit should receive a physical subgraph where it can run
    # without overlapping another circuit's qubits.
    accepted_circuits, multiple_partition = get_simultaneous_partition(
        circuit_list,
        hardware_graph,
        hardware,
        cnot_error_matrix,
        readout_error,
        qubit_physical_degree,
        largest_physical_degree,
        largest_logical_degrees,
        2,                          # weight_lambda from original experiment
        partition_hardware_heuristic,
        0.1,                        # epsilon from original experiment
    )

    # If the partitioner cannot place all circuits, this function fails and the
    # caller can fall back to normal QED-C execution.
    if multiple_partition is None or len(accepted_circuits) != len(circuit_list):
        raise RuntimeError("Parallel partition failed or rejected some circuits.")

    # Compute the logical-qubit -> physical-qubit mapping for each accepted
    # circuit inside its assigned partition.
    circuit_partitions, per_circuit_mappings, _ = compute_initial_mapping(
        accepted_circuits,
        hardware,
        multiple_partition,
    )

    # Convert mapping dictionaries into ordered physical-qubit tuples required
    # by Qiskit Experiments.
    physical_qubits_per_circuit = extract_physical_qubits(
        accepted_circuits,
        circuit_partitions,
        per_circuit_mappings,
    )

    # Remove unused qubits from each circuit before wrapping it as a component
    # experiment. This keeps the component circuit size consistent with its
    # assigned physical qubit tuple.
    compact_circuits = [strip_idle_qubits(c) for c in accepted_circuits]

    # Wrap every circuit as a Qiskit BaseExperiment-compatible object. Each
    # CircuitExperiment knows which physical qubits it is allowed to use.
    experiments = [
        CircuitExperiment(
            circuit=compact_circuits[i],
            physical_qubits=physical_qubits_per_circuit[i],
            label=getattr(accepted_circuits[i], "name", f"circuit_{i}"),
        )
        for i in range(len(accepted_circuits))
    ]

    # Combine all component experiments into one ParallelExperiment. This creates
    # a single composite circuit where each original circuit runs on a disjoint
    # physical-qubit subset.
    parallel = ParallelExperiment(
        experiments=experiments,
        backend=run_backend,
        flatten_results=False,
    )

    # Keep transpilation minimal so Qiskit is less likely to remap a component
    # circuit outside its assigned physical-qubit subset.
    parallel.set_transpile_options(optimization_level=0)

    # Execute the combined experiment once.
    expdata = parallel.run(backend=run_backend, shots=num_shots)
    expdata.block_for_results()

    # Extract one counts dictionary per child/component experiment.
    # These counts may still need bitstring localization/normalization by the
    # caller before constructing QED-C's ExecutionResult.
    counts_list = []
    for child in expdata.child_data():
        datum = child.data(0)
        counts_list.append(datum.get("counts", datum))

    return counts_list

def _localize_counts(counts, num_qubits):
    """
    Convert counts returned by Qiskit ParallelExperiment into the bitstring
    format expected by QED-C.

    ParallelExperiment may return child experiment counts with extra classical
    bits from the full combined circuit. QED-C expects each circuit's counts to
    contain only that circuit's local output bits.

    Example:
        For a 3-qubit circuit, ParallelExperiment may return "011000".
        QED-C expects only "011".
    """
    local = {}

    for key, value in counts.items():
        # Remove spaces that Qiskit may insert between classical registers.
        key = key.replace(" ", "")

        # Keep only the local measurement bits for this circuit.
        local_key = key[:num_qubits]

        # Merge counts in case multiple padded global keys map to the same
        # local bitstring.
        local[local_key] = local.get(local_key, 0) + value

    return local

def execute_circuits_parallel(circuits, num_shots):
    """
    Execute a list of QED-C circuits using the integrated Qiskit
    ParallelExperiment path.

    This function is the QED-C entry point for parallel circuit execution.
    Instead of calling QED-C's normal sequential execute_circuits() path, it
    attempts to run the input circuits together as one parallel experiment.

    The flow is:

        1. Receive a list of QED-C generated QuantumCircuit objects.
        2. Call _run_qiskit_parallel_experiment(), which:
             - removes final measurements,
             - partitions the hardware into disjoint qubit regions,
             - maps each circuit to one region,
             - wraps each circuit as a Qiskit Experiment,
             - runs them together with ParallelExperiment.
        3. Convert the returned counts into QED-C's expected per-circuit
           bitstring format.
        4. Wrap the counts in QED-C's ExecutionResult object.
        5. If anything fails, fall back to QED-C's original sequential execution.

    Args:
        circuits:
            List of Qiskit QuantumCircuit objects generated by QED-C.

        num_shots:
            Number of shots to execute for each circuit.

    Returns:
        A tuple of:

            ("parallel_experiment", ExecutionResult)

        on successful parallel execution, or the normal execute_circuits()
        return value if the parallel path fails.
    """
    import execute as ex

    if ex.verbose:
        print(f"... execute_circuits_parallel: {len(circuits)} circuits, {num_shots} shots")

    print(f">>> execute_circuits_parallel [qiskit]: {len(circuits)} circuits, {num_shots} shots")

    try:
        # Run the circuits through the custom multiprogramming +
        # Qiskit ParallelExperiment pipeline.
        print("Uses the integrated Qiskit ParallelExperiment workflow. If the parallel path fails, execution automatically falls back to the standard QED-C execution path.")
        counts_list = _run_qiskit_parallel_experiment(circuits, num_shots)

        # ParallelExperiment may return each child result with extra classical
        # bits from the full combined circuit. Convert each result back to the
        # local bitstring width expected by QED-C.
        counts_list = [
            _localize_counts(counts_list[i], circuits[i].num_qubits)
            for i in range(len(counts_list))
        ]

        # Convert the raw counts list into QED-C's standard result object so the
        # rest of the benchmark framework does not need to know that the circuits
        # were executed through ParallelExperiment.
        result = ex.ExecutionResult(counts_list)

        return "parallel_experiment", result

    except Exception as err:
        # If partitioning, mapping, Qiskit Experiments, or backend execution
        # fails, preserve QED-C behavior by falling back to the original
        # sequential execution path.
        print(f"... parallel experiment failed: {err}")
        print("... falling back to normal QED-C execution")

        # Disable parallel_execution temporarily so execute_circuits() does not
        # recursively call execute_circuits_parallel().
        ex.parallel_execution = False
        try:
            return ex.execute_circuits(circuits, num_shots)
        finally:
            ex.parallel_execution = True


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
