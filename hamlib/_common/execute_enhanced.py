'''
Hamiltonian Simulation Benchmark Program - Enhanced Circuit Execution
(C) Quantum Economic Development Consortium (QED-C) 2025.

Functions for executing arrays of circuits with optional shot distribution
based on Pauli term coefficient weights. Extracted from hamlib_simulation_benchmark.py.
'''

import numpy as np
from typing import List

debug = False


def execute_circuits_enhanced(
        backend_id: str = None,
        circuits: list = None,
        num_shots: int = 100,
        distribute_shots: bool = False,
        pauli_term_groups: list = None,
        ds_method: str = 'max_sq',
        gpus_per_circuit: int = None,
        verbose: bool = False,
    ) -> list:
    """
    Execute an array of circuits with the given number of shots on the specified backend.
    With default execution, the shots are divided evenly across all circuits in the group.
    If "distribute_shots" is set to True, the pauli_term_groups are used to distribute shots
    across the circuits based on the weights of the coefficients in the terms of the group
    and according to the ds_method (default = 'max_sq').

    Args:
        gpus_per_circuit: Number of GPUs to pool per circuit (None = use all available).
            1 = each GPU runs independently (max parallelism).
            M = M GPUs pool per circuit, P/M circuits in parallel.
    """
    if verbose:
        for circuit, group in list(zip(circuits, pauli_term_groups)):
            print(group)
            #print(circuit)

    # call api-specific function to execute circuits
    if not distribute_shots:
        #print(f"... number of shots per circuit = {int(num_shots / len(circuits))}")
        # execute the entire list of circuits, same shots each
        import execute as ex
        job_id, results = ex.execute_circuits(
                circuits = circuits,
                num_shots = int(num_shots / len(circuits)),
                gpus_per_circuit = gpus_per_circuit
                )
    else:
        # execute with shots distributed by weight of coefficients
        results, pauli_term_groups = execute_circuits_distribute_shots(
                backend_id = backend_id,
                circuits = circuits,
                num_shots = num_shots,
                groups = pauli_term_groups,
                ds_method = ds_method,
                gpus_per_circuit = gpus_per_circuit,
                verbose = verbose,
                )

    return results, pauli_term_groups


#########################################
# Distribute shots across circuits weighted by Pauli term coefficients,
# then bucket circuits with similar shot counts to reduce execution overhead.

def execute_circuits_distribute_shots(
        backend_id: str = None,
        circuits: list = None,
        num_shots: int = 100,
        groups: list = None,
        ds_method: str = 'max_sq',
        gpus_per_circuit: int = None,
        verbose: bool = False,
    ) -> list:

    if verbose or debug:
        print(f"... execute_circuits_distribute_shots({backend_id}, {len(circuits)}, {num_shots}, {groups})")

    # distribute shots; obtain total and distribute according to weights
    circuit_count = len(circuits)
    total_shots = num_shots         # to match current behavior
    if verbose or debug:
        print(f"... distributing shots, total shots = {total_shots} shots")

    # determine the number of shots to execute for each circuit, weighted by largest coefficient
    num_shots_list = get_distributed_shot_counts(total_shots, groups, ds_method)

    if verbose or debug:
        print(f"  ... num_shots_list = {num_shots_list}")

    # This approach  uses bucketing, to reduce number of circuits to be executed
    if debug:
        print("************* NEW WAY")
        for group in groups:
            print(group)

        # print(f"  in circuits = {circuits}")

    # determine optimal bucketing for these circuits, based on distribution of shots needed
    from hamlib._common.shot_distribution import bucket_numbers_kmeans, compute_bucket_averages

    # get buckets of terms with similar shots counts, and index of original position
    max_buckets = 3 if len(groups) < 50 else 4
    buckets_kmeans, indices_kmeans = bucket_numbers_kmeans(num_shots_list, max_buckets=max_buckets)

    # find the average number of shots required for each bucket
    # (sum of all shots for all circuits, nested, should be same as the incoming total)
    bucket_avg_shots = compute_bucket_averages(buckets_kmeans)

    if debug:
        print('  ... bucket kmeans:', buckets_kmeans)
        print('  ... indices_kmeans:', indices_kmeans)
        print('  ... bucket_avg_shots:', bucket_avg_shots)

    circuits_list = [[circuits[idx] for idx in indices] for indices in indices_kmeans]
    group_list = [[groups[idx] for idx in indices] for indices in indices_kmeans]

    if debug:
        # print(f"  circuits_list after bucketing = {circuits_list}")
        print(f"  ... group_list after bucketing....")
        for group in group_list:
            for g in group:
                print(g)
            print('-----')

    if verbose or debug:
        print(f"  ... bucketed shots list after bucketing = {buckets_kmeans} avg = {bucket_avg_shots}")

    # Execute each circuit in the list using the num_shots in the associated num_shots_list
    # Accumulate the count dicts in results object as if circuits were executed individually
    results = execute_circuits_with_mixed_shots(
        backend_id = backend_id,
        circuits_list = circuits_list,
        num_shots_list = bucket_avg_shots,
        gpus_per_circuit = gpus_per_circuit,
        )

    # Create a flattened list of all groups
    group_list = [item for sublist in group_list for item in sublist]

    if debug:
        print(f"... results.get_counts() = {results.get_counts()}")
        print(f"... results.get_counts() ({len(results.get_counts())}) = {results.get_counts()}")
        print(f"... group_list len = {len(group_list)}")
        print(f"  ... group_list after subgroups = ")
        for group in group_list:
            print(group)

    return results, group_list


# Allocate shots proportionally across Pauli term groups based on coefficient magnitudes.
# Supports 4 weighting methods: max_sq (default), mean_sq, max, mean.
def get_distributed_shot_counts(
        num_shots: int = 100,
        groups: list = None,
        ds_method: str = 'max_sq',
) -> List:
    weights = []
    for group in groups:
        w_sqs = [abs(coeff) ** 2 for pauli, coeff in group]
        ws = [abs(coeff) for pauli, coeff in group]

        if ds_method == 'max_sq':
            weights.append(max(w_sqs))
        elif ds_method == 'mean_sq':
            weights.append(sum(w_sqs) / len(w_sqs))
        elif ds_method == 'max':
            weights.append(max(ws))
        else:
            weights.append(sum(ws) / len(ws))

    # Step 2: Normalize weights to compute shot proportions
    total_weight = sum(weights)

    # make sure we don't have 0 shot allocation
    shot_allocations = [max(1, int((w / total_weight) * num_shots)) for w in weights]

    # Step 3: Adjust to ensure total shots match exactly
    while sum(shot_allocations) < num_shots:
        max_index = np.argmax(shot_allocations)
        shot_allocations[max_index] += 1
    while sum(shot_allocations) > num_shots:
        max_index = np.argmax(shot_allocations)
        shot_allocations[max_index] -= 1

    # print('shot allocation:', shot_allocations)

    return shot_allocations

#####################
# Execute bucketed circuit groups, each with a different shot count.
# Accumulates results into a single ExecutionResult as if all circuits ran individually.

def execute_circuits_with_mixed_shots(
        backend_id: str = None,
        circuits_list: list = None,
        num_shots_list: List[int] = None,
        gpus_per_circuit: int = None,
    ):

    # Loop over the circuit lists associated with each bucket
    counts_array = []
    for circuits, num_shots in zip(circuits_list, num_shots_list):

        if debug:
            # print(f"  ...    cccc = {circuits}")
            print(f"... len circs = {len(circuits)}")

        # execute this list of circuits, with same shots for each circuit in list
        import execute as ex
        job_id, results = ex.execute_circuits(
                circuits = circuits,
                num_shots = num_shots,
                gpus_per_circuit = gpus_per_circuit
                )

        # accumulate list of returned raw count dicts to parallel the group list
        # Qiskit returns an array of counts if array executed, but single counts for one circuit
        if len(circuits) > 1:
            for counts in results.get_counts():
                counts_array.append(counts)
        else:
            counts_array.append(results.get_counts())

    # Construct a normalized result object with counts structure to match circuits
    results = ex.ExecutionResult(counts_array)

    return results
