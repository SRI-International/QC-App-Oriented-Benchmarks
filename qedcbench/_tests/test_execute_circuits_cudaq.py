"""
Quick tests for the new execute_circuits / process_circuit_results / submit_circuits functions.
CUDA-Q version — tests on local GPU simulator.

Run from benchmark root:
    python _tests/test_execute_circuits_cudaq.py
"""

import sys
from pathlib import Path

# Add benchmark root to path
benchmark_root = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, benchmark_root)

import cudaq
from qedclib import metrics
from qedclib.cudaq import execute as ex


# Define a simple kernel for testing
@cudaq.kernel
def bell_kernel(num_qubits: int):
    qubits = cudaq.qvector(num_qubits)
    h(qubits[0])
    for i in range(1, num_qubits):
        cx(qubits[0], qubits[i])
    mz(qubits)


@cudaq.kernel
def return_bit_kernel() -> bool:
    qubit = cudaq.qubit()
    x(qubit)
    return mz(qubit)


def create_circuit(num_qubits=2):
    """Create a cudaq circuit tuple [kernel, [args]]."""
    return [bell_kernel, [num_qubits]]


def create_circuits_info(num_circuits=3, num_qubits=2, num_shots=100):
    """Create a list of circuit info dicts for testing."""
    circuits_info = []
    for i in range(num_circuits):
        qc = create_circuit(num_qubits)
        circuits_info.append({
            "qc": qc,
            "group": str(num_qubits),
            "circuit": str(i),
            "shots": num_shots
        })
    return circuits_info


def create_circuits_dict(num_circuits=3, num_qubits=2):
    """Create a nested dict {group: {circuit_id: qc}} for testing submit_circuits."""
    circuits = {}
    circuits[str(num_qubits)] = {}
    for i in range(num_circuits):
        circuits[str(num_qubits)][str(i)] = create_circuit(num_qubits)
    return circuits


# Track result_handler calls for verification
handler_calls = []

def test_result_handler(qc, result, group, circuit_id, shots):
    """Test result handler that records each call."""
    counts = result.get_counts()
    handler_calls.append({
        "group": group, "circuit_id": circuit_id,
        "shots": shots, "counts": counts
    })


###########################################################################

def test_execute_circuits_basic():
    """Test: execute_circuits with 3 bell-state circuits returns correct results."""
    print("\n=== test_execute_circuits_basic ===")

    circuits = [create_circuit() for _ in range(3)]
    job_id, results = ex.execute_circuits(circuits, num_shots=100)

    assert job_id is not None, "job_id should not be None"
    assert results is not None, "results should not be None"

    counts_list = results.get_counts()
    if isinstance(counts_list, dict):
        counts_list = [counts_list]
    assert len(counts_list) == 3, f"Expected 3 count dicts, got {len(counts_list)}"

    for counts in counts_list:
        assert isinstance(counts, dict), f"Each count should be dict, got {type(counts)}"
        assert len(counts) > 0, "Counts should not be empty"

    print(f"  PASS: job_id={job_id}, got {len(counts_list)} count dicts")


def test_execute_circuits_single():
    """Test: 1-element array still works correctly."""
    print("\n=== test_execute_circuits_single ===")

    circuits = [create_circuit()]
    job_id, results = ex.execute_circuits(circuits, num_shots=100)

    counts_list = results.get_counts()
    if isinstance(counts_list, dict):
        counts_list = [counts_list]
    assert len(counts_list) == 1, f"Expected 1 count dict, got {len(counts_list)}"

    print(f"  PASS: single circuit, job_id={job_id}")


def test_execute_circuits_wait_false():
    """Test: wait=False returns job_id with result=None."""
    print("\n=== test_execute_circuits_wait_false ===")

    circuits = [create_circuit()]
    job_id, results = ex.execute_circuits(circuits, num_shots=100, wait=False)

    assert job_id is not None, "job_id should not be None even with wait=False"
    assert results is None, "results should be None when wait=False"

    print(f"  PASS: wait=False, job_id={job_id}, result=None")


def test_process_circuit_results():
    """Test: process_circuit_results calls result_handler for each circuit."""
    print("\n=== test_process_circuit_results ===")

    handler_calls.clear()

    ex.init_execution(test_result_handler)

    circuits_info = create_circuits_info(num_circuits=3)
    circuits = [ci["qc"] for ci in circuits_info]
    job_id, results = ex.execute_circuits(circuits, num_shots=100)

    ex.process_circuit_results(circuits_info, results, job_id=job_id, elapsed_time=1.23)

    assert len(handler_calls) == 3, f"Expected 3 handler calls, got {len(handler_calls)}"
    for i, call in enumerate(handler_calls):
        assert call["group"] == "2", f"Expected group '2', got '{call['group']}'"
        assert call["circuit_id"] == str(i), f"Expected circuit_id '{i}', got '{call['circuit_id']}'"
        assert call["shots"] == 100, f"Expected shots 100, got {call['shots']}"

    print(f"  PASS: result_handler called {len(handler_calls)} times with correct args")


def test_process_circuit_results_timing():
    """Test: elapsed_time and job_id stored in metrics for each circuit."""
    print("\n=== test_process_circuit_results_timing ===")

    handler_calls.clear()

    metrics.init_metrics()
    ex.init_execution(test_result_handler)

    circuits_info = create_circuits_info(num_circuits=2)
    circuits = [ci["qc"] for ci in circuits_info]
    job_id, results = ex.execute_circuits(circuits, num_shots=100)

    ex.process_circuit_results(circuits_info, results, job_id=job_id, elapsed_time=2.5)

    per_circuit_times = results._per_circuit_times
    for index, ci in enumerate(circuits_info):
        g, c = ci["group"], ci["circuit"]
        cm = metrics.circuit_metrics.get(g, {}).get(c, {})
        assert "elapsed_time" in cm, f"elapsed_time not stored for {g}/{c}"
        assert cm["elapsed_time"] == round(per_circuit_times[index], 4), (
            f"Expected elapsed_time {per_circuit_times[index]}, got {cm['elapsed_time']}"
        )
        assert "job_id" in cm, f"job_id not stored for {g}/{c}"
        assert cm["job_id"] == job_id, f"Expected job_id {job_id}, got {cm['job_id']}"

    print(f"  PASS: elapsed_time and job_id stored correctly")


def test_submit_circuits_end_to_end():
    """Test: submit_circuits executes and processes results with nested dict input."""
    print("\n=== test_submit_circuits_end_to_end ===")

    handler_calls.clear()

    metrics.init_metrics()
    ex.init_execution(test_result_handler)

    circuits = create_circuits_dict(num_circuits=4)
    ex.submit_circuits(circuits, num_shots=100)

    assert len(handler_calls) == 4, f"Expected 4 handler calls, got {len(handler_calls)}"

    for cid in circuits["2"]:
        cm = metrics.circuit_metrics.get("2", {}).get(cid, {})
        assert "elapsed_time" in cm, f"elapsed_time not stored for 2/{cid}"

    print(f"  PASS: end-to-end submit_circuits, {len(handler_calls)} circuits processed")


def test_submit_circuits_max_batch_size():
    """Test: max_batch_size chunks execution into multiple batches."""
    print("\n=== test_submit_circuits_max_batch_size ===")

    handler_calls.clear()

    metrics.init_metrics()
    ex.init_execution(test_result_handler)

    circuits = create_circuits_dict(num_circuits=6)
    ex.submit_circuits(circuits, num_shots=100, max_batch_size=2)

    assert len(handler_calls) == 6, f"Expected 6 handler calls, got {len(handler_calls)}"

    for cid in circuits["2"]:
        cm = metrics.circuit_metrics.get("2", {}).get(cid, {})
        assert "elapsed_time" in cm, f"elapsed_time not stored for 2/{cid}"

    print(f"  PASS: max_batch_size=2, all 6 circuits processed in 3 batches")


def test_submit_circuits_batch_by_group():
    """Test: batch_by_group=True groups circuits by qubit width."""
    print("\n=== test_submit_circuits_batch_by_group ===")

    handler_calls.clear()

    metrics.init_metrics()
    ex.init_execution(test_result_handler)

    # Create circuits from 2 different groups (qubit widths)
    circuits = {}
    for num_qubits in [2, 3]:
        circuits[str(num_qubits)] = {}
        for i in range(3):
            circuits[str(num_qubits)][str(i)] = create_circuit(num_qubits)

    ex.submit_circuits(circuits, num_shots=100, batch_by_group=True)

    assert len(handler_calls) == 6, f"Expected 6 handler calls, got {len(handler_calls)}"

    for i in range(3):
        assert handler_calls[i]["group"] == "2", f"Expected group '2' at index {i}"
    for i in range(3, 6):
        assert handler_calls[i]["group"] == "3", f"Expected group '3' at index {i}"

    print(f"  PASS: batch_by_group=True, groups processed separately")


def test_default_noise_model_channels():
    """Test: default model matches Qiskit rates and covers arbitrary qubits."""
    print("\n=== test_default_noise_model_channels ===")

    noise_model = ex.default_noise_model()

    for gate in ("x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "r1"):
        channels = noise_model.get_channels(gate, [47])
        assert len(channels) == 1, f"Expected one 1Q channel for {gate}"
        assert channels[0].parameters == [0.0005], (
            f"Wrong 1Q error rate for {gate}: {channels[0].parameters}"
        )

    for gate in ("x", "y", "z", "h", "rx", "ry", "rz", "r1"):
        channels = noise_model.get_channels(gate, [47], [46])
        assert len(channels) == 1, f"Expected one controlled channel for {gate}"
        assert channels[0].parameters == [0.005], (
            f"Wrong 2Q error rate for controlled {gate}: {channels[0].parameters}"
        )

    assert metrics.QV == 2048, f"Expected QV 2048, got {metrics.QV}"
    print("  PASS: Qiskit-matched 1Q/2Q rates registered on all qubits")


def test_noise_exec_options():
    """Test: dict/JSON overrides resolve without leaking across targets."""
    print("\n=== test_noise_exec_options ===")

    ex.set_noise_model("default")
    ex.set_execution_target(
        "density-matrix-cpu",
        exec_options='{"option": "fp64", "noise_model": "default"}',
    )
    assert isinstance(ex._resolve_noise_model(), cudaq.NoiseModel)
    assert ex._resolved_target_options["option"] == "fp64"

    ex.set_execution_target(
        "density-matrix-cpu", exec_options={"noise_model": None}
    )
    assert ex._resolve_noise_model() is None

    custom_noise = cudaq.NoiseModel()
    custom_noise.add_all_qubit_channel("x", cudaq.BitFlipChannel(1.0))
    ex.set_execution_target(
        "density-matrix-cpu", exec_options={"noise_model": custom_noise}
    )
    assert ex._resolve_noise_model() is custom_noise

    ex.set_execution_target("density-matrix-cpu")
    assert ex._resolve_noise_model() is ex.noise
    assert ex._resolve_noise_model() is not custom_noise
    print("  PASS: default, disabled, custom, and JSON noise options resolve correctly")


def test_default_noise_behavior():
    """Test: matched default noise produces Bell-state leakage."""
    print("\n=== test_default_noise_behavior ===")

    cudaq.set_random_seed(1234)
    ex.set_execution_target(
        "density-matrix-cpu", exec_options={"noise_model": None}
    )
    _, ideal_result = ex.execute_circuits([create_circuit()], num_shots=5000)
    ideal_counts = ideal_result.get_counts()
    assert set(ideal_counts).issubset({"00", "11"}), ideal_counts

    cudaq.set_random_seed(1234)
    ex.set_execution_target("density-matrix-cpu")
    _, noisy_result = ex.execute_circuits([create_circuit()], num_shots=5000)
    noisy_counts = noisy_result.get_counts()
    leakage = noisy_counts.get("01", 0) + noisy_counts.get("10", 0)
    assert leakage > 0, f"Default 2Q noise produced no Bell-state leakage: {noisy_counts}"
    print(f"  PASS: default model produced {leakage}/5000 leakage shots")


def test_run_path_noise():
    """Test: cudaq.run receives custom noise for typed-return kernels."""
    print("\n=== test_run_path_noise ===")

    circuit = [return_bit_kernel, [], {"result_width": 1}]
    ex.set_execution_target(
        "density-matrix-cpu", exec_options={"noise_model": None}
    )
    _, ideal_result = ex.execute_circuits([circuit], num_shots=100)
    assert ideal_result.get_counts() == {"1": 100}

    bit_flip = cudaq.NoiseModel()
    bit_flip.add_all_qubit_channel("x", cudaq.BitFlipChannel(1.0))
    ex.set_execution_target(
        "density-matrix-cpu", exec_options={"noise_model": bit_flip}
    )
    _, noisy_result = ex.execute_circuits([circuit], num_shots=100)
    assert noisy_result.get_counts() == {"0": 100}
    print("  PASS: typed-return run path applied the configured noise model")


def test_hamlib_observe_noise():
    """Test: HamLib SpinOperator execution uses the central observe wrapper."""
    print("\n=== test_hamlib_observe_noise ===")

    from qedcbench.hamlib.cudaq import hamlib_simulation_kernel as hamlib_kernel

    circuit = [hamlib_kernel.get_initial_state, [1]]
    pauli_terms = [({0: "Z"}, 1.0)]

    ex.set_execution_target(
        "density-matrix-cpu", exec_options={"noise_model": None}
    )
    ideal = hamlib_kernel.get_expectation(
        circuit, 1, pauli_terms, observe_fn=ex.observe
    )

    bit_flip = cudaq.NoiseModel()
    bit_flip.add_all_qubit_channel("x", cudaq.BitFlipChannel(1.0))
    ex.set_execution_target(
        "density-matrix-cpu", exec_options={"noise_model": bit_flip}
    )
    noisy = hamlib_kernel.get_expectation(
        circuit, 1, pauli_terms, observe_fn=ex.observe
    )

    assert ideal != noisy, f"Expected observe noise to change energy, got {ideal}"
    print(f"  PASS: HamLib expectation changed from {ideal} to {noisy}")


###########################################################################

if __name__ == '__main__':
    print("Testing new execute_circuits / process_circuit_results / submit_circuits (CUDA-Q)")
    print(f"Benchmark root: {benchmark_root}")

    tests = [
        test_execute_circuits_basic,
        test_execute_circuits_single,
        test_execute_circuits_wait_false,
        test_process_circuit_results,
        test_process_circuit_results_timing,
        test_submit_circuits_end_to_end,
        test_submit_circuits_max_batch_size,
        test_submit_circuits_batch_by_group,
        test_default_noise_model_channels,
        test_noise_exec_options,
        test_default_noise_behavior,
        test_run_path_noise,
        test_hamlib_observe_noise,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed!")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
