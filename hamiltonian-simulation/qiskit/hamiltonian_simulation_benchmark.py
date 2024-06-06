"""
Hamiltonian-Simulation Benchmark Program - Qiskit

This program benchmarks Hamiltonian simulation using Qiskit. 
The central function is the `run()` method, which orchestrates the entire benchmarking process.

HamiltonianSimulation forms the trotterized circuit used in the benchmark.

HamiltonianSimulationExact runs a classical calculation that perfectly simulates hamiltonian evolution, although it does not scale well. 
"""

#This is Anish's branch
import json
import os
import sys
import time

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit"]
import execute as ex
import metrics as metrics

# Benchmark Name
benchmark_name = "Hamiltonian Simulation"

np.random.seed(0)

verbose = False

# Saved circuits and subcircuits for display
QC_ = None
XX_ = None
YY_ = None
ZZ_ = None
XXYYZZ_ = None
# Mirror Gates of the previous four gates
XX_mirror_ = None
YY_mirror_ = None
ZZ_mirror_ = None
XXYYZZ_mirror_ = None

# For validating the implementation of XXYYZZ operation
_use_XX_YY_ZZ_gates = True

# Import precalculated data to compare against
filename = os.path.join(os.path.dirname(__file__), os.path.pardir, "_common", "precalculated_data.json")
with open(filename, 'r') as file:
    data = file.read()
precalculated_data = json.loads(data)

############### Circuit Definition

def initial_state(n_spins: int, initial_state: str = "checker") -> QuantumCircuit:
    """
    Initialize the quantum state.
    
    Args:
        n_spins (int): Number of spins (qubits).
        initial_state (str): The chosen initial state. By default applies the checkerboard state, but can also be set to "ghz", the GHZ state.

    Returns:
        QuantumCircuit: The initialized quantum circuit.
    """
    qc = QuantumCircuit(n_spins)

    if initial_state.strip().lower() == "checkerboard" or initial_state.strip().lower() == "neele":
        # Checkerboard state, or "Neele" state
        for k in range(0, n_spins, 2):
            qc.x([k])
    elif initial_state.strip().lower() == "ghz":
        # GHZ state: 1/sqrt(2) (|00...> + |11...>)
        qc.h(0)
        for k in range(1, n_spins):
            qc.cx(k-1, k)

    return qc


def HamiltonianSimulation(n_spins: int, K: int, t: float, hamiltonian: str, w: float, hx: list[float], hz: list[float], method: int) -> QuantumCircuit:
    """
    Construct a Qiskit circuit for Hamiltonian simulation.

    Args:
        n_spins (int): Number of spins (qubits).
        K (int): The Trotterization order.
        t (float): Duration of simulation.
        hamiltonian (str): Which hamiltonian to run. "heisenberg" by default but can also choose "TFIM". 
        w (float): Strength of two-qubit interactions for heisenberg hamiltonian. 
        hx (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 
        hz (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 

    Returns:
        QuantumCircuit: The constructed Qiskit circuit.
    """
    num_qubits = n_spins
    secret_int = f"{K}-{t}"

    # Allocate qubits
    qr = QuantumRegister(n_spins)
    cr = ClassicalRegister(n_spins)
    qc = QuantumCircuit(qr, cr, name=f"hamsim-{num_qubits}-{secret_int}")
    tau = t / K

    h_x = hx[:n_spins]
    h_z = hz[:n_spins]

    hamiltonian = hamiltonian.strip().lower()

    if hamiltonian == "heisenberg": 

        init_state = "checkerboard"

        # apply initial state
        qc.append(initial_state(n_spins, init_state), qr)
        qc.barrier()

        # Loop over each Trotter step, adding gates to the circuit defining the Hamiltonian
        for k in range(K):
            # Pauli spin vector product
            [qc.rx(2 * tau * w * h_x[i], qr[i]) for i in range(n_spins)]
            [qc.rz(2 * tau * w * h_z[i], qr[i]) for i in range(n_spins)]
            qc.barrier()
            
            # Basic implementation of exp(i * t * (XX + YY + ZZ))
            if _use_XX_YY_ZZ_gates:
                # XX operator on each pair of qubits in linear chain
                for j in range(2):
                    for i in range(j%2, n_spins - 1, 2):
                        qc.append(xx_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

                # YY operator on each pair of qubits in linear chain
                for j in range(2):
                    for i in range(j%2, n_spins - 1, 2):
                        qc.append(yy_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

                # ZZ operation on each pair of qubits in linear chain
                for j in range(2):
                    for i in range(j%2, n_spins - 1, 2):
                        qc.append(zz_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
                        
            else:
                # Optimized XX + YY + ZZ operator on each pair of qubits in linear chain
                for j in range(2):
                    for i in range(j % 2, n_spins - 1, 2):
                        qc.append(xxyyzz_opt_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
            qc.barrier()

        if (method == 3):
            # Add mirror gates for negative time simulation
            for k in range(K): 
                # Basic implementation of exp(-i * t * (XX + YY + ZZ)):
                if _use_XX_YY_ZZ_gates:
                    # regular inverse of XX + YY + ZZ operators on each pair of quibts in linear chain
                    # XX operator on each pair of qubits in linear chain
                    for j in range(2):
                        for i in range(j%2, n_spins - 1, 2):
                            qc.append(zz_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

                    # YY operator on each pair of qubits in linear chain
                    for j in range(2):
                        for i in range(j%2, n_spins - 1, 2):
                            qc.append(yy_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

                    # ZZ operation on each pair of qubits in linear chain
                    for j in range(2):
                        for i in range(j%2, n_spins - 1, 2):
                            qc.append(xx_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])

                else:
                    # optimized Inverse of XX + YY + ZZ operator on each pair of qubits in linear chain
                    for j in range(2):
                        for i in range(j % 2, n_spins - 1, 2):
                            qc.append(xxyyzz_opt_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
                    
                # the Pauli spin vector product
                [qc.rz(-2 * tau * w * h_z[i], qr[i]) for i in range(n_spins)]
                [qc.rx(-2 * tau * w * h_x[i], qr[i]) for i in range(n_spins)]
                qc.barrier()
    
    elif hamiltonian == "tfim":
        h = 0.2  # Strength of transverse field
        init_state = "ghz"

        #apply initial state
        qc.append(initial_state(n_spins, init_state), qr)
        qc.barrier()

        # Calculate TFIM
        for k in range(K):
            for i in range(n_spins):
                qc.rx(2 * tau * h, qr[i])
            qc.barrier()

            for j in range(2):
                for i in range(j % 2, n_spins - 1, 2):
                    qc.append(zz_gate(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
            qc.barrier()
        
        if (method == 3):
            for k in range(k):
                for j in range(2):
                    for i in range(j % 2, n_spins - 1, 2):
                        qc.append(zz_gate_mirror(tau).to_instruction(), [qr[i], qr[(i + 1) % n_spins]])
                qc.barrier()
                for i in range(n_spins):
                    qc.rx(-2 * tau * h, qr[i])
                qc.barrier()

    else:
        raise ValueError("Invalid Hamiltonian specification.")

    # Measure all qubits
    for i_qubit in range(n_spins):
        qc.measure(qr[i_qubit], cr[i_qubit])

    # Save smaller circuit example for display
    global QC_
    if QC_ is None or n_spins <= 6:
        if n_spins < 9:
            QC_ = qc

    return qc

############### XX, YY, ZZ Gate Implementations

def xx_gate(tau: float) -> QuantumCircuit:
    """
    Simple XX gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The XX gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="xx_gate")
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    
    global XX_
    XX_ = qc
    
    return qc

def yy_gate(tau: float) -> QuantumCircuit:
    """
    Simple YY gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The YY gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="yy_gate")
    qc.s(qr[0])
    qc.s(qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.sdg(qr[0])
    qc.sdg(qr[1])

    global YY_
    YY_ = qc

    return qc

def zz_gate(tau: float) -> QuantumCircuit:
    """
    Simple ZZ gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The ZZ gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="zz_gate")
    qc.cx(qr[0], qr[1])
    qc.rz(3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])

    global ZZ_
    ZZ_ = qc

    return qc

def xxyyzz_opt_gate(tau: float) -> QuantumCircuit:
    """
    Optimal combined XXYYZZ gate (with double coupling) on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The optimal combined XXYYZZ gate circuit.
    """
    alpha = tau
    beta = tau
    gamma = tau
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="xxyyzz_opt")
    qc.rz(3.1416 / 2, qr[1])
    qc.cx(qr[1], qr[0])
    qc.rz(3.1416 * gamma - 3.1416 / 2, qr[0])
    qc.ry(3.1416 / 2 - 3.1416 * alpha, qr[1])
    qc.cx(qr[0], qr[1])
    qc.ry(3.1416 * beta - 3.1416 / 2, qr[1])
    qc.cx(qr[1], qr[0])
    qc.rz(-3.1416 / 2, qr[0])

    global XXYYZZ_
    XXYYZZ_ = qc

    return qc

############### Mirrors of XX, YY, ZZ Gate Implementations   
def xx_gate_mirror(tau: float) -> QuantumCircuit:
    """
    Simple XX mirror gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The XX_mirror_ gate circuit.
    """
    qr = QuantumRegister(2, 'q')
    qc = QuantumCircuit(qr, name="xx_gate_mirror")
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(-3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])

    global XX_mirror_
    XX_mirror_ = qc

    return qc

def yy_gate_mirror(tau: float) -> QuantumCircuit:
    """
    Simple YY mirror gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The YY_mirror_ gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="yy_gate_mirror")
    qc.s(qr[0])
    qc.s(qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(-3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.sdg(qr[0])
    qc.sdg(qr[1])

    global YY_mirror_
    YY_mirror_ = qc

    return qc   

def zz_gate_mirror(tau: float) -> QuantumCircuit:
    """
    Simple ZZ mirror gate on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The ZZ_mirror_ gate circuit.
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="zz_gate_mirror")
    qc.cx(qr[0], qr[1])
    qc.rz(-3.1416 * tau, qr[1])
    qc.cx(qr[0], qr[1])

    global ZZ_mirror_
    ZZ_mirror_ = qc

    return qc

def xxyyzz_opt_gate_mirror(tau: float) -> QuantumCircuit:
    """
    Optimal combined XXYYZZ mirror gate (with double coupling) on q0 and q1 with angle 'tau'.

    Args:
        tau (float): The rotation angle.

    Returns:
        QuantumCircuit: The optimal combined XXYYZZ_mirror_ gate circuit.
    """
    alpha = tau
    beta = tau
    gamma = tau
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="xxyyzz_opt_mirror")
    qc.rz(3.1416 / 2, qr[0])
    qc.cx(qr[1], qr[0])
    qc.ry(-3.1416 * beta + 3.1416 / 2, qr[1])
    qc.cx(qr[0], qr[1])
    qc.ry(-3.1416 / 2 + 3.1416 * alpha, qr[1])
    qc.rz(-3.1416 * gamma + 3.1416 / 2, qr[0])
    qc.cx(qr[1], qr[0])
    qc.rz(-3.1416 / 2, qr[1])

    global XXYYZZ_mirror_
    XXYYZZ_mirror_ = qc

    return qc

############### Result Data Analysis

def analyze_and_print_result(qc: QuantumCircuit, result, num_qubits: int, type: str, num_shots: int, hamiltonian: str, method: int) -> tuple:
    """
    Analyze and print the measured results. Compute the quality of the result based on operator expectation for each state.

    Args:
        qc (QuantumCircuit): The quantum circuit.
        result: The result from the execution.
        num_qubits (int): Number of qubits.
        type (str): Type of the simulation.
        num_shots (int): Number of shots.
        hamiltonian (str): Which hamiltonian to run. "heisenberg" by default but can also choose "TFIM". 
        method (int): Method for fidelity checking (1 for noiseless trotterized quantum, 2 for exact classical).

    Returns:
        tuple: Counts and fidelity.
    """
    counts = result.get_counts(qc)

    if verbose:
        print(f"For type {type} measured: {counts}")

    hamiltonian = hamiltonian.strip().lower()

    # Precalculated correct distribution
    if method == 1 and hamiltonian == "heisenberg":
        correct_dist = precalculated_data[f"Heisenberg - Qubits{num_qubits}"]
    elif method == 2 and hamiltonian == "heisenberg":
        correct_dist = precalculated_data[f"Exact Heisenberg - Qubits{num_qubits}"]
    elif method == 1 and hamiltonian == "tfim":
        correct_dist = precalculated_data[f"TFIM - Qubits{num_qubits}"]
    elif method == 2 and hamiltonian == "tfim":
        correct_dist = precalculated_data[f"Exact TFIM - Qubits{num_qubits}"]
    elif method == 3 and hamiltonian == "heisenberg":
        correct_dist = {''.join(['1' if i % 2 == 0 else '0' for i in range(num_qubits)]) if num_qubits % 2 != 0 else ''.join(['0' if i % 2 == 0 else '1' for i in range(num_qubits)]):num_shots}
    elif method == 3 and hamiltonian == "tfim":
        raise NotImplemetedError("Not implemeted yet.")
    else:
        raise ValueError("Method is not 1 or 2, or hamiltonian is not tfim or heisenberg.")

    if verbose:
        print(f"Correct dist: {correct_dist}")

    # Use polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)

    return counts, fidelity

############### Benchmark Loop

def run(min_qubits: int = 2, max_qubits: int = 8, max_circuits: int = 3, skip_qubits: int = 1, num_shots: int = 100,
        use_XX_YY_ZZ_gates: bool = True, backend_id: str = 'qasm_simulator', provider_backend = None,
        hub: str = "ibm-q", group: str = "open", project: str = "main", exec_options = None,
        hamiltonian: str = "heisenberg", method: int = 1, 
        context = None):
    """
    Execute program with default parameters.

    Args:
        min_qubits (int): Minimum number of qubits (smallest circuit is 2 qubits).
        max_qubits (int): Maximum number of qubits.
        max_circuits (int): Maximum number of circuits to execute per group.
        skip_qubits (int): Increment of number of qubits.
        num_shots (int): Number of shots for each circuit execution.
        use_XX_YY_ZZ_gates (bool): Flag to use unoptimized XX, YY, ZZ gates.
        backend_id (str): Backend identifier for execution.
        provider_backend: Provider backend instance.
        hub (str): IBM Quantum hub.
        group (str): IBM Quantum group.
        project (str): IBM Quantum project.
        exec_options: Execution options.

        hamiltonian (str): Which hamiltonian to run. "heisenberg" by default but can also choose "TFIM". 
        method (int): Method for fidelity checking (1 for noiseless trotterized quantum, 2 for exact classical).
        context: Execution context.

    Returns:
        None
    """
    print(f"{benchmark_name} Benchmark Program - Qiskit")
    
    # Validate parameters (smallest circuit is 2 qubits)
    max_qubits = max(2, max_qubits)
    min_qubits = min(max(2, min_qubits), max_qubits)
    if min_qubits % 2 == 1: min_qubits += 1  # min_qubits must be even
    skip_qubits = max(1, skip_qubits)

    # Create context identifier
    if context is None: context = f"{benchmark_name} Benchmark"
    
    # Set the flag to use an XX YY ZZ shim if given
    global _use_XX_YY_ZZ_gates
    _use_XX_YY_ZZ_gates = use_XX_YY_ZZ_gates
    if _use_XX_YY_ZZ_gates:
        print("... using unoptimized XX YY ZZ gates")
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, type, num_shots):
        # Determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, expectation_a = analyze_and_print_result(qc, result, num_qubits, type, num_shots, hamiltonian, method)
        metrics.store_metric(num_qubits, type, 'fidelity', expectation_a)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):

        # Reset random seed
        np.random.seed(0)

        # Determine number of circuits to execute for this group
        num_circuits = max(1, max_circuits)
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")

        # Parameters of simulation
        #### CANNOT BE MODIFIED W/O ALSO MODIFYING PRECALCULATED DATA #########
        w = precalculated_data['w']  # Strength of disorder
        k = precalculated_data['k']   # Trotter error.
               # A large Trotter order approximates the Hamiltonian evolution better.
               # But a large Trotter order also means the circuit is deeper.
               # For ideal or noise-less quantum circuits, k >> 1 gives perfect Hamiltonian simulation.
        t = precalculated_data['t']  # Time of simulation
        #######################################################################

        # Loop over only 1 circuit
        for circuit_id in range(num_circuits):
            ts = time.time()
            hx = precalculated_data['hx'][:num_qubits]  # Precalculated random numbers between [-1, 1]
            hz = precalculated_data['hz'][:num_qubits]

            qc = HamiltonianSimulation(num_qubits, K=k, t=t, hamiltonian=hamiltonian, w=w, hx = hx, hz = hz, method = method)
            metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time() - ts)
            qc.draw()
            
            # Collapse the sub-circuits used in this benchmark (for Qiskit)
            qc2 = qc.decompose()

            # Submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc2, num_qubits, circuit_id, num_shots)
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
    
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    # Print a sample circuit
    print("Sample Circuit:")
    print(QC_ if QC_ is not None else "  ... too large!")
   
    if _use_XX_YY_ZZ_gates:
        print("\nXX, YY, ZZ =")
        print(XX_)
        print(YY_)
        print(ZZ_)
    else:
        print("\nXXYYZZ_opt =")
        print(XXYYZZ_)
       
    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - {benchmark_name} - Qiskit")

if __name__ == '__main__': run()

# if no noise model, put exec_options = {"noise_model" : None} as a parameter for run().