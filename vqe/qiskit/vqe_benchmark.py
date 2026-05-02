"""
Variational Quantum Eigensolver Benchmark Program - Qiskit

NOTE: The benchmark-level code in this file will be migrated to the parent directory.
This file will eventually contain only the Qiskit-specific kernel code.
To run this benchmark, use the script in the parent directory:
    python vqe/vqe_benchmark.py
"""

import json
import os
import time

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter

import execute as ex
from _common import metrics

# Benchmark Name
benchmark_name = "VQE Simulation"

verbose= False

# saved circuits for display
QC_ = None
Hf_ = None
CO_ = None

################### Circuit Definition #######################################

# Construct a Qiskit circuit for VQE Energy evaluation with UCCSD ansatz
# param: n_spin_orbs - The number of spin orbitals.
# return: return a Qiskit circuit for this VQE ansatz
def VQEEnergy(n_spin_orbs, na, nb, circuit_id=0, method=1):

    # number of alpha spin orbitals
    norb_a = int(n_spin_orbs / 2)

    # construct the Hamiltonian
    qubit_op = ReadHamiltonian(n_spin_orbs)

    # allocate qubits
    num_qubits = n_spin_orbs

    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr, name=f"vqe-ansatz({method})-{num_qubits}-{circuit_id}")

    # initialize the HF state
    Hf = HartreeFock(num_qubits, na, nb)
    qc.append(Hf, qr)

    # form the list of single and double excitations 
    excitationList = []
    for occ_a in range(na):
        for vir_a in range(na, norb_a):
            excitationList.append((occ_a, vir_a))

    for occ_b in range(norb_a, norb_a+nb):
        for vir_b in range(norb_a+nb, n_spin_orbs):
            excitationList.append((occ_b, vir_b))

    for occ_a in range(na):
        for vir_a in range(na, norb_a):
            for occ_b in range(norb_a, norb_a+nb):
                for vir_b in range(norb_a+nb, n_spin_orbs):
                    excitationList.append((occ_a, vir_a, occ_b, vir_b))

    # get cluster operators in Paulis
    pauli_list = readPauliExcitation(n_spin_orbs, circuit_id)

    # loop over the Pauli operators
    for index, PauliOp in enumerate(pauli_list):
        # get circuit for exp(-iP)
        cluster_qc = ClusterOperatorCircuit(PauliOp, excitationList[index])

        # add to ansatz
        qc.append(cluster_qc, [i for i in range(cluster_qc.num_qubits)])
        
    # method 1, only compute the last term in the Hamiltonian
    if method == 1:
        # last term in Hamiltonian
        qc_with_mea, is_diag = ExpectationCircuit(qc, qubit_op[1], num_qubits)

        # return the circuit
        return qc_with_mea

    # now we need to add the measurement parts to the circuit
    # circuit list 
    qc_list = []
    diag = []
    off_diag = []
    global normalization
    normalization = 0.0

    # add the first non-identity term
    identity_qc = qc.copy()
    identity_qc.measure_all()
    qc_list.append(identity_qc) # add to circuit list
    diag.append(qubit_op[1])
    normalization += abs(qubit_op[1].coeffs[0]) # add to normalization factor
    diag_coeff = abs(qubit_op[1].coeffs[0]) # add to coefficients of diagonal terms

    # loop over rest of terms 
    for index, p in enumerate(qubit_op[2:]):
        
        # get the circuit with expectation measurements
        qc_with_mea, is_diag = ExpectationCircuit(qc, p, num_qubits)

        # accumulate normalization 
        normalization += abs(p.coeffs[0])

        # add to circuit list if non-diagonal
        if not is_diag:
            qc_list.append(qc_with_mea)
        else:
            diag_coeff += abs(p.coeffs[0])

        # diagonal term
        if is_diag:
            diag.append(p)
        # off-diagonal term
        else:
            off_diag.append(p)

    # modify the name of diagonal circuit
    qc_list[0].name = qubit_op[1].to_list()[0][0] + " " + str(np.real(diag_coeff))
    normalization /= len(qc_list)
    return qc_list

# Function that constructs the circuit for a given cluster operator
def ClusterOperatorCircuit(pauli_op, excitationIndex):

    num_qubits = pauli_op.num_qubits

    # compute exp(-iP) with 1st order Trotter step
    qc_op = PauliEvolutionGate(pauli_op, synthesis=LieTrotter())
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_op, range(num_qubits))
    qc.name = f'Cluster Op {excitationIndex}'
    
    global CO_
    if CO_ == None or qc.num_qubits <= 4:
        if qc.num_qubits < 7: CO_ = qc

    # return this circuit
    return qc


# Function that adds expectation measurements to the raw circuits
def ExpectationCircuit(qc, pauli, nqubit, method=2):

    # copy the unrotated circuit
    raw_qc = qc.copy()

    # whether this term is diagonal
    is_diag = True

    # primitive Pauli string
    PauliString = pauli.to_list()[0][0]

    # coefficient
    coeff = pauli.coeffs[0]

    # basis rotation
    for i, p in enumerate(PauliString):
    
        target_qubit = nqubit - i - 1 
        if (p == "X"):
            is_diag = False
            raw_qc.h(target_qubit)
        elif (p == "Y"):
            raw_qc.sdg(target_qubit)
            raw_qc.h(target_qubit)
            is_diag = False

    # perform measurements
    raw_qc.measure_all()

    # name of this circuit
    raw_qc.name = PauliString + " " + str(np.real(coeff))

    # save circuit
    global QC_
    if QC_ == None or nqubit <= 4:
        if nqubit < 7: QC_ = raw_qc

    return raw_qc, is_diag

# Function that implements the Hartree-Fock state 
def HartreeFock(norb, na, nb):

    # initialize the quantum circuit
    qc = QuantumCircuit(norb, name="Hf")
    
    # alpha electrons
    for ia in range(na):
        qc.x(ia)

    # beta electrons
    for ib in range(nb):
        qc.x(ib+int(norb/2))

    # Save smaller circuit
    global Hf_
    if Hf_ == None or norb <= 4:
        if norb < 7: Hf_ = qc

    # return the circuit
    return qc

################ Helper Functions

# Function that converts a list of single and double excitation operators to Pauli operators
def readPauliExcitation(norb, circuit_id=0):

    # load pre-computed data
    filename = os.path.join(os.path.dirname(__file__), f'./ansatzes/{norb}_qubit_{circuit_id}.txt')
    with open(filename) as f:
        data = f.read()
    ansatz_dict = json.loads(data)

    # initialize Pauli list
    pauli_list = []

    # current coefficients 
    cur_coeff = 1e5

    # current Pauli list 
    cur_list = []

    # loop over excitations
    for ext in ansatz_dict:

        if cur_coeff > 1e4:
            cur_coeff = ansatz_dict[ext]
            cur_list = [(ext, ansatz_dict[ext])]
        elif abs(abs(ansatz_dict[ext]) - abs(cur_coeff)) > 1e-4:
            pauli_list.append(SparsePauliOp.from_list(cur_list))
            cur_coeff = ansatz_dict[ext]
            cur_list = [(ext, ansatz_dict[ext])]
        else:
            cur_list.append((ext, ansatz_dict[ext]))
        
    # add the last term
    pauli_list.append(SparsePauliOp.from_list(cur_list))

    # return Pauli list
    return pauli_list

# Get the Hamiltonian by reading in pre-computed file
def ReadHamiltonian(nqubit):

    # load pre-computed data
    filename = os.path.join(os.path.dirname(__file__), f'./Hamiltonians/{nqubit}_qubit.txt')
    with open(filename) as f:
        data = f.read()
    ham_dict = json.loads(data)

    # pauli list 
    pauli_list = []
    for p in ham_dict:
        pauli_list.append( (p, ham_dict[p]) )

    # build Hamiltonian
    ham = SparsePauliOp.from_list(pauli_list)

    # return Hamiltonian
    return ham

################ Result Data Analysis

## Analyze and print measured results
## Compute the quality of the result based on measured probability distribution for each state
def analyze_and_print_result(qc, result, num_qubits, num_shots, references=None):

    # total circuit name (pauli string + coefficient)
    total_name = qc.name

    # pauli string
    pauli_string = total_name.split()[0]

    # get results counts
    counts = result.get_counts(qc)

    # get the correct measurement
    if (len(total_name.split()) == 2):
        correct_dist = references[pauli_string]
    else:
        circuit_id = int(total_name.split()[2])
        correct_dist = references[f"Qubits - {num_qubits} - {circuit_id}"]

    # compute fidelity
    fidelity = metrics.polarization_fidelity(counts, correct_dist)
    
    if verbose:
        print(f"... fidelity = {fidelity}")

    # modify fidelity based on the coefficient (only for method 2)
    # Note: method 1 total name has 3 components, method 2 has only 2; 
    if (len(total_name.split()) == 2):
           
        coefficient = abs(float(total_name.split()[1])) / normalization
        fidelity = {f : v * coefficient for f, v in fidelity.items()}
        if verbose:
            print(f"... total_name={total_name}, coefficient={coefficient}, product_fidelity={fidelity}")
    
    if verbose:
        print(f"... total fidelity = {fidelity}")
    
    return fidelity

############### Get Circuits

import inspect

MAX_QUBITS = 12

def get_circuits(
    # Standard args (common across benchmarks)
    min_qubits=4, max_qubits=8, skip_qubits=1,
    max_circuits=3, num_shots=4092, method=1,
    api=None,
):
    """Create VQE benchmark circuits.

    Standard args (common to all benchmarks):
        min_qubits: smallest circuit width (default 4, must be even)
        max_qubits: largest circuit width (default 8, clamped to 12)
        skip_qubits: increment between widths (default 1)
        max_circuits: max circuits per qubit group (default 3; method 2 forces 1)
        num_shots: measurement shots, stored in metrics (default 4092)
        method: 1=single ansatz circuit, 2=per-Pauli-term circuits (default 1)
        api: programming API; None = use qedc_set_api() value (default None)

    Returns (all_qcs, circuit_metrics) — nested circuit dict and creation metrics.
    """

    max_qubits = max(max_qubits, min_qubits)
    max_qubits = min(max_qubits, MAX_QUBITS)
    min_qubits = min(max(4, min_qubits), max_qubits)
    if min_qubits % 2 == 1: min_qubits += 1
    skip_qubits = max(1, skip_qubits)

    if method == 2: max_circuits = 1

    if max_qubits < 4:
        print(f"Max number of qubits {max_qubits} is too low to run method {method} of VQE algorithm")
        return {}, {}

    metrics.init_metrics()

    # Build circuits at each qubit width (even widths only)
    all_qcs = {}
    for input_size in range(min_qubits, max_qubits + 1, 2):

        np.random.seed(0)
        num_circuits = min(3, max_circuits)
        num_qubits = input_size

        na = int(num_qubits/4)
        nb = int(num_qubits/4)

        np.random.seed(0)
        ts = time.time()

        qc_list = []

        if method == 1:
            for circuit_id in range(num_circuits):
                qc_single = VQEEnergy(num_qubits, na, nb, circuit_id, method)
                qc_single.name = qc_single.name + " " + str(circuit_id)
                qc_list.append(qc_single)
        elif method == 2:
            qc_list = VQEEnergy(num_qubits, na, nb, 0, method)

        print(f"************\nCreating [{len(qc_list)}] circuits with num_qubits = {num_qubits}")
        all_qcs[str(num_qubits)] = {}

        # Store each circuit with its ID, decomposing sub-circuits
        for qc in qc_list:
            if method == 1:
                circuit_id = qc.name.split()[2]
            else:
                circuit_id = qc.name.split()[0]

            metrics.store_metric(input_size, circuit_id, 'create_time', time.time() - ts)
            qc2 = qc.decompose()
            all_qcs[str(num_qubits)][str(circuit_id)] = qc2

    return all_qcs, metrics.circuit_metrics


############### Run Circuits

def run_circuits(all_qcs,
    num_shots=4092, method=1, max_batch_size=None,
    backend_id=None, provider_backend=None,
    hub="ibm-q", group="open", project="main",
    exec_options=None, context=None, api=None,
):
    """Execute benchmark circuits and collect metrics.

    Args:
        all_qcs: circuit dict from get_circuits()
        num_shots: measurement shots per circuit (default 4092)
        method: algorithm method, for result analysis (default 1)
        max_batch_size: max circuits per batch; None = no limit (default None)
        backend_id: backend identifier (default None = qasm_simulator)
        provider_backend: provider backend instance (default None)
        hub, group, project: IBMQ credentials (defaults "ibm-q"/"open"/"main")
        exec_options: additional execution options dict (default None)
        context: context identifier for metrics (default None)
        api: programming API if not already initialized (default None)
    """
    ex.verbose = verbose

    if context is None:
        context = f"{benchmark_name} ({method}) Benchmark"

    # Result handler: loads precalculated reference data and computes fidelity
    def execution_handler(qc, result, num_qubits, circuit_id, num_shots):
        num_qubits = int(num_qubits)

        if len(qc.name.split()) == 2:
            filename = os.path.join(os.path.dirname(__file__),
                    f'../_common/precalculated_data_{num_qubits}_qubit.json')
        else:
            filename = os.path.join(os.path.dirname(__file__),
                    f'../_common/precalculated_data_{num_qubits}_qubit_method2.json')
        with open(filename) as f:
            references = json.load(f)

        fidelity = analyze_and_print_result(qc, result, num_qubits, num_shots,
                references=references)

        if len(qc.name.split()) == 2:
            metrics.store_metric(num_qubits, qc.name.split()[0], 'fidelity', fidelity)
        else:
            metrics.store_metric(num_qubits, qc.name.split()[2], 'fidelity', fidelity)

    # Set up execution target and submit all circuits as a batch
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)

    ex.compute_all_circuit_metrics(all_qcs)
    ex.submit_circuits(all_qcs, num_shots=num_shots, max_batch_size=max_batch_size)
    metrics.finalize_all_groups()


############### Plot Results

def plot_results(
    num_shots=4092, max_circuits=3,
    api=None, draw_circuits=True, plot_results=True,
):
    """Draw sample circuits and plot benchmark metrics.

    Args:
        num_shots: shots, for plot subtitle (default 4092)
        max_circuits: circuit reps, for plot subtitle (default 3)
        api: programming API name for plot title (default None)
        draw_circuits: draw sample circuit diagrams (default True)
        plot_results: generate metrics plots (default True)
    """
    if draw_circuits:
        print("Sample Circuit:"); print(QC_ if QC_ is not None else "  ... too large!")
        print("\nHartree Fock Generator 'Hf' ="); print(Hf_ if Hf_ is not None else " ... too large!")
        print("\nCluster Operator Example 'Cluster Op' ="); print(CO_ if CO_ is not None else " ... too large!")

    if plot_results:
        options = {"shots": num_shots, "reps": max_circuits}
        metrics.plot_metrics(
            f"Benchmark Results - {benchmark_name} - {api if api is not None else 'Qiskit'}",
            options=options)


############### Run (convenience)

def run(**kwargs):
    """Create circuits, execute, and plot. Accepts any arg from
    get_circuits(), run_circuits(), or plot_results()."""

    def _for(func):
        return {k: kwargs[k] for k in kwargs if k in inspect.signature(func).parameters}

    get_circuits_only = kwargs.pop('get_circuits', False)

    print(f"{benchmark_name} ({kwargs.get('method', 1)}) Benchmark Program - Qiskit")

    # Step 1: Create the benchmark circuits
    all_qcs, circuit_metrics = get_circuits(**_for(get_circuits))
    if not all_qcs: return

    # Step 2: If user just wants circuits, return them now
    if get_circuits_only:
        print(f"************\nReturning circuits and circuit information")
        return all_qcs, circuit_metrics

    # Step 3: Execute circuits on the target backend
    run_circuits(all_qcs, **_for(run_circuits))

    # Step 4: Draw sample circuit and plot metrics
    plot_results(**_for(plot_results))


if __name__ == "__main__":
    print("Please run this benchmark from the parent directory:")
    print("  python vqe/vqe_benchmark.py")
        
