"""
Shor's Factoring Algorithm Benchmark - Qiskit
"""

import sys
sys.path[1:1] = ["_common", "_common/qiskit", "shors/_common", "quantum-fourier-transform/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit", "../../shors/_common", "../../quantum-fourier-transform/qiskit"]

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import math
from math import gcd
from fractions import Fraction
import time
import random
import numpy as np
np.random.seed(0)
import execute as ex
import metrics as metrics
from qft_benchmark import inv_qft_gate
from qft_benchmark import qft_gate

import copy
from utils import getAngles, getAngle, modinv, generate_numbers, choose_random_base, determine_factors

verbose = True

############### Circuit Definition

# Creation of the circuit that performs addition by a in Fourier Space
# Can also be used for subtraction by setting the parameter inv to a value different from 0
def phiADD(num_qubits, a):
    qc = QuantumCircuit(num_qubits, name="\u03C6ADD")

    angle = getAngles(a, num_qubits)
    for i in range(0, num_qubits):
        # addition
        qc.u1(angle[i], i)

    global PHIADD_
    if PHIADD_ == None or num_qubits <= 3:
        if num_qubits < 4: PHIADD_ = qc

    return qc


# Single controlled version of the phiADD circuit
def cphiADD(num_qubits, a):
    phiadd_gate = phiADD(num_qubits, a).to_gate()
    cphiadd_gate = phiadd_gate.control(1)
    return cphiadd_gate


# Doubly controlled version of the phiADD circuit
def ccphiADD(num_qubits, a):
    phiadd_gate = phiADD(num_qubits, a).to_gate()
    ccphiadd_gate = phiadd_gate.control(2)
    return ccphiadd_gate


# Circuit that implements doubly controlled modular addition by a (num qubits should be bit count for number N)
def ccphiADDmodN(num_qubits, a, N):
    qr_ctl = QuantumRegister(2)
    qr_main = QuantumRegister(num_qubits + 1)
    qr_ancilla = QuantumRegister(1)
    qc = QuantumCircuit(qr_ctl, qr_main, qr_ancilla, name="cc\u03C6ADDmodN")

    # Generate relevant gates for circuit
    ccphiadda_gate = ccphiADD(num_qubits + 1, a)
    ccphiadda_inv_gate = ccphiADD(num_qubits + 1, a).inverse()
    phiaddN_inv_gate = phiADD(num_qubits + 1, N).inverse()
    cphiaddN_gate = cphiADD(num_qubits + 1, N)

    # Create relevant temporary qubit lists
    ctl_main_qubits = [i for i in qr_ctl];
    ctl_main_qubits.extend([i for i in qr_main])
    anc_main_qubits = [qr_ancilla[0]];
    anc_main_qubits.extend([i for i in qr_main])

    # Create circuit
    qc.append(ccphiadda_gate, ctl_main_qubits)
    qc.append(phiaddN_inv_gate, qr_main)

    qc.append(inv_qft_gate(num_qubits + 1), qr_main)
    qc.cx(qr_main[-1], qr_ancilla[0])
    qc.append(qft_gate(num_qubits + 1), qr_main)

    qc.append(cphiaddN_gate, anc_main_qubits)
    qc.append(ccphiadda_inv_gate, ctl_main_qubits)

    qc.append(inv_qft_gate(num_qubits + 1), qr_main)

    qc.x(qr_main[-1])
    qc.cx(qr_main[-1], qr_ancilla[0])
    qc.x(qr_main[-1])

    qc.append(qft_gate(num_qubits + 1), qr_main)

    qc.append(ccphiadda_gate, ctl_main_qubits)

    global CCPHIADDMODN_
    if CCPHIADDMODN_ == None or num_qubits <= 2:
        if num_qubits < 3: CCPHIADDMODN_ = qc

    return qc


# Circuit that implements the inverse of doubly controlled modular addition by a
def ccphiADDmodN_inv(num_qubits, a, N):
    cchpiAddmodN_circ = ccphiADDmodN(num_qubits, a, N)
    cchpiAddmodN_inv_circ = cchpiAddmodN_circ.inverse()
    return cchpiAddmodN_inv_circ


# Creates circuit that implements single controlled modular multiplication by a. n represents the number of bits
# needed to represent the integer number N
def cMULTamodN(n, a, N):
    qr_ctl = QuantumRegister(1)
    qr_x = QuantumRegister(n)
    qr_main = QuantumRegister(n + 1)
    qr_ancilla = QuantumRegister(1)
    qc = QuantumCircuit(qr_ctl, qr_x, qr_main, qr_ancilla, name="cMULTamodN")

    # quantum Fourier transform only on auxillary qubits
    qc.append(qft_gate(n + 1), qr_main)

    for i in range(n):
        ccphiADDmodN_gate = ccphiADDmodN(n, (2 ** i) * a % N, N)

        # Create relevant temporary qubit list
        qubits = [qr_ctl[0]];
        qubits.extend([qr_x[i]])
        qubits.extend([i for i in qr_main]);
        qubits.extend([qr_ancilla[0]])

        qc.append(ccphiADDmodN_gate, qubits)

    # inverse quantum Fourier transform only on auxillary qubits
    qc.append(inv_qft_gate(n + 1), qr_main)

    global CMULTAMODN_
    if CMULTAMODN_ == None or n <= 2:
        if n < 3: CMULTAMODN_ = qc

    return qc


# Creates circuit that implements single controlled Ua gate. n represents the number of bits
# needed to represent the integer number N
def controlled_Ua(n, a, exponent, N):
    qr_ctl = QuantumRegister(1)
    qr_x = QuantumRegister(n)
    qr_main = QuantumRegister(n)
    qr_ancilla = QuantumRegister(2)
    qc = QuantumCircuit(qr_ctl, qr_x, qr_main, qr_ancilla, name=f"C-U^{a}^{exponent}")

    # Generate Gates
    a_inv = modinv(a ** exponent, N)
    cMULTamodN_gate = cMULTamodN(n, a ** exponent, N)
    cMULTamodN_inv_gate = cMULTamodN(n, a_inv, N).inverse()

    # Create relevant temporary qubit list
    qubits = [i for i in qr_ctl];
    qubits.extend([i for i in qr_x]);
    qubits.extend([i for i in qr_main])
    qubits.extend([i for i in qr_ancilla])

    qc.append(cMULTamodN_gate, qubits)

    for i in range(n):
        qc.cswap(qr_ctl, qr_x[i], qr_main[i])

    qc.append(cMULTamodN_inv_gate, qubits)

    global CUA_
    if CUA_ == None or n <= 2:
        if n < 3: CUA_ = qc

    return qc


# Execute Shor's Order Finding Algorithm given a 'number' to factor,
# the 'base' of exponentiation, and the number of qubits required 'input_size'

def ShorsAlgorithm(number, base, method, verbose=verbose):
    # Create count of qubits to use to represent the number to factor
    # NOTE: this should match the number of bits required to represent (number)
    n = int(math.ceil(math.log(number, 2)))

    # this will hold the 2n measurement results
    measurements = [0] * (2 * n)

    # Standard Shors Algorithm
    if method == 1:
        num_qubits = 4 * n + 2

        if verbose:
            print(
                f"... running Shors to factor number [ {number} ] with base={base} using num_qubits={n}")

        # Create a circuit and allocate necessary qubits
        qr_up = QuantumRegister(2 * n)  # Register for sequential QFT
        qr_down = QuantumRegister(n)  # Register for multiplications
        qr_aux = QuantumRegister(n + 2)  # Register for addition and multiplication
        cr_data = ClassicalRegister(2 * n)  # Register for measured values of QFT
        qc = QuantumCircuit(qr_up, qr_down, qr_aux, cr_data, name="main")

        # Initialize down register to 1 and up register to superposition state
        qc.h(qr_up)
        qc.x(qr_down[0])

        qc.barrier()

        # Apply Multiplication Gates for exponentiation
        for i in range(2 * n):
            cUa_gate = controlled_Ua(n, int(base), 2 ** i, number)

            # Create relevant temporary qubit list
            qubits = [qr_up[i]];
            qubits.extend([i for i in qr_down]);
            qubits.extend([i for i in qr_aux])

            qc.append(cUa_gate, qubits)

        qc.barrier()

        i = 0
        while i < ((qr_up.size - 1) / 2):
            qc.swap(qr_up[i], qr_up[2 * n - 1 - i])
            i = i + 1
        qc.append(inv_qft_gate(2 * n), qr_up)

        # Measure up register
        qc.measure(qr_up, cr_data)

    elif method == 2:

        # Create a circuit and allocate necessary qubits
        num_qubits = 2 * n + 3

        if verbose:
            print(f"... running Shors to factor number [ {number} ] with base={base} using num_qubits={n}")

        qr_up = QuantumRegister(1)  # Single qubit for sequential QFT
        qr_down = QuantumRegister(n)  # Register for multiplications
        qr_aux = QuantumRegister(n + 2)  # Register for addition and multiplication
        cr_data = ClassicalRegister(2 * n)  # Register for measured values of QFT
        cr_aux = ClassicalRegister(1)  # Register to reset the state of the up register based on previous measurements
        qc = QuantumCircuit(qr_down, qr_up, qr_aux, cr_data, cr_aux, name="main")

        # Initialize down register to 1
        qc.x(qr_down[0])

        # perform modular exponentiation 2*n times
        for k in range(2 * n):

            # one iteration of 1-qubit QPE

            # Reset the top qubit to 0 if the previous measurement was 1
            qc.x(qr_up).c_if(cr_aux, 1)
            qc.h(qr_up)

            cUa_gate = controlled_Ua(n, base ** (2 ** (2 * n - 1 - k)), number)
            qc.append(cUa_gate, [qr_up[0], qr_down, qr_aux])

            # perform inverse QFT --> Rotations conditioned on previous outcomes
            for i in range(2 ** k):
                qc.u1(getAngle(i, k), qr_up[0]).c_if(cr_data, i)

            qc.h(qr_up)
            qc.measure(qr_up[0], cr_data[k])
            qc.measure(qr_up[0], cr_aux[0])

    global QC_, QFT_
    if QC_ == None or n <= 2:
        if n < 3: QC_ = qc
    if QFT_ == None or n <= 2:
        if n < 3: QFT_ = qft_gate(n + 1)

    # turn the measured values into a number in [0,1) by summing their binary values
    ma = [(measurements[2 * n - 1 - i]*1. / (1 << (i + 1))) for i in range(2 * n)]
    y = sum(ma)

    y = 0.833
    # continued fraction expansion to get denominator (the period?)
    r = Fraction(y).limit_denominator(number - 1).denominator
    f = Fraction(y).limit_denominator(number - 1)

    if verbose:
        print(f"  ... y = {y}  fraction = {f.numerator} / {f.denominator}  r = {f.denominator}")

    # return the (potential) period
    return r

    # DEVNOTE: need to resolve this; currently not using the standard 'execute module'
    # measure and flush are taken care of in other methods; do not add here
    return qc


# Execute Shor's Factoring Algorithm given a 'number' to factor,
# the 'base' of exponentiation, and the number of qubits required 'input_size'


# Filter function, which defines the gate set for the first optimization
# (don't decompose QFTs and iQFTs to make cancellation easier)
'''
def high_level_gates(eng, cmd):
    g = cmd.gate
    if g == QFT or get_inverse(g) == QFT or g == Swap:
        return True
    if isinstance(g, BasicMathGate):
        #return False
        return True
        print("***************** should never get here !")
        if isinstance(g, AddConstant):
            return True
        elif isinstance(g, AddConstantModN):
            return True
        return False
    return eng.next_engine.is_available(cmd)
'''

# Attempt to execute Shor's Algorithm to find factors, up to a max number of tries
# Returns number of failures, 0 if success
def attempt_factoring(input_size, number, verbose):
    max_tries = 5
    trials = 0
    failures = 0
    while trials < max_tries:
        trials += 1

        # choose a base at random
        base = choose_random_base(number)
        if base == 0: break

        # execute the algorithm which determines the period given this base
        r = ShorsAlgorithm(input_size, number, base, verbose=verbose)

        # try to determine the factors from the period  'r'
        f1, f2 = determine_factors(r, base, number)

        # Success! if these are the factors and both are greater than 1
        if (f1 * f2) == number and f1 > 1 and f2 > 1:
            if verbose:
                print(f"  ==> Factors found :-) : {f1} * {f2} = {number}")
            break

        else:
            failures += 1
            if verbose:
                print(f"  ==> Bad luck: Found {f1} and {f2} which are not the factors")
                print(f"  ... trying again ...")

    return failures


############### Circuit end

# Print analyzed results
# Analyze and print measured results
# Expected result is always the secret_int, so fidelity calc is simple
def analyze_and_print_result(qc, result, num_qubits, marked_item, num_shots):
    if verbose: print(f"For marked item {marked_item} measured: {result}")
    key = format(marked_item, f"0{num_qubits}b")[::-1]
    fidelity = result[key]
    return fidelity


# Define custom result handler
def execution_handler(result, num_qubits, number, num_shots):
    # determine fidelity of result set
    num_qubits = int(num_qubits)
    fidelity = analyze_and_print_result(result, num_qubits - 1, int(number), num_shots)
    metrics.store_metric(num_qubits, number, 'fidelity', fidelity)


#################### Benchmark Loop

# Execute program with default parameters
def run(min_qubits=5, max_circuits=3, max_qubits=10, num_shots=100,
        verbose=verbose, interactive=False,
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main"):
    print("Shor's Factoring Algorithm Benchmark - Qiskit")

    # Generate array of numbers to factor
    numbers = generate_numbers()

    min_qubits = max(min_qubits, 5)  # need min of 5
    max_qubits = max(max_qubits, min_qubits)  # max must be >= min
    max_qubits = min(max_qubits, len(numbers))  # max cannot exceed available numbers

    # Initialize metrics module
    metrics.init_metrics()

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
                            hub=hub, group=group, project=project)

    if interactive:
        do_interactive_test(verbose=verbose)
        return;

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1):

        input_size = num_qubits - 1

        # determine number of circuits to execute for this group
        num_circuits = min(2 ** (input_size), max_circuits)

        print(
            f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}, input_size = {input_size}")
        ##print(f"... choices at {input_size} = {numbers[input_size]}")

        # determine array of numbers to factor
        numbers_to_factor = np.random.choice(numbers[input_size], num_circuits, False)
        ##print(f"... numbers = {numbers_to_factor}")

        # Number of times the factoring attempts failed for each qubit size
        failures = 0

        # loop over all of the numbers to factor
        for number in numbers_to_factor:
            # convert from np form (created by np.random)
            number = int(number)

            # create the circuit for given qubit size and secret string, store time metric
            ts = time.time()

            # not currently used
            # n_iterations = int(np.pi * np.sqrt(2 ** input_size) / 4)

            # attempt to execute Shor's Algorithm to find factors
            failures += attempt_factoring(input_size, number, verbose)

            metrics.store_metric(num_qubits, number, 'create_time', time.time() - ts)
            metrics.store_metric(num_qubits, number, 'exec_time', time.time() - ts)

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            # ex.submit_circuit(eng, qureg, num_qubits, number, num_shots)

        # Report how many factoring failures occurred
        if failures > 0:
            print(f"*** Factoring attempts failed {failures} times!")

        # execute all circuits for this group, aggregate and report metrics when complete
        # ex.execute_circuits()
        metrics.aggregate_metrics_for_group(num_qubits)
        metrics.report_metrics_for_group(num_qubits)

    # Alternatively, execute all circuits, aggregate and report metrics
    # ex.execute_circuits()
    # metrics.aggregate_metrics_for_group(num_qubits)
    # metrics.report_metrics_for_group(num_qubits)

    # print the last circuit created
    # print(qc)

    # Plot metrics for all circuit sizes
    metrics.plot_metrics("Benchmark Results - Shor's Factoring Algorithm - Qiskit")


# For interactive_shors_factoring testing
def do_interactive_test(verbose):
    done = False
    while not done:

        s = input('\nEnter the number to factor: ')
        if len(s) < 1:
            break

        number = int(s)

        print(f"Factoring number = {number}\n")

        input_size = int(math.ceil(math.log(number, 2)))

        # attempt to execute Shor's Algorithm to find factors
        failures = attempt_factoring(input_size, number, verbose)

        # Report how many factoring failures occurred
        if failures > 0:
            print(f"*** Factoring attempts failed {failures} times!")

    print("... exiting")


# if main, execute method
if __name__ == '__main__': run()  # max_qubits = 6, max_circuits = 5, num_shots=100)
