"""
HHL Benchmark Program - Qiskit

NOTE: The benchmark-level code in this file will be migrated to the parent directory.
This file will eventually contain only the Qiskit-specific kernel code.
To run this benchmark, use the script in the parent directory:
    python hhl/hhl_benchmark.py
"""

import time

import numpy as np
pi = np.pi

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

import sparse_Ham_sim as shs
import uniform_controlled_rotation as ucr

# cannot use the QFT common yet, as HHL seems to use reverse bit order
# from quantum_fourier_transform.qiskit.qft_benchmark import qft_gate, inv_qft_gate

import execute as ex
from qedclib import metrics

# Benchmark Name
benchmark_name = "HHL"

np.random.seed(0)

verbose = False

# Variable for number of resets to perform after mid circuit measurements
num_resets = 1

# saved circuits for display
QC_ = None
U_ = None
UI_ = None
QFT_ = None
QFTI_ = None
HP_ = None
INVROT_ = None


############### Circuit Definitions 

''' replaced with code below ... 
def qft_dagger(qc, clock, n):      
    qc.h(clock[1]);
    for j in reversed(range(n)):
      for k in reversed(range(j+1,n)):
        qc.cu1(-np.pi/float(2**(k-j)), clock[k], clock[j]);
    qc.h(clock[0]);

def qft(qc, clock, n):
    qc.h(clock[0]);
    for j in reversed(range(n)):
      for k in reversed(range(j+1,n)):
        qc.cu1(np.pi/float(2**(k-j)), clock[k], clock[j]);
    qc.h(clock[1]);
'''

'''
DEVNOTE: the QFT and IQFT are defined here as they are in the QFT benchmark - almost;
Here, the sign of the angles is reversed and the QFT is actually used as the inverse QFT.
This is an inconsistency that needs to be resolved later. 
The QPE part of the algorithm should be using the inverse QFT, but the qubit order is not correct.
The QFT as defined in the QFT benchmark operates on qubits in the opposite order from the HHL pattern.
'''

def initialize_state(qc, qreg, b):
    """ b (int): initial basis state |b> """
    
    n = qreg.size
    b_bin = np.binary_repr(b, width=n)
    if verbose:
        print(f"... initializing |b> to {b}, binary repr = {b_bin}")
    
    for q in range(n):
        if b_bin[n-1-q] == '1':
            qc.x(qreg[q])
    
    return qc


def IQFT(qc, qreg):
    """ inverse QFT
        qc : QuantumCircuit
        qreg : QuantumRegister belonging to qc
        
        does not include SWAP at end of the circuit
    """
    
    n = int(qreg.size)
    
    for i in reversed(range(n)):
        for j in range(i+1,n): 
            phase = -pi/2**(j-i) 
            qc.cp(phase, qreg[i], qreg[j])
        qc.h(qreg[i])
    
    return qc


def QFT(qc, qreg):
    """   QFT
          qc : QuantumCircuit
        qreg : QuantumRegister belonging to qc
        
        does not include SWAP at end of circuit
    """
    
    n = int(qreg.size)
    
    for i in range(n):
        qc.h(qreg[i])
        for j in reversed(range(i+1,n)):
            phase = pi/2**(j-i)
            qc.cp(phase, qreg[i], qreg[j])
         
    return qc


def inv_qft_gate(input_size, method=1):
#def qft_gate(input_size):
    #global QFT_
    qr = QuantumRegister(input_size)
    #qc = QuantumCircuit(qr, name="qft")
    qc = QuantumCircuit(qr, name="IQFT")
    
    if method == 1:
    
        # Generate multiple groups of diminishing angle CRZs and H gate
        for i_qubit in range(0, input_size):
        
            # start laying out gates from highest order qubit (the hidx)
            hidx = input_size - i_qubit - 1
            
            # if not the highest order qubit, add multiple controlled RZs of decreasing angle
            if hidx < input_size - 1:   
                num_crzs = i_qubit
                for j in range(0, num_crzs):
                    divisor = 2 ** (num_crzs - j)
                    #qc.crz( math.pi / divisor , qr[hidx], qr[input_size - j - 1])
                    ##qc.crz( -np.pi / divisor , qr[hidx], qr[input_size - j - 1])
                    qc.cp(-np.pi / divisor, qr[hidx], qr[input_size - j - 1])
                
            # followed by an H gate (applied to all qubits)
            qc.h(qr[hidx])
    
    elif method == 2:
        # apply IQFT to register
        for i in range(input_size)[::-1]:
            for j in range(i+1,input_size):
                phase = -np.pi/2**(j-i)
                qc.cp(phase, qr[i], qr[j])
            qc.h(qr[i])    
        
    qc.barrier()
    
    return qc

############### Inverse QFT Circuit

def qft_gate(input_size, method=1):
#def inv_qft_gate(input_size):
    #global QFTI_
    qr = QuantumRegister(input_size)
    #qc = QuantumCircuit(qr, name="inv_qft")
    qc = QuantumCircuit(qr, name="QFT")
    
    if method == 1:
        # Generate multiple groups of diminishing angle CRZs and H gate
        for i_qubit in reversed(range(0, input_size)):
        
            # start laying out gates from highest order qubit (the hidx)
            hidx = input_size - i_qubit - 1
            
            # precede with an H gate (applied to all qubits)
            qc.h(qr[hidx])
            
            # if not the highest order qubit, add multiple controlled RZs of decreasing angle
            if hidx < input_size - 1:   
                num_crzs = i_qubit
                for j in reversed(range(0, num_crzs)):
                    divisor = 2 ** (num_crzs - j)
                    #qc.crz( -math.pi / divisor , qr[hidx], qr[input_size - j - 1])
                    ##qc.crz( np.pi / divisor , qr[hidx], qr[input_size - j - 1])
                    qc.cp( np.pi / divisor , qr[hidx], qr[input_size - j - 1])
    
    elif method == 2:
        # apply QFT to register
        for i in range(input_size):
            qc.h(qr[i])
            for j in range(i+1, input_size):
                phase = np.pi/2**(j-i)
                qc.cp(phase, qr[i], qr[j])
                
    qc.barrier()   

    return qc

 
############# Controlled U Gate

#Construct the U gates for A
def ctrl_u(exponent):

    qc = QuantumCircuit(1, name=f"U^{exponent}")
    
    for i in range(exponent):
        #qc.u(np.pi/2, -np.pi/2, np.pi/2, 3*np.pi/4, target);
        #qc.cu(np.pi/2, -np.pi/2, np.pi/2, 3*np.pi/4, control, target);
        qc.u(np.pi/2, -np.pi/2, np.pi/2, 0)
    
    cu_gate = qc.to_gate().control(1)

    return cu_gate, qc

#Construct the U^-1 gates for reversing A
def ctrl_ui(exponent):

    qc = QuantumCircuit(1, name=f"U^-{exponent}")
    
    for i in range(exponent):
        #qc.u(np.pi/2, -np.pi/2, np.pi/2, 3*np.pi/4, target);
        #qc.cu(np.pi/2, -np.pi/2, np.pi/2, 3*np.pi/4, control, target);
        qc.u(np.pi/2, np.pi/2, -np.pi/2, 0)
    
    cu_gate = qc.to_gate().control(1)

    return cu_gate, qc


####### DEVNOTE: The following functions (up until the make_circuit) are from the first inccarnation
#       of this benchmark and are not used here.  Should be removed, but kept here for reference for now


############# Quantum Phase Estimation
   
# DEVNOTE: The QPE and IQPE methods below mirror the mechanism in Hector_Wong
# Need to investigate whether the clock qubits are in the correct, as this implementation
# seems to require the QFT be implemented in reverse also.  TODO

# Append a series of Quantum Phase Estimation gates to the circuit   
def qpe(qc, clock, target, extra_qubits=None, ancilla=None, A=None, method=1):
    qc.barrier()

    ''' original code from Hector_Wong 
    # e^{i*A*t}
    #qc.cu(np.pi/2, -np.pi/2, np.pi/2, 3*np.pi/4, clock[0], target, label='U');
    
    # The CU gate is equivalent to a CU1 on the control bit followed by a CU3
    qc.u1(3*np.pi/4, clock[0]);
    qc.cu3(np.pi/2, -np.pi/2, np.pi/2, clock[0], target);
    
    # e^{i*A*t*2}
    #qc.cu(np.pi, np.pi, 0, 0, clock[1], target, label='U2');
    qc.cu3(np.pi, np.pi, 0, clock[1], target);
    
    qc.barrier();
    '''

    # apply series of controlled U operations to the state |1>
    # does nothing to state |0> 
    # DEVNOTE: have not found a way to create a controlled operation that contains a U gate 
    # with the global phase; instead do it piecemeal for now
    
    if method == 1:
    
        repeat = 1
        #for j in reversed(range(len(clock))):
        for j in (range(len(clock))):
    
            # create U with exponent of 1, but in a loop repeating N times
            for k in range(repeat):
            
                # this global phase is applied to clock qubit
                qc.u1(3*np.pi/4, clock[j])
                
                # apply the rest of U controlled by clock qubit
                #cp, _ = ctrl_u(repeat)
                cp, _ = ctrl_u(1)
                qc.append(cp, [clock[j], target])  
            
            repeat *= 2
            
            qc.barrier()
    
        #Define global U operator as the phase operator (for printing later)
        _, U_ = ctrl_u(1)
        
        
    if method == 2:
        
        for j in range(len(clock)):
            
            control = clock[j]
            phase = -(2*np.pi)*2**j
            con_H_sim = shs.control_Ham_sim(A, phase)
            qubits = [control] + [q for q in target] + [q for q in extra_qubits] + [ancilla[0]]
            qc.append(con_H_sim, qubits)
    
    # Perform an inverse QFT on the register holding the eigenvalues
    qc.append(inv_qft_gate(len(clock), method), clock)
            

# Append a series of Inverse Quantum Phase Estimation gates to the circuit    
def inv_qpe(qc, clock, target, extra_qubits=None, ancilla=None, A=None, method=1):
    
    # Perform a QFT on the register holding the eigenvalues
    qc.append(qft_gate(len(clock), method), clock)
    
    qc.barrier()
    
    if method == 1:
    
        ''' original code from Hector_Wong 
        # e^{i*A*t*2}
        #qc.cu(np.pi, np.pi, 0, 0, clock[1], target, label='U2');
        qc.cu3(np.pi, np.pi, 0, clock[1], target);
    
        # e^{i*A*t}
        #qc.cu(np.pi/2, np.pi/2, -np.pi/2, -3*np.pi/4, clock[0], target, label='U');
        # The CU gate is equivalent to a CU1 on the control bit followed by a CU3
        qc.u1(-3*np.pi/4, clock[0]);
        qc.cu3(np.pi/2, np.pi/2, -np.pi/2, clock[0], target);
    
        qc.barrier()
        '''
        
        # apply inverse series of controlled U operations to the state |1>
        # does nothing to state |0> 
        # DEVNOTE: have not found a way to create a controlled operation that contains a U gate 
        # with the global phase; instead do it piecemeal for now
        
        repeat = 2 ** (len(clock) - 1)
        for j in reversed(range(len(clock))):
        #for j in (range(len(clock))):
    
            # create U with exponent of 1, but in a loop repeating N times
            for k in range(repeat):
    
                # this global phase is applied to clock qubit
                qc.u1(-3*np.pi/4, clock[j])
                
                # apply the rest of U controlled by clock qubit
                #cp, _ = ctrl_u(repeat)
                cp, _ = ctrl_ui(1)
                qc.append(cp, [clock[j], target])  
            
            repeat = int(repeat / 2)
            
            qc.barrier()
    
        #Define global U operator as the phase operator (for printing later)
        _, UI_ = ctrl_ui(1)
    
    if method == 2:
        
        for j in reversed(range(len(clock))):
            
            control = clock[j]
            phase = (2*np.pi)*2**j
            con_H_sim = shs.control_Ham_sim(A, phase)
            qubits = [control] + [q for q in target] + [q for q in extra_qubits] + [ancilla[0]]
            qc.append(con_H_sim, qubits)
    

############### Make HHL Circuit

# Make the HHL circuit 
def make_circuit(A, b, num_clock_qubits):
    """ Generate top-level circuit for HHL algo A|x>=|b>
    
        A : sparse Hermitian matrix
        b (int): between 0,...,2^n-1. Initial basis state |b>
    """
    
    # save smaller circuit example for display
    global QC_, U_, UI_, QFT_, QFTI_, HP_, INVROT_

    # read in number of qubits
    N = len(A)
    n = int(np.log2(N))
    n_t = num_clock_qubits # number of qubits in clock register
    
    num_qubits = 2*n + n_t + 1
    
    # lower bound on eigenvalues of A. Fixed for now
    C = 1/4
    
    ''' Define sets of qubits for this algorithm '''
    
    # create 'input' quantum and classical measurement register
    qr = QuantumRegister(n, name='input')
    qr_b = QuantumRegister(n, name='in_anc') # ancillas for Hamiltonian simulation (?)
    cr = ClassicalRegister(n)
    
    # create 'clock' quantum register
    qr_t = QuantumRegister(n_t, name='clock') # for phase estimation
    
    # create 'ancilla' quantum and classical measurement register
    qr_a = QuantumRegister(1, name='ancilla') # ancilla qubit
    cr_a = ClassicalRegister(1)
    
    # create the top-level HHL circuit, with all the registers
    qc = QuantumCircuit(qr, qr_b, qr_t, qr_a, cr, cr_a, name=f"hhl-{num_qubits}-{b}")

    ''' Initialize the input and clock qubits '''
    
    # initialize the |b> state - the 'input'
    qc = initialize_state(qc, qr, b)
    
    #qc.barrier()

    # Hadamard the phase estimation register - the 'clock'
    for q in range(n_t):
        qc.h(qr_t[q])

    qc.barrier()
     
    ''' Perform Quantum Phase Estimation on input (b), clock, and ancilla '''
    
    # perform controlled e^(i*A*t)
    for q in range(n_t):
        control = qr_t[q]
        anc = qr_a[0]
        phase = -(2*pi)*2**q  
        qc_u = shs.control_Ham_sim(n, A, phase)
        if phase <= 0:
            qc_u.name = "e^{" + str(q) + "iAt}"
        else:
            qc_u.name = "e^{-" + str(q) + "iAt}"
        if U_ == None:
            U_ = qc_u
        qc.append(qc_u, qr[0:len(qr)] + qr_b[0:len(qr_b)] + [control] + [anc])

    qc.barrier()
    
    ''' Perform Inverse Quantum Fourier Transform on clock qubits '''
    
    #qc = IQFT(qc, qr_t)
    
    qc_qfti = inv_qft_gate(n_t, method=2)
    qc.append(qc_qfti, qr_t)

    if QFTI_ == None:
        QFTI_ = qc_qfti
    
    qc.barrier()
    
    ''' Perform inverse rotation with ancilla '''
    
    # reset ancilla
    qc.reset(qr_a[0])
    
    # compute angles for inversion rotations
    alpha = [2*np.arcsin(C)]
    for x in range(1,2**n_t):
        x_bin_rev = np.binary_repr(x, width=n_t)[::-1]
        lam = int(x_bin_rev,2)/(2**n_t)
        if lam < C:
            alpha.append(0)
        elif lam >= C:
            alpha.append(2*np.arcsin(C/lam))
    theta = ucr.alpha2theta(alpha)
        
    # do inversion step

    qc_invrot = ucr.uniformly_controlled_rot(n_t, theta)
    qc.append(qc_invrot, qr_t[0:len(qr_t)] + [qr_a[0]])
    
    if INVROT_ == None:
        INVROT_ = qc_invrot
    
    # and measure ancilla
    
    qc.measure(qr_a[0], cr_a[0])
    qc.reset(qr_a[0])

    qc.barrier()
    
    ''' Perform Quantum Fourier Transform on clock qubits '''
   
    #qc = QFT(qc, qr_t)
    
    qc_qft = qft_gate(n_t, method=2)
    qc.append(qc_qft, qr_t)

    if QFT_ == None:
        QFT_ = qc_qft
    
    qc.barrier()
    
    ''' Perform Inverse Quantum Phase Estimation on input (b), clock, and ancilla '''
    
    # uncompute phase estimation
    # perform controlled e^(-i*A*t)
    for q in reversed(range(n_t)):
        control = qr_t[q]
        phase = (2*pi)*2**q  
        qc_ui = shs.control_Ham_sim(n, A, phase)
        if phase <= 0:
            qc_ui.name = "e^{" + str(q) + "iAt}"
        else:
            qc_ui.name = "e^{-" + str(q) + "iAt}"
        if UI_ == None:
            UI_ = qc_ui
        qc.append(qc_ui, qr[0:len(qr)] + qr_b[0:len(qr_b)] + [control] + [anc])

    qc.barrier()
    
    # Hadamard (again) the phase estimation register - the 'clock'
    for q in range(n_t):
        qc.h(qr_t[q])
    
    qc.barrier()
    
    ''' Perform final measurements '''
    
    # measure ancilla and main register
    qc.measure(qr[0:], cr[0:])

    if QC_ == None:
        QC_ = qc
        #print(f"... made circuit = \n{QC_}")

    return qc

 
############### Result Data Analysis

saved_result = None

# Compute the expected distribution, given the matrix A and input value b
def true_distr(A, b=0):
    
    N = len(A)
    n = int(np.log2(N))
    b_vec = np.zeros(N); b_vec[b] = 1.0
    #b = np.array([1,1])/np.sqrt(2)
    
    x = np.linalg.inv(A) @ b_vec
    # normalize x
    x_n = x/np.linalg.norm(x)
    probs = np.array([np.abs(xj)**2 for xj in x_n])
    
    distr = {}
    for j, prob in enumerate(probs):
        if prob > 1e-8:
            j_bin = np.binary_repr(j, width=n)
            distr[j_bin] = prob
    
    distr = {out:distr[out]/sum(distr.values()) for out in distr}
    
    return distr

 # post-select counts where ancilla was measured as |1>
def postselect(outcomes, return_probs=True):
    
    mar_out = {}
    for b_str, counts in outcomes.items():
        # SamplerV2 result does not include white spaces between classical registers
        # E.g., backend.run: "0 0", SamplerV2: "00"
        b_str = b_str.replace(" ", "")
        if b_str[0] == '1':
            mar_out[b_str[1:]] = counts
            
    # compute postselection rate
    ps_shots = sum(mar_out.values())
    shots = sum(outcomes.values())
    rate = ps_shots/shots
    
    # convert to probability distribution
    if return_probs == True:
        mar_out = {b_str:round(mar_out[b_str]/ps_shots, 4) for b_str in mar_out}  
    
    return mar_out, rate
    
# Analyze the quality of the result obtained from executing circuit qc 
def analyze_and_print_result (qc, result, num_qubits, num_shots, s_int=None):

    global saved_result
    saved_result = result
    
    # obtain counts from the result object
    counts = result.get_counts(qc)

    if verbose:
        print(f"... for circuit = {num_qubits} {s_int}, counts = {counts}")
    
    # post-select counts where ancilla was measured as |1>
    post_counts, rate = postselect(counts)
    num_input_qubits = len(list(post_counts.keys())[0])
    
    if verbose: 
        print(f'... ratio of counts with ancilla measured |1> : {round(rate, 4)}')
    
    # compute true distribution from secret int
    off_diag_index = 0
    b = 0
    
    # remove instance index from s_int
    s_int = s_int - 1000 * int(s_int/1000)
    
    # get off_diag_index and b
    s_int_o = int(s_int)
    s_int_b = int(s_int)   
    
    while (s_int_o % 2) == 0:
        s_int_o = int(s_int_o/2)
        off_diag_index += 1
        
    while (s_int_b % 3) == 0:
        s_int_b = int(s_int_b/3)
        b += 1
    
    if verbose:
        print(f"... rem(s_int) = {s_int}, b = {b}, odi = {off_diag_index}")
        
    # temporarily fix diag and off-diag matrix elements
    diag_el = 0.5
    off_diag_el = -0.25
    A = shs.generate_sparse_H(num_input_qubits, off_diag_index,
                              diag_el=diag_el, off_diag_el=off_diag_el)
    ideal_distr = true_distr(A, b)
      
    # # compute total variation distance
    # tvd = TVD(ideal_distr, post_counts)
    
    # # use TVD as infidelity
    # fidelity = 1 - tvd
    # #fidelity = metrics.polarization_fidelity(post_counts, ideal_distr)

    fidelity = metrics.polarization_fidelity(post_counts, ideal_distr)
    
    return post_counts, fidelity


############### Get Circuits

import inspect

def get_circuits(
    # Standard args (common across benchmarks)
    min_qubits=4, max_qubits=6, skip_qubits=1,
    max_circuits=3, num_shots=100, method=1,
    # App-specific args
    use_best_widths=None, min_register_qubits=1,
    # Explicit input/clock qubit ranges (override min/max_qubits if provided)
    min_input_qubits=None, max_input_qubits=None,
    min_clock_qubits=None, max_clock_qubits=None,
    api=None,
):
    """Create HHL benchmark circuits over a range of input and clock qubit widths.

    Standard args (common to all benchmarks):
        min_qubits: smallest total circuit width (default 4)
        max_qubits: largest total circuit width (default 6)
        skip_qubits: increment between widths (default 1)
        max_circuits: max circuits per qubit group (default 3)
        num_shots: measurement shots, stored in metrics (default 100)
        method: algorithm method (default 1)

    App-specific args:
        use_best_widths: use optimal input/clock split for each total width (default True)
        min_register_qubits: skip widths where input or clock < this (default 1)
        min/max_input_qubits: explicit input qubit range; overrides min/max_qubits (default None)
        min/max_clock_qubits: explicit clock qubit range; overrides min/max_qubits (default None)
        api: programming API; None = use qedc_set_api() value (default None)

    Returns (all_qcs, circuit_metrics) — nested circuit dict and creation metrics.
    """

    # If explicit input/clock ranges not given, compute from min/max_qubits
    # using formula: num_qubits = 2*input + clock + 1 (ancilla), input ~= (N-1)/3
    if min_input_qubits is None:
        # Compute input/clock ranges from total qubit range
        max_qubits = max(4, max_qubits)
        min_qubits = min(max(4, min_qubits), max_qubits)

        min_input_qubits = int((min_qubits - 1) / 3)
        max_input_qubits = int((max_qubits - 1) / 3)
        min_clock_qubits = min_qubits - 1 - 2 * min_input_qubits
        max_clock_qubits = max_qubits - 1 - 2 * max_input_qubits

        # When computing ranges, default to filtering optimal widths only
        if use_best_widths is None:
            use_best_widths = True
    else:
        # When explicit ranges provided, default to showing all combinations
        if use_best_widths is None:
            use_best_widths = False

    # validate
    min_input_qubits = min(max(1, min_input_qubits), max_input_qubits)
    max_input_qubits = max(min_input_qubits, max_input_qubits)
    min_clock_qubits = min(max(1, min_clock_qubits), max_clock_qubits)
    max_clock_qubits = max(min_clock_qubits, max_clock_qubits)
    skip_qubits = max(1, skip_qubits)

    # initialize saved circuits for display
    global QC_, U_, UI_, QFT_, QFTI_, HP_, INVROT_
    QC_ = None; U_ = None; UI_ = None; QFT_ = None
    QFTI_ = None; HP_ = None; INVROT_ = None

    metrics.init_metrics()

    # temporarily fix diag and off-diag matrix elements
    diag_el = 0.5
    off_diag_el = -0.25

    # Build circuits over input x clock qubit range
    all_qcs = {}
    for num_input_qubits in range(min_input_qubits, max_input_qubits + 1, skip_qubits):
        N = 2**num_input_qubits

        for num_clock_qubits in range(min_clock_qubits, max_clock_qubits + 1, skip_qubits):
            num_qubits = 2 * num_input_qubits + num_clock_qubits + 1
            num_circuits = max_circuits

            # if flagged, skip non-optimal input/clock combinations
            if use_best_widths:
                if num_input_qubits != int((num_qubits - 1) / 3) or \
                   num_clock_qubits != (num_qubits - 1 - 2 * num_input_qubits):
                    if verbose:
                        print(f"... SKIPPING {num_circuits} circuits with {num_qubits} qubits, "
                              f"using {num_input_qubits} input and {num_clock_qubits} clock qubits")
                    continue

            # skip if input or clock size smaller than minimum
            if min_register_qubits > 1 and (num_input_qubits < min_register_qubits or
                                            num_clock_qubits < min_register_qubits):
                if verbose:
                    print(f"... SKIPPING circuits with {num_input_qubits} input "
                          f"and {num_clock_qubits} clock qubits")
                continue

            print(f"************\nCreating {num_circuits} circuits with {num_qubits} qubits, "
                  f"using {num_input_qubits} input and {num_clock_qubits} clock qubits")
            all_qcs[str(num_qubits)] = {}

            # Create circuits with randomly generated problem instances
            for i in range(num_circuits):
                b = np.random.choice(range(1, N))
                off_diag_index = np.random.choice(range(1, N))

                # encode instance index, b, and off_diag_index into secret_int
                s_int = 1000 * (i+1) + (2**off_diag_index) * (3**b)
                circuit_id = s_int

                if verbose:
                    print(f"... create A for b = {b}, off_diag_index = {off_diag_index}, s_int = {s_int}")

                A = shs.generate_sparse_H(num_input_qubits, off_diag_index,
                                          diag_el=diag_el, off_diag_el=off_diag_el)

                ts = time.time()
                qc = make_circuit(A, b, num_clock_qubits)
                metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time()-ts)

                # collapse the sub-circuits used in this benchmark (for qiskit)
                qc2 = qc.decompose()

                all_qcs[str(num_qubits)][str(circuit_id)] = qc2

    return all_qcs, metrics.circuit_metrics


############### Run Circuits

def run_circuits(all_qcs,
    num_shots=100, method=1, max_batch_size=None,
    backend_id=None, provider_backend=None,
    hub="ibm-q", group="open", project="main",
    exec_options=None, context=None, api=None,
):
    """Execute benchmark circuits and collect metrics.

    Args:
        all_qcs: circuit dict from get_circuits()
        num_shots: measurement shots per circuit (default 100)
        method: algorithm method, for plot options (default 1)
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
        context = f"{benchmark_name} Benchmark"

    # Result handler: called for each circuit after execution completes
    def execution_handler(qc, result, num_qubits, circuit_id, num_shots):
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, num_shots,
                s_int=int(circuit_id))
        metrics.store_metric(num_qubits, circuit_id, 'fidelity', fidelity)

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
    method=1, num_shots=100, max_circuits=3,
    api=None, draw_circuits=True, plot_results=True,
):
    """Draw sample circuits and plot benchmark metrics.

    Args:
        method: algorithm method, for plot options (default 1)
        num_shots: shots, for plot subtitle (default 100)
        max_circuits: circuit reps, for plot subtitle (default 3)
        api: programming API name for plot title (default None)
        draw_circuits: draw sample circuit diagrams (default True)
        plot_results: generate metrics plots (default True)
    """
    if draw_circuits:
        print("Sample Circuit:"); print(QC_ if QC_ is not None else "  ... too large!")
        print("\nU Circuit ="); print(U_ if U_ is not None else "  ... too large!")
        print("\nU^-1 Circuit ="); print(UI_ if UI_ is not None else "  ... too large!")
        print("\nQFT Circuit ="); print(QFT_ if QFT_ is not None else "  ... too large!")
        print("\nInverse QFT Circuit ="); print(QFTI_ if QFTI_ is not None else "  ... too large!")
        print("\nHamiltonian Phase Estimation Circuit ="); print(HP_ if HP_ is not None else "  ... too large!")
        print("\nControlled Rotation Circuit ="); print(INVROT_ if INVROT_ is not None else "  ... too large!")

    if plot_results:
        options = {"method": method, "shots": num_shots, "reps": max_circuits}
        metrics.plot_metrics(f"Benchmark Results - {benchmark_name} - Qiskit", options=options)


############### Run (convenience)

def run(**kwargs):
    """Create circuits, execute, and plot. Accepts any arg from
    get_circuits(), run_circuits(), or plot_results().

    For explicit input/clock qubit ranges, pass min_input_qubits,
    max_input_qubits, min_clock_qubits, max_clock_qubits instead
    of min_qubits/max_qubits."""

    def _for(func):
        return {k: kwargs[k] for k in kwargs if k in inspect.signature(func).parameters}

    get_circuits_only = kwargs.pop('get_circuits', False)

    print(f"{benchmark_name} Benchmark Program - Qiskit")

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

# Backward-compatible alias for notebooks that call run2() with explicit qubit ranges
run2 = run


if __name__ == '__main__':
    print("Please run this benchmark from the parent directory:")
    print("  python hhl/hhl_benchmark.py")
