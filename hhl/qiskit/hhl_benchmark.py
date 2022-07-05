"""
HHL Benchmark Program - Qiskit

TODO:
    - switch from total variation distance to Hellinger fidelity
"""

import sys
import time

import numpy as np
pi = np.pi

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute

import sparse_Ham_sim as shs
import uniform_controlled_rotation as ucr

# include QFT in this list, so we can refer to the QFT sub-circuit definition
#sys.path[1:1] = ["_common", "_common/qiskit", "quantum-fourier-transform/qiskit"]
#sys.path[1:1] = ["../../_common", "../../_common/qiskit", "../../quantum-fourier-transform/qiskit"]

# cannot use the QFT common yet, as HHL seems to use reverse bit order
sys.path[1:1] = ["_common", "_common/qiskit", "quantum-fourier-transform/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit", "../../quantum-fourier-transform/qiskit"]
#from qft_benchmark import qft_gate, inv_qft_gate

import execute as ex
import metrics as metrics

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
    qr = QuantumRegister(input_size);
    #qc = QuantumCircuit(qr, name="qft")
    qc = QuantumCircuit(qr, name="QFT†")
    
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
                    qc.cu1(-np.pi / divisor, qr[hidx], qr[input_size - j - 1]);
                
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
    qr = QuantumRegister(input_size);
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
                    qc.cu1( np.pi / divisor , qr[hidx], qr[input_size - j - 1])
    
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
        qc.u(np.pi/2, -np.pi/2, np.pi/2, 0);
    
    cu_gate = qc.to_gate().control(1)

    return cu_gate, qc

#Construct the U^-1 gates for reversing A
def ctrl_ui(exponent):

    qc = QuantumCircuit(1, name=f"U^-{exponent}")
    
    for i in range(exponent):
        #qc.u(np.pi/2, -np.pi/2, np.pi/2, 3*np.pi/4, target);
        #qc.cu(np.pi/2, -np.pi/2, np.pi/2, 3*np.pi/4, control, target);
        qc.u(np.pi/2, np.pi/2, -np.pi/2, 0);
    
    cu_gate = qc.to_gate().control(1)

    return cu_gate, qc



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
                qc.u1(3*np.pi/4, clock[j]);
                
                # apply the rest of U controlled by clock qubit
                #cp, _ = ctrl_u(repeat)
                cp, _ = ctrl_u(1)
                qc.append(cp, [clock[j], target])  
            
            repeat *= 2
            
            qc.barrier();
    
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
                qc.u1(-3*np.pi/4, clock[j]);
                
                # apply the rest of U controlled by clock qubit
                #cp, _ = ctrl_u(repeat)
                cp, _ = ctrl_ui(1)
                qc.append(cp, [clock[j], target])  
            
            repeat = int(repeat / 2)
            
            qc.barrier();
    
        #Define global U operator as the phase operator (for printing later)
        _, UI_ = ctrl_ui(1)
    
    if method == 2:
        
        for j in reversed(range(len(clock))):
            
            control = clock[j]
            phase = (2*np.pi)*2**j
            con_H_sim = shs.control_Ham_sim(A, phase)
            qubits = [control] + [q for q in target] + [q for q in extra_qubits] + [ancilla[0]]
            qc.append(con_H_sim, qubits)
    
    

def hhl_routine(qc, ancilla, clock, input_qubits, measurement, extra_qubits=None, A=None, method=1):
    
    qpe(qc, clock, input_qubits, extra_qubits, ancilla, A, method)
    qc.reset(ancilla)

    qc.barrier()
    
    if method == 1:
        # This section is to test and implement C = 1   
        # since we are not swapping after the QFT, reverse order of qubits from what is in papers
        qc.cry(np.pi, clock[1], ancilla)
        qc.cry(np.pi/3, clock[0], ancilla)
    
    # uniformly-controlled rotation
    elif method == 2:
        
        n_clock = clock.size
        C = 1/2**n_clock # constant in rotation (lower bound on eigenvalues A)
        
        # compute angles for inversion rotations
        alpha = [2*np.arcsin(C)]
        for x in range(1,2**n_clock):
            x_bin_rev = np.binary_repr(x, width=n_clock)[::-1]
            lam = int(x_bin_rev,2)/(2**n_clock)
            alpha.append(2*np.arcsin(C/lam))
        theta = ucr.alpha2theta(alpha)
        
        # do inversion step
        qc = ucr.uniformly_controlled_rot(qc, clock, ancilla, theta)
        
    qc.barrier()
    
    qc.measure(ancilla, measurement[0])
    qc.reset(ancilla)
    
    qc.barrier()
    inv_qpe(qc, clock, input_qubits, extra_qubits, ancilla, A, method)
    


def HHL(num_qubits, num_input_qubits, num_clock_qubits, beta, A=None, method=1):
    
    if method == 1:
    
        # Create the various registers needed
        clock = QuantumRegister(num_clock_qubits, name='clock')
        input_qubits = QuantumRegister(num_input_qubits, name='b')
        ancilla = QuantumRegister(1, name='ancilla')
        measurement = ClassicalRegister(2, name='c')
    
        # Create an empty circuit with the specified registers
        qc = QuantumCircuit(ancilla, clock, input_qubits, measurement)
    
        # size of input is one less than available qubits
        input_size = num_qubits - 1
            
        # State preparation. (various initial values, done with initialize method)
        # intial_state = [0,1]
        # intial_state = [1,0]
        # intial_state = [1/np.sqrt(2),1/np.sqrt(2)]
        # intial_state = [np.sqrt(0.9),np.sqrt(0.1)]
        ##intial_state = [np.sqrt(1 - beta), np.sqrt(beta)]
        ##qc.initialize(intial_state, 3)
        
        # use an RY rotation to initialize the input state between 0 and 1
        qc.ry(2 * np.arcsin(beta), input_qubits)
    
        # Put clock qubits into uniform superposition
        qc.h(clock)
    
        # Perform the HHL routine
        hhl_routine(qc, ancilla, clock, input_qubits, measurement)
    
        # Perform a Hadamard Transform on the clock qubits
        qc.h(clock)
    
        qc.barrier()
    
        # measure the input, which now contains the answer
        qc.measure(input_qubits, measurement[1])
    
    # sparse Hamiltonian simulation by quantum random walk
    if method == 2:
        
        qc = make_circuit(A, num_clock_qubits)

    
    # save smaller circuit example for display
    #global QC_, U_, UI_, QFT_, QFTI_
    #if QC_ == None or num_qubits <= 6:
    #    if num_qubits < 9: QC_ = qc
    
    #if U_ == None or num_qubits <= 6:    
    #    _, U_ = ctrl_u(1)
        #U_ = ctrl_u(np.pi/2, 2, 0, 1)
        
    #if UI_ == None or num_qubits <= 6:    
    #    _, UI_ = ctrl_ui(1)
        #UI_ = ctrl_ui(np.pi/2, 2, 0, 1)
        
    #if QFT_ == None or num_qubits <= 5:
    #    if num_qubits < 9: QFT_ = qft_gate(len(clock))
    #if QFTI_ == None or num_qubits <= 5:
    #    if num_qubits < 9: QFTI_ = inv_qft_gate(len(clock))

    # return a handle on the circuit
    return qc


def make_circuit(A, b, num_clock_qubits):
    """ circuit for HHL algo A|x>=|b>
    
        A : sparse Hermitian matrix
        b (int): between 0,...,2^n-1. Initial basis state |b>
    """
    
    # constant in inversion step, fixed for now
    #C = 0.25
    
    # read in number of qubits
    N = len(A)
    n = int(np.log2(N))
    n_t = num_clock_qubits # number of qubits in clock register
    
    #C = 1/2**n_t
    C = 1/4 # lower bound on eigenvalues of A
    
    # create quantum registers
    qr = QuantumRegister(n)
    qr_b = QuantumRegister(n) # ancillas for Hamiltonian simulation
    cr = ClassicalRegister(n)
    qr_t = QuantumRegister(n_t) # for phase estimation
    qr_a = QuantumRegister(1) # ancilla qubit
    cr_a = ClassicalRegister(1)
    
    # temporary measure phase estimation register
    #cr_t = ClassicalRegister(num_t_qubits)
    
    qc = QuantumCircuit(qr, qr_b, qr_t, qr_a, cr, cr_a)
    #qc = QuantumCircuit(qr, qr_b, qr_t, qr_a, cr_t)

    # initialize the |b> state
    qc = initialize_state(qc, qr, b)
    
    # Hadamard phase estimation register
    for q in range(n_t):
        qc.h(qr_t[q])
        
    # perform controlled e^(i*A*t)
    for q in range(n_t):
        control = qr_t[q]
        phase = -(2*pi)*2**q  
        qc = shs.control_Ham_sim(qc, A, phase, control, qr, qr_b, qr_a[0])
    
    # inverse QFT
    qc = IQFT(qc, qr_t)
    #qc = QFT(qc, qr_t)
    
    # reset ancilla
    qc.reset(qr_a[0])
    
    # measure phase register
    #qc.measure(qr_t[:], cr_t[:])
        
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
        
    # do inversion step and measure ancilla
    qc = ucr.uniformly_controlled_rot(qc, qr_t, qr_a, theta)
    qc.measure(qr_a[0], cr_a[0])
    qc.reset(qr_a[0])
    

    # QFT
    qc = QFT(qc, qr_t)
    
    # uncompute phase estimation
    # perform controlled e^(-i*A*t)
    for q in reversed(range(n_t)):
        control = qr_t[q]
        phase = (2*pi)*2**q  
        qc = shs.control_Ham_sim(qc, A, phase, control, qr, qr_b, qr_a[0])
    
    # Hadamard phase estimation register
    for q in range(n_t):
        qc.h(qr_t[q])
    
    # measure ancilla and main register
    qc.barrier()
    #qc.measure(qr_a[0], cr_a[0])
    qc.measure(qr[0:], cr[0:])
    
    return qc



def sim_circuit(qc, shots):
    
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=shots).result()
    outcomes = result.get_counts(qc)

    return outcomes


def postselect(outcomes, return_probs=True):
    
    mar_out = {}
    for b_str in outcomes:
        if b_str[0] == '1':
            counts = outcomes[b_str]
            mar_out[b_str[2:]] = counts
            
    # compute postselection rate
    ps_shots = sum(mar_out.values())
    shots = sum(outcomes.values())
    rate = ps_shots/shots
    
    # convert to probability distribution
    if return_probs == True:
        mar_out = {b_str:round(mar_out[b_str]/ps_shots, 4) for b_str in mar_out}
    
    
    return mar_out, rate

 
############### Result Data Analysis


saved_result = None


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


def TVD(distr1, distr2):  
    """ compute total variation distance between distr1 and distr2
        which are represented as dictionaries of bitstrings and probabilities
    """
    
    tvd = 0.0
    for out1 in distr1:
        if out1 in distr2:
            p1, p2 = distr1[out1], distr2[out1]
            tvd += np.abs(p1-p2)/2
        else:
            p1 = distr1[out1]
            tvd += p1/2
    
    for out2 in distr2:
        if out2 not in distr1:
            p2 = distr2[out2]
            tvd += p2/2
    
    return tvd


def analyze_and_print_result (qc, result, num_qubits, s_int, num_shots):
    
    global saved_result
    saved_result = result
    
    # obtain counts from the result object
    counts = result.get_counts(qc)
    
    # post-select counts where ancilla was measured as |1>
    post_counts, rate = postselect(counts)
    num_input_qubits = len(list(post_counts.keys())[0])
    
    if verbose: 
        print(f'Ratio of counts with ancilla measured |1> : {round(rate, 4)}')
    
    # compute true distribution from secret int
    off_diag_index = 0
    b = 0
    s_int_o = int(s_int)
    s_int_b = int(s_int)
    while (s_int_o % 2) == 0:
        s_int_o = int(s_int_o/2)
        off_diag_index += 1
    while (s_int_b % 3) == 0:
        s_int_b = int(s_int_b/3)
        b += 1
    
    # temporarily fix diag and off-diag matrix elements
    diag_el = 0.5
    off_diag_el = -0.25
    A = shs.generate_sparse_H(num_input_qubits, off_diag_index,
                              diag_el=diag_el, off_diag_el=off_diag_el)
    ideal_distr = true_distr(A, b)
    
    
    # compute total variation distance
    tvd = TVD(ideal_distr, post_counts)
    
    # use TVD as infidelity
    fidelity = 1 - tvd
    
    
    return post_counts, fidelity

################ Benchmark Loop

# Execute program with default parameters
def run (min_input_qubits=1, max_input_qubits=3, min_clock_qubits=2, max_clock_qubits=3,
         max_circuits=3, num_shots=100, 
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None):

    print("HHL Benchmark Program - Qiskit")

    # validate parameters (smallest circuit is 4 qubits, largest 6)
    #max_qubits = max(6, max_qubits)
    #min_qubits = min(max(4, min_qubits), max_qubits)
    #print(f"min, max qubits = {min_qubits} {max_qubits}")
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, s_int, num_shots):
     
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        #counts, fidelity = analyze_and_print_result(qc, result, num_qubits, ideal_distr)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
        metrics.store_metric(num_qubits, s_int, 'fidelity', fidelity)
    
    # Variable for new qubit group ordering if using mid_circuit measurements
    mid_circuit_qubit_group = []

    # If using mid_circuit measurements, set transform qubit group to true
    transform_qubit_group = False
    
    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options)

    # for noiseless simulation, set noise model to be None
    ex.set_noise_model(None)
    
    # temporarily fix diag and off-diag matrix elements
    diag_el = 0.5
    off_diag_el = -0.25
    
    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_input_qubits in range(min_input_qubits, max_input_qubits+1):
        N = 2**num_input_qubits # matrix size
        
        for num_clock_qubits in range(min_clock_qubits, max_clock_qubits+1):
            
            num_qubits = 2*num_input_qubits + num_clock_qubits + 1
        
    
            # determine number of circuits to execute for this group
            num_circuits = min(N*(N-1), max_circuits)
            
            print(f"************\nExecuting {num_circuits} circuits with (num_input_qubits, num_clock_qubits) = {(num_input_qubits, num_clock_qubits)}")
            
            # loop over randomly generated problem instances
            for i in range(num_circuits):
                b = np.random.choice(range(N))
                off_diag_index = np.random.choice(range(1,N))
                A = shs.generate_sparse_H(num_input_qubits, off_diag_index,
                                          diag_el=diag_el, off_diag_el=off_diag_el)
                
                # define secret_int
                s_int = (2**off_diag_index)*(3**b)
                
                
                # create the circuit for given qubit size and secret string, store time metric
                ts = time.time()
                qc = make_circuit(A, b, num_clock_qubits)
                metrics.store_metric(num_qubits, s_int, 'create_time', time.time()-ts)


                # submit circuit for execution on target (simulator, cloud simulator, or hardware)
                ex.submit_circuit(qc, num_qubits, s_int, shots=num_shots)
        
            # Wait for some active circuits to complete; report metrics when groups complete
            ex.throttle_execution(metrics.finalize_group)
        
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    # print a sample circuit
    #print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    #if method == 1: print("\nQuantum Oracle 'Uf' ="); print(Uf_ if Uf_ != None else " ... too large!")
    #print("\nU Circuit ="); print(U_ if U_ != None else "  ... too large!")
    #print("\nU^-1 Circuit ="); print(UI_ if UI_ != None else "  ... too large!")
    #print("\nQFT Circuit ="); print(QFT_ if QFT_ != None else "  ... too large!")
    #print("\nInverse QFT Circuit ="); print(QFTI_ if QFTI_ != None else "  ... too large!")

    # Plot metrics for all circuit sizes
    metrics.plot_metrics("Benchmark Results - HHL - Qiskit",
                         transform_qubit_group = transform_qubit_group, new_qubit_group = mid_circuit_qubit_group)

# if main, execute method
if __name__ == '__main__': run()
   
