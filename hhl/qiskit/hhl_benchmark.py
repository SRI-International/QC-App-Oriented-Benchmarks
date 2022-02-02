"""
HHL Benchmark Program - Qiskit

Issues:
    - The QPE U gates are not implemented as exponentiated sub-circuit
        since the current hard-coded U uses a CU gate which takes a global phase parameter.
        Have not found a way to create a controlled version of a subcircuit that contains a U gate
        with 4 parameters, like the CU gate.  The U gate only has 3, theta, phi, lambda. CU adds gamma.
    - The clock qubits may be implemented in reverse of how QPE is normally done. The QPE and QFT code 
        is written to assume a reverse order for the clock qubits.  Needs investigation to be like other BMs.
TODO:
    - Implement a way to create U from A
    - Find proper way to calculate fidelity; currently compares number of correct answers, not expectation
    - Find way to make input larger and initialize properly for larger input vectors; currently increase 
         in num_qubits increases the clock qubits; should make input size larger too
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


def IQFT(qc, qreg):
    """ inverse QFT
          qc : QuantumCircuit
        qreg : QuantumRegister belonging to qc
    """
    
    n = int(qreg.size)
    
    # apply IQFT to register
    for i in range(n)[::-1]:
        yi = qreg[i]
        for j in range(i+1,n):
            yj = qreg[j]
            phase = -pi/2**(j-i)
            qc.cp(phase, yi, yj)
        qc.h(yi)
    
    return qc


def QFT(qc, qreg):
    """   QFT
          qc : QuantumCircuit
        qreg : QuantumRegister belonging to qc
    """
    
    n = int(qreg.size)
    
    # apply QFT to register
    for i in range(n):
        yi = qreg[i]
        qc.h(yi)
        for j in range(i+1,n):
            yj = qreg[j]
            phase = pi/2**(j-i)
            qc.cp(phase, yi, yj)
        
    
    return qc


def initialize_state(qc, qreg):
    """ currently prepares |0...0> """
    
    #qc.h(qreg[0])
    
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


def make_circuit(A, num_clock_qubits):
    """ circuit for HHL algo A|x>=|b> """
    
    # constant in inversion step, fixed for now
    #C = 0.25
    
    # read in number of qubits
    N = len(A)
    n = int(np.log2(N))
    n_t = int(num_clock_qubits) # number of qubits in clock register
    
    #C = min(np.linalg.eigh(A)[0])
    C = 1/2**n_t
    
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
    qc = initialize_state(qc, qr)
    
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
    
    # reset ancilla
    qc.reset(qr_a[0])
    
    # measure phase register
    #qc.measure(qr_t[:], cr_t[:])
        
    # compute angles for inversion rotations
    alpha = [2*np.arcsin(C)]
    for x in range(1,2**n_t):
        x_bin_rev = np.binary_repr(x, width=n_t)[::-1]
        lam = int(x_bin_rev,2)/(2**n_t)
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
    for j in range(n_t):
        q = n_t - 1 - j
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


def sim_circuit(qc, shots, post_select=True, return_probs=True):
    
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=shots).result()
    outcomes = result.get_counts(qc)
    
    # post select
    if post_select == True:
        
        mar_out = {}
        for b_str in outcomes:
            if b_str[0] == '1':
                counts = outcomes[b_str]
                mar_out[b_str[2:]] = counts
                
        outcomes = dict(mar_out)
    
    # convert to probability distribution
    if return_probs == True:
        shots = sum(outcomes.values())
        outcomes = {b_str:round(outcomes[b_str]/shots, 4) for b_str in outcomes}

    return outcomes

 
############### Result Data Analysis


saved_result = None

# Analyze and print measured results
# Expected result is always the secret_int, so fidelity calc is simple

# NOTE: for the hard-coded matrix A:  [ 1, -1/3, -1/3, 1 ]
#       x - y/3 = 1 - beta
#       -x/3 + y = beta
#  ==
#       x = 9/8 - 3*beta/4
#       y = 3/8 + 3*beta/4
#
#   and beta is stored as secret_int / 10000
#   This allows us to calculate the expected distribution
#
#   NOTE: we are not actually calculating the distribution, since it would have to include the ancilla
#   For now, we just return a distribution of only the 01 and 11 counts
#   Then we compare the ratios obtained with expected ratio to determine fidelity (incorrectly)

def compute_expectation(A, b):
    """
    A (np.array) : NxN matrix in problem instance
    b (np.array) : dim N vector in problem instance
    """
    
    n = int(np.log2(len(A)))
    
    # hard-code A for now
    #A = np.array([[0.75, -0.25],[-0.25, 0.75]])
    
    # initial state |b>
    #b = np.array([np.sqrt(1-beta**2),beta])
    
    # solution vector satisfying Ax=b
    x = np.linalg.inv(A) @ b
    # normalize x
    x = x/np.linalg.norm(x)
    
    #x, y = v[0], v[1]
    #ratio = x / y
    #ratio_sq = ratio * ratio
    #print(f"  ... x,y = {x, y} ratio={ratio} ratio_sq={ratio_sq}")
    
    #iy = int(num_shots / (1 + ratio_sq))
    #ix = num_shots - iy
    #print(f"    ... ix,iy = {ix, iy}")
    
    # compute true distribution
    true_dist = {}
    for j, xj in enumerate(x):
        j_bin = np.binary_repr(j, width=n)
        true_dist[j_bin] = np.abs(xj)**2

    return true_dist


def analyze_and_print_result (qc, result, num_qubits, secret_int, num_shots):
    global saved_result
    saved_result = result
    
    # obtain counts from the result object
    counts = result.get_counts(qc)
    # post-select counts where ancilla was measured as |1>
    post_counts = {}
    for b_str in counts:
        if b_str[-1] == '1':
            post_counts[b_str[:-1]] = counts[b_str]
    if verbose: 
        #print(f"For secret int {secret_int} measured: {counts}")
        anc_succ_ratio = sum(post_counts.values())/sum(counts.values())
        print(f'Ratio of counts with ancilla measured |1> : {round(anc_succ_ratio, 4)}')
    
    # compute beta from secret_int, and get expected distribution
    # compute ratio of 01 to 11 measurements for both expected and obtained
    #beta = secret_int / 10000
    A = np.array([[0.75, -0.25],[-0.25, 0.75]])
    b = np.array([1,0])
    
    shots = sum(post_counts.values())
    true_dist = compute_expectation(A, b)
    #expected_dist = {b_str:int(shots*true_dist[b_str]) for b_str in true_dist}
    # compute Total Variation Distance
    tvd = 0.0
    for b_str in true_dist:
        p0 = true_dist[b_str]
        if b_str in post_counts:
            p1 = post_counts[b_str]/shots
        else:
            p1 = 0.0
        tvd += abs(p0-p1)/2
    
    # use TVD as infidelity
    fidelity = 1 - tvd
            
    
    
    #print(f"... expected = {expected_dist}")
    
    #try:
    #    ratio_exp = expected_dist['01'] / expected_dist['11']
    #    ratio_counts = counts['01'] / counts['11']
    #except Exception as e:
    #    ratio_exp = 0
    #    ratio_counts = 0
    
    #print(f"  ... ratio_exp={ratio_exp}  ratio_counts={ratio_counts}")
    
    # (NOTE: we should use this fidelity calculation, but cannot since we don't know actual expected)
    # use our polarization fidelity rescaling
    #fidelity = metrics.polarization_fidelity(post_counts, expected_dist)
    
    # instead, approximate fidelity by comparing ratios
    #if ratio_exp == 0 or ratio_counts == 0:
    #    fidelity = 0
    #elif ratio_exp > ratio_counts:
    #    fidelity = ratio_counts / ratio_exp
    #else:
    #    fidelity = ratio_exp / ratio_counts
    #if verbose: print(f"  ... fidelity = {fidelity}")
    
    
    return post_counts, fidelity

################ Benchmark Loop

# Execute program with default parameters
def run (min_qubits=5, max_qubits=6, max_circuits=3, num_shots=100,
        method = 2, 
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None):

    print("HHL Benchmark Program - Qiskit")
    
    # hard code A for now
    A = np.array([[0.75,-0.25],[-0.25,0.75]])

    # validate parameters (smallest circuit is 4 qubits, largest 6)
    #max_qubits = max(6, max_qubits)
    #min_qubits = min(max(4, min_qubits), max_qubits)
    #print(f"min, max qubits = {min_qubits} {max_qubits}")

    # Variable for new qubit group ordering if using mid_circuit measurements
    mid_circuit_qubit_group = []

    # If using mid_circuit measurements, set transform qubit group to true
    #transform_qubit_group = True if method == 2 else False
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler (qc, result, num_qubits, s_int, num_shots):  
     
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
        metrics.store_metric(num_qubits, s_int, 'fidelity', fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options)

    # for noiseless simulation, set noise model to be None
    # ex.set_noise_model(None)

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1):
        
        # based on num_qubits, determine input and clock sizes
        num_input_qubits = 1
        if method == 1:
            num_clock_qubits = num_qubits - num_input_qubits - 1  # need 1 for ancilla also
        if method == 2:
            num_clock_qubits = num_qubits - 2*num_input_qubits - 1 # need 1 for ancilla also
    
        # determine number of circuits to execute for this group
        num_circuits = min(2**(num_qubits), max_circuits)
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        
        ''' 
        #beta range not calculated dynamically yet
        num_counting_qubits = 1
        
        # determine range of secret strings to loop over
        if 2**(num_counting_qubits) <= max_circuits:
            beta_range = [i/(2**(num_counting_qubits)) for i in list(range(num_circuits))]
        else:
            beta_range = [i/(2**(num_counting_qubits)) for i in np.random.choice(2**(num_counting_qubits), num_circuits, False)]
        '''
        
        # supply hard-coded beta array (during initial testing)
        beta_range = [0.0]
        if max_circuits < len(beta_range): beta_range = beta_range[:max_circuits]   
        
        # loop over limited # of inputs for this
        for i in range(len(beta_range)):
            beta = beta_range[i]
            
            # create integer that represents beta to precision 4; use s_int as circuit id
            s_int = int(beta * 10000)
            print(f"  ... i={i} s_int={s_int}  beta={beta}")
            
            # create the circuit for given qubit size and secret string, store time metric
            ts = time.time()
            qc = HHL(num_qubits, num_input_qubits, num_clock_qubits, beta, A=A, method=method)
            metrics.store_metric(num_qubits, s_int, 'create_time', time.time()-ts)

            # collapse the sub-circuit levels used in this benchmark (for qiskit)
            qc2 = qc.decompose().decompose()

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc2, num_qubits, s_int, shots=num_shots)
        
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
    #metrics.plot_metrics(f"Benchmark Results - HHL ({method}) - Qiskit",
                         #transform_qubit_group = transform_qubit_group, new_qubit_group = mid_circuit_qubit_group)

# if main, execute method
#if __name__ == '__main__': run()
   
