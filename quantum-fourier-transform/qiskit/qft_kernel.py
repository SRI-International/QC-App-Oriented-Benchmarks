'''
Quantum Fourier Transformation Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

QC_ = None # Quantum Circuit
QFT_ = None # Quantum Fourier Transformation Circuit
QFTI_ = None # Quantum Inverse Fourier Transformation Circuit
QFTDI_ = None # Quantum Dynamic Inverse Fourier Transformation Circuit

############### Circuit Definition

def QuantumFourierTransform(num_qubits, secret_int,  bitset = None, method=1, use_midcircuit_measurement=False):
    global num_gates, depth
    num_gates = 0
    depth = 0
    # Size of input is one less than available qubits
    input_size = num_qubits
    
    # allocate qubits
    qr = QuantumRegister(num_qubits); cr = ClassicalRegister(num_qubits, name = "c0")
    qc = QuantumCircuit(qr, cr, name=f"qft({method})-{num_qubits}-{secret_int}")

    if method==1:

        # Perform X on each qubit that matches a bit in secret string
        s = ('{0:0'+str(input_size)+'b}').format(secret_int)
        for i_qubit in range(input_size):
            if s[input_size-1-i_qubit]=='1':
                qc.x(qr[i_qubit])
                num_gates += 1

        depth += 1

        qc.barrier()

        # perform QFT on the input
        qc.append(qft_gate(input_size).to_instruction(), qr)


        # End with Hadamard on all qubits (to measure the z rotations)
        ''' don't do this unless NOT doing the inverse afterwards
        for i_qubit in range(input_size):
             qc.h(qr[i_qubit])

        qc.barrier()
        '''

        qc.barrier()
        
        # some compilers recognize the QFT and IQFT in series and collapse them to identity;
        # perform a set of rotations to add one to the secret_int to avoid this collapse
        for i_q in range(0, num_qubits):
            divisor = 2 ** (i_q)
            qc.rz( 1 * math.pi / divisor , qr[i_q])
            num_gates+=1
        
        qc.barrier()

        # Uses dynamic inverse QFT if the flag is set; otherwise, use static version.
        # Dynamic circuits can only be added using "compose" because they are not unitary.
        # The "append" method requires the added circuit to be unitary, which dynamic circuits are not.
        if use_midcircuit_measurement:
            dynamic_inv_qft = dyn_inv_qft_gate(input_size)
            qc.compose(dynamic_inv_qft, qubits=qr, clbits = cr, inplace=True)
        else:
            qc.append(inv_qft_gate(input_size).to_instruction(), qr)
        
        qc.barrier()

    elif method == 2:

        for i_q in range(0, num_qubits):
            qc.h(qr[i_q])
            num_gates += 1

        for i_q in range(0, num_qubits):
            divisor = 2 ** (i_q)
            qc.rz(secret_int * math.pi / divisor, qr[i_q])
            num_gates += 1

        depth += 1

        if use_midcircuit_measurement:
            dynamic_inv_qft = dyn_inv_qft_gate(input_size)
            qc.compose(dynamic_inv_qft, qubits=qr, clbits = cr, inplace=True)
        else:
            qc.append(inv_qft_gate(input_size).to_instruction(), qr)

    # This method is a work in progress
    elif method==3:

        for i_q in range(0, secret_int):
            qc.h(qr[i_q])
            num_gates+=1

        for i_q in range(secret_int, num_qubits):
            qc.x(qr[i_q])
            num_gates+=1
            
        depth += 1
        
        if use_midcircuit_measurement:
            dynamic_inv_qft = dyn_inv_qft_gate(input_size)
            qc.compose(dynamic_inv_qft, qr[:], clbits = cr, inplace=True)
        else:
            qc.append(inv_qft_gate(input_size).to_instruction(), qr)
        
    else:
        exit("Invalid QFT method")

    # measure all qubits
    qc.measure(qr, cr)
    num_gates += num_qubits
    depth += 1

    # save smaller circuit example for display
    global QC_    
    if QC_ == None or num_qubits <= 5:
        if num_qubits < 9: QC_ = qc
    
    # collapse the sub-circuit levels used in this benchmark (for qiskit)
    qc2 = qc.decompose()
            
    # return a handle on the circuit
    return qc2

############### QFT Circuit

def qft_gate(input_size):
    global QFT_, num_gates, depth
    # avoid name "qft" as workaround of https://github.com/Qiskit/qiskit/issues/13174
    qr = QuantumRegister(input_size); qc = QuantumCircuit(qr, name="qft_")
    
    # Generate multiple groups of diminishing angle CRZs and H gate
    for i_qubit in range(0, input_size):
    
        # start laying out gates from highest order qubit (the hidx)
        hidx = input_size - i_qubit - 1
        
        # if not the highest order qubit, add multiple controlled RZs of decreasing angle
        if hidx < input_size - 1:   
            num_crzs = i_qubit
            for j in range(0, num_crzs):
                divisor = 2 ** (num_crzs - j)
                qc.crz( math.pi / divisor , qr[hidx], qr[input_size - j - 1])
                num_gates += 1
                depth += 1
            
        # followed by an H gate (applied to all qubits)
        qc.h(qr[hidx])
        num_gates += 1
        depth += 1
        
        qc.barrier()
    
    if QFT_ == None or input_size <= 5:
        if input_size < 9: QFT_ = qc
        
    return qc


############### Inverse QFT Circuit

def inv_qft_gate(input_size):
    global QFTI_, num_gates, depth
    qr = QuantumRegister(input_size); qc = QuantumCircuit(qr, name="inv_qft")
    
    # Generate multiple groups of diminishing angle CRZs and H gate
    for i_qubit in reversed(range(0, input_size)):
    
        # start laying out gates from highest order qubit (the hidx)
        hidx = input_size - i_qubit - 1
        
        # precede with an H gate (applied to all qubits)
        qc.h(qr[hidx])
        num_gates += 1
        depth += 1
        
        # if not the highest order qubit, add multiple controlled RZs of decreasing angle
        if hidx < input_size - 1:   
            num_crzs = i_qubit
            for j in reversed(range(0, num_crzs)):
                divisor = 2 ** (num_crzs - j)
                qc.crz( -math.pi / divisor , qr[hidx], qr[input_size - j - 1])
                
                num_gates += 1
                depth += 1
            
        qc.barrier()  
    
    if QFTI_ == None or input_size <= 5:
        if input_size < 9: QFTI_= qc
        
    return qc


def dyn_inv_qft_gate(input_size):
   
    global QFTDI_, num_gates, depth
    qr = QuantumRegister(input_size, name="q_dyn_inv"); cr = ClassicalRegister(input_size, name="c0")
    qc = QuantumCircuit(qr, cr, name="dyn_inv_qft")

    # mirror the static inv-QFT loop order, but with mid-circuit feed-forward
    for i_qubit in reversed(range(input_size)):
        hidx = input_size - 1 - i_qubit

        # H on the “hidx” wire
        qc.h(qr[hidx])
        num_gates += 1
        depth += 1
        qc.barrier()

        # measure for feed-forward
        qc.measure(qr[hidx], cr[hidx])
        depth += 1
        qc.barrier()

        # if measured == 1, apply RZ(-θ) on each target
        if hidx < input_size - 1:
            for j in reversed(range(i_qubit)):
                θ = math.pi / (2 ** (i_qubit - j))
                with qc.if_test((cr[hidx], 1)):
                    qc.rz(-θ, qr[input_size - 1 - j])
                    num_gates += 1
                    depth += 1
        qc.barrier()

    # cache a small-size example for printing
    if QFTDI_ is None and input_size <= 5:
        QFTDI_ = qc

    return qc

############### BV Circuit Drawer

# Draw the circuits of this benchmark program

def kernel_draw():
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    # Always show both QFT and inverse-QFT, regardless of 'method'
    print("\nQFT Circuit =");         print(QFT_  if QFT_  != None else "  ... too large!")
    if QFTI_ == None:
        print("\nDynamic Inverse QFT Circuit ="); print(QFTDI_ if QFTDI_ != None else "  ... too large!")
    else:
        print("\nInverse QFT Circuit ="); print(QFTI_ if QFTI_ != None else "  ... too large!")
    

    