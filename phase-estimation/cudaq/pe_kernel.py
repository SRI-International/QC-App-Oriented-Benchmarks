'''
Phase Estimation Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import cudaq
    
# saved circuits for display
QC_ = None
Uf_ = None

############### Phase Esimation Circuit Definition

# Inverse Quantum Fourier Transform
@cudaq.kernel
def iqft(register: cudaq.qview):
    M_PI = 3.1415926536
    
    input_size = register.size()
     
    # use this as a barrier when drawing circuit; comment out otherwise
    for i in range(input_size / 2):
        swap(register[i], register[input_size - i - 1])
        swap(register[i], register[input_size - i - 1])
            
    # Generate multiple groups of diminishing angle CRZs and H gate
    for i_qubit in range(input_size):
        ri_qubit = input_size - i_qubit - 1             # map to cudaq qubits
        
        # precede with an H gate (applied to all qubits)
        h(register[ri_qubit])
        
        # number of controlled Z rotations to perform at this level
        num_crzs = input_size - i_qubit - 1
        
        # if not the highest order qubit, add multiple controlled RZs of decreasing angle
        if i_qubit < input_size - 1:   
            for j in range(0, num_crzs):
                #divisor = 2 ** (j + 1)     # DEVNOTE: #1663 This does not work on Quantum Cloud

                X = j + 1                   # WORKAROUND
                divisor = 1
                for i in range(X):
                    divisor *= 2
                
                r1.ctrl( -M_PI / divisor , register[ri_qubit], register[ri_qubit - j - 1])       
    
@cudaq.kernel           
#def pe_kernel (num_qubits: int, theta: float, do_uopt: bool):
def pe_kernel (num_qubits: int, theta: float, use_midcircuit_measurement: bool):
    M_PI = 3.1415926536
    
    init_phase = 2 * M_PI * theta
    do_uopt = True
    
    # size of input is one less than available qubits
    input_size = num_qubits - 1
        
    # Allocate on less than the specified number of qubits for phase register
    counting_qubits = cudaq.qvector(input_size)
    
    # Allocate an extra qubit for the state register (can expand this later)
    state_register = cudaq.qubit()

    # Prepare the auxillary qubit.
    x(state_register)

    # Place the phase register in a superposition state.
    h(counting_qubits)

    # Perform `ctrl-U^j`        (DEVNOTE: need to pass in as lambda later)
    for i in range(input_size):
        ii_qubit = input_size - i - 1
        i_qubit = i
        
        # optimize by collapsing all phase gates at one level to single gate
        if do_uopt:
            exp_phase = init_phase * (2 ** i_qubit) 
            r1.ctrl(exp_phase, counting_qubits[i_qubit], state_register);
        else:
            for j in range(2 ** i_qubit):
                r1.ctrl(init_phase, counting_qubits[i_qubit], state_register);
    
    # Apply inverse quantum Fourier transform
    iqft(counting_qubits)

    # Apply measurement gates to just the `qubits`
    # (excludes the auxillary qubit).
    mz(counting_qubits)
        
 
def PhaseEstimation (num_qubits: int, theta: float, use_midcircuit_measurement: bool):

    qc = [pe_kernel, [num_qubits, theta, use_midcircuit_measurement]]
    
    global QC_
    if num_qubits <= 6:
        QC_ = qc

    return qc

############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
    print("Sample Circuit:");
    if QC_ != None:
        print(cudaq.draw(QC_[0], *QC_[1]))
    else:
        print("  ... too large!")
    