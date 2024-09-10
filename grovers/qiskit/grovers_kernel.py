'''
Grover's Search Benchmark Program - Qiskit Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# saved circuits for display
QC_ = None
grover_oracle = None
diffusion_operator = None
 
# for validating the implementation of an mcx shim  
_use_mcx_shim = False 

############### Circuit Definition

def GroversSearch(num_qubits, marked_item, n_iterations, use_mcx_shim):

    global _use_mcx_shim
    _use_mcx_shim = use_mcx_shim
        
    # allocate qubits
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(qr, cr, name=f"grovers-{num_qubits}-{marked_item}")

    # Start with Hadamard on all qubits
    for i_qubit in range(num_qubits):
        qc.h(qr[i_qubit])

    # loop over the estimated number of iterations
    for _ in range(n_iterations):

        qc.barrier()
    
        # add the grover oracle
        qc.append(add_grover_oracle(num_qubits, marked_item).to_instruction(), qr)
        
        # add the diffusion operator
        qc.append(add_diffusion_operator(num_qubits).to_instruction(), qr)

    qc.barrier()
        
    # measure all qubits
    qc.measure(qr, cr)

    # save smaller circuit example for display
    global QC_    
    if QC_ == None or num_qubits <= 5:
        if num_qubits < 9: QC_ = qc
        
    # collapse the sub-circuit levels used in this benchmark (for qiskit)
    qc2 = qc.decompose()
            
    # return a handle on the circuit
    return qc2

############## Grover Oracle

def add_grover_oracle(num_qubits, marked_item):
    global grover_oracle
    
    marked_item_bits = format(marked_item, f"0{num_qubits}b")[::-1]

    qr = QuantumRegister(num_qubits); qc = QuantumCircuit(qr, name="oracle")

    for (q, bit) in enumerate(marked_item_bits):
        if not int(bit):
            qc.x(q)

    qc.h(num_qubits - 1)
    
    if _use_mcx_shim:
        add_mcx(qc, [x for x in range(num_qubits - 1)], num_qubits - 1)
    else:
        qc.mcx([x for x in range(num_qubits - 1)], num_qubits - 1)
        
    qc.h(num_qubits - 1)

    qc.barrier()

    for (q, bit) in enumerate(marked_item_bits):
        if not int(bit):
            qc.x(q)

    if grover_oracle == None or num_qubits <= 5:
        if num_qubits < 9: grover_oracle = qc
        
    return qc

############## Grover Diffusion Operator

def add_diffusion_operator(num_qubits):
    global diffusion_operator

    qr = QuantumRegister(num_qubits); qc = QuantumCircuit(qr, name="diffuser")

    for i_qubit in range(num_qubits):
        qc.h(qr[i_qubit])
    for i_qubit in range(num_qubits):
        qc.x(qr[i_qubit])
    qc.h(num_qubits - 1)
    
    if _use_mcx_shim:
        add_mcx(qc, [x for x in range(num_qubits - 1)], num_qubits - 1)
    else:
        qc.mcx([x for x in range(num_qubits - 1)], num_qubits - 1)
        
    qc.h(num_qubits - 1)

    qc.barrier()

    for i_qubit in range(num_qubits):
        qc.x(qr[i_qubit])
    for i_qubit in range(num_qubits):
        qc.h(qr[i_qubit])
        
    if diffusion_operator == None or num_qubits <= 5:
        if num_qubits < 9: diffusion_operator = qc
        
    return qc

############### MCX shim

# single cx / cp unit for mcx implementation
def add_cx_unit(qc, cxcu1_unit, controls, target):
    num_controls = len(controls)
    i_qubit = cxcu1_unit[1]
    j_qubit = cxcu1_unit[0]
    theta = cxcu1_unit[2]
    
    if j_qubit != None:
        qc.cx(controls[j_qubit], controls[i_qubit]) 
    qc.cp(theta, controls[i_qubit], target)
        
    i_qubit = i_qubit - 1
    if j_qubit == None:
        j_qubit = i_qubit + 1
    else:
        j_qubit = j_qubit - 1
        
    if theta < 0:
        theta = -theta
    
    new_units = []
    if i_qubit >= 0:
        new_units += [ [ j_qubit, i_qubit, -theta ] ]
        new_units += [ [ num_controls - 1, i_qubit, theta ] ]
        
    return new_units

# mcx recursion loop 
def add_cxcu1_units(qc, cxcu1_units, controls, target):
    new_units = []
    for cxcu1_unit in cxcu1_units:
        new_units += add_cx_unit(qc, cxcu1_unit, controls, target)
    cxcu1_units.clear()
    return new_units

# mcx gate implementation: brute force and inefficent
# start with a single CP on last control and target
# and recursively expand for each additional control
def add_mcx(qc, controls, target):
    num_controls = len(controls)
    theta = math.pi / 2**num_controls
    qc.h(target)
    cxcu1_units = [ [ None, num_controls - 1, theta] ]
    while len(cxcu1_units) > 0:
        cxcu1_units += add_cxcu1_units(qc, cxcu1_units, controls, target)
    qc.h(target)


############### BV Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
    
    # print a sample circuit created (if not too large)
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    print("\nOracle ="); print(grover_oracle if grover_oracle!= None else "  ... too large!")
    print("\nDiffuser ="); print(diffusion_operator )
