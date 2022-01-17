'''
Keysight TrueQ - Randomized Compilation
'''

import trueq as tq
import numpy as np
import trueq.compilation as tqc
import qiskit as qs

def local_rc(circuit, backend, n_compilations=20):   
    compiler = tq.Compiler.basic(
    entangler=tq.Gate.cnot, # Must choose a Clifford entangler for RC
    passes=(tqc.Native2Q,
            tqc.UnmarkCycles,
            tqc.MarkCycles,
            tqc.RCLocal,
            tqc.UnmarkCycles,
            tqc.Justify
        )
    )

    tq.interface.qiskit.set_from_backend(backend)
    circuit_tq, metadata = tq.interface.qiskit.to_trueq_circ(circuit)
    
    for qubit in circuit.qubits:
        if qubit.index not in circuit_tq.labels:
            del metadata.mapping[qubit.index]

    rc_collection = [
                tq.interface.qiskit.from_trueq_circ(compiler.compile(circuit_tq), 
                metadata=metadata)
                for j in range(n_compilations)
    ]
    
    return rc_collection
        
def full_rc(circuit, backend, n_compilations=20):
    compiler = tq.Compiler.basic(
        entangler=tq.Gate.cnot, # Must choose a Clifford entangler for RC
        passes=(tqc.Native2Q,
                tqc.UnmarkCycles,
                tqc.MarkCycles,
                tqc.RCCycle,
                tqc.Justify
            )
    )
    tq.interface.qiskit.set_from_backend(backend)
    circuit_tq, metadata = tq.interface.qiskit.to_trueq_circ(circuit)
    
    for qubit in circuit.qubits:
        if qubit.index not in circuit_tq.labels:
            del metadata.mapping[qubit.index]
    
    rc_collection = [
                tq.interface.qiskit.from_trueq_circ(compiler.compile(circuit_tq), 
                metadata=metadata)
                for j in range(n_compilations)
    ]
    return rc_collection