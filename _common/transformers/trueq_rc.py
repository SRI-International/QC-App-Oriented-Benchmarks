'''
Keysight TrueQ - Randomized Compilation
'''

import trueq as tq
import numpy as np
import trueq.compilation as tqc
import qiskit as qs

def do_rc(circuit, backend, n_compilations=20):
    print("  ... performing randomized compilation.")
    rc_compressed_compiler = tq.Compiler.basic(
        entangler=tq.Gate.cnot, # Must choose a Clifford entangler for RC
        passes=(tqc.Native2Q,
                tqc.MarkCycles,
                tqc.RCLocal,
                tqc.UnmarkCycles,
                tqc.Merge,
                tqc.RemoveId, 
                tqc.RemoveEmptyCycle,
                tqc.MarkCycles
            )
    )
    qiskit_passes = None
    tq.interface.qiskit.set_from_backend(backend)
    circuit_tq =tqc.UnmarkCycles().apply(circuit.to_trueq()) 
    rc_collection = [rc_compressed_compiler.compile(circuit_tq).to_qiskit(passes=qiskit_passes)
            for j in range(n_compilations)
                    ]
    return rc_collection
    