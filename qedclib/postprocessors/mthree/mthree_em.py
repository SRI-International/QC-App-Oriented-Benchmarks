#################################################################
# 
# Post-processor module: mthree error mitigation handlers

import importlib
import mthree
from qiskit import Aer

mit = None
mit_num_qubits = 2

verbose = False

# Handler called to postprocess results before they are passed to calling program
def mthree_handler(result):
    if verbose: print(f'Before: {dict(sorted(result.results[0].data.counts.items())) = }')

    raw_counts = result.get_counts()
    shots = sum(raw_counts.values())
    
    quasi_counts = mit.apply_correction(raw_counts, range(len(list(raw_counts.keys())[0])))
    for k, v in quasi_counts.items():
        quasi_counts[k] = round(v * shots) 
    
    if verbose: print(f'Quasi: {quasi_counts = }')
    
    qmin = min(quasi_counts.values())
    qmax = max(quasi_counts.values())
    qshots = sum(quasi_counts.values())
    qrange = (qmax - qmin)
    if verbose: print(f"{qmin = } {qmax = } {qrange = } {qshots = } {shots = }")

    # store modified counts in result object
    result.results[0].data.counts = quasi_counts

    if verbose: print(f'After: {result.results[0].data.counts = }')

    return result

# Handler called to configure mthree for number of qubits in executing circuit
def mthree_width_handler(circuit):
    global mit_num_qubits
    
    num_qubits = circuit.num_qubits
    # print(circuit)

    if num_qubits != mit_num_qubits:
        if verbose: print(f'... updating mthree width to {num_qubits = }')
        mit_num_qubits = num_qubits
        mit.cals_from_system(range(num_qubits))

# Routine to initialize mthree and return two handlers
def get_mthree_handlers(backend_id, provider_backend):
    global mit

    # special handling for qasm_simulator
    if backend_id.endswith("qasm_simulator"):
        provider_backend = Aer.get_backend(backend_id) 

    # special handling for fake backends
    elif 'fake' in backend_id:
        backend = getattr(
            importlib.import_module(
                f'qiskit.providers.fake_provider.backends.{backend_id.split("_")[-1]}.{backend_id}'
            ),
            backend_id.title().replace('_', '')
        )
        provider_backend = backend()

    # initialize mthree with given backend
    mit = mthree.M3Mitigation(provider_backend)

    if verbose: print(f"... initializing mthree for backend_id = {backend_id}")
    
    mit.cals_from_system(range(mit_num_qubits))

    return (mthree_handler, mthree_width_handler)