import importlib
import mthree
from qiskit import Aer

mit = None
mit_num_qubits = 0

verbose = False

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

def mthree_width_handler(circuit):
    global mit_num_qubits
    num_qubits = circuit.num_qubits #- circuit.num_ancillas
    # print(circuit)

    if num_qubits != mit_num_qubits:
        if verbose: print(f'... updating mthree width to {num_qubits = }')
        mit_num_qubits = num_qubits
        mit.cals_from_system(range(num_qubits))

def get_mthree_handlers(backend_id, provider_backend, num_qubits):
    print("Inside get_mthree_handler")
    global mit

    if backend_id.endswith("qasm_simulator"):
        provider_backend = Aer.get_backend(backend_id) 

    elif 'fake' in backend_id:
        backend = getattr(
            importlib.import_module(
                f'qiskit.providers.fake_provider.backends.{backend_id.split("_")[-1]}.{backend_id}'
            ),
            backend_id.title().replace('_', '')
        )
        provider_backend = backend()

    mit = mthree.M3Mitigation(provider_backend)

    global mit_num_qubits 
    mit_num_qubits = num_qubits

    if verbose: print("... initializing mthree")
    
    mit.cals_from_system(range(num_qubits))

    return (mthree_handler, mthree_width_handler)