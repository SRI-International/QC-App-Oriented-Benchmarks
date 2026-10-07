import pygsti
import tqdm
import numpy as np
import scipy as sp
from qiskit import transpile, QuantumCircuit
import re

# Below code created by Timothy Proctor, July 16 2024, using code developed by various members of Sandia's QPL.
# Code has been edited to accept random pauli bitstrings by Sonny Rappaport, July 23 2024.
# Additional functions below also by Sonny based on Tim's code. 

def sample_mirror_circuits(c, num_mcs=100, randomized_state_preparation=True, rand_state = None, random_pauli = True):
    if rand_state is None:
        rand_state = np.random.RandomState()

    mirror_circuits = []
    mirror_circuits_bitstrings = []
    for j in range(num_mcs):
        mc, bs = central_pauli_mirror_circuit(c, randomized_state_preparation=randomized_state_preparation,random_pauli = True, rand_state = rand_state)
        mirror_circuits.append(mc)
        mirror_circuits_bitstrings.append(bs)

    return mirror_circuits, mirror_circuits_bitstrings

def sample_reference_circuits(qubits, num_ref=100, randomized_state_preparation=True, rand_state=None):
    if rand_state is None:
        rand_state = np.random.RandomState()
              
    ref_circuits = []
    ref_circuits_bitstrings = []
    for j in range(num_ref):
        empty_circuit = pygsti.circuits.Circuit('', line_labels=qubits)
        ref_c, bs = central_pauli_mirror_circuit(empty_circuit, 
                                                 randomized_state_preparation=randomized_state_preparation,
                                                 rand_state=rand_state)
        ref_circuits.append(ref_c)
        ref_circuits_bitstrings.append(bs)

    return ref_circuits, ref_circuits_bitstrings

def central_pauli_mirror_circuit(circ, randomized_state_preparation=True, random_pauli = True, rand_state = None):
    if rand_state is None:
        rand_state = np.random.RandomState()

    qubits = circ.line_labels
    if randomized_state_preparation:
        prep_circ = pygsti.circuits.Circuit([haar_random_u3_layer(qubits, rand_state)], line_labels=qubits)
        circuit_to_mirror = prep_circ + circ
    else: 
        circuit_to_mirror = circ

    n = circuit_to_mirror.width
    d = circuit_to_mirror.depth

    # Sonny's edit here: Rather than generating central_pauli randomly, now we can use a non-random one if desired. 
    if random_pauli is True:
        rand_state = np.random.RandomState()
        central_pauli = 2 * np.random.randint(0, 2, 2*n) #old code 
    else: 
        central_pauli = 2 * np.ones(2*n, dtype = np.int8)  

    central_pauli_layer = pauli_vector_to_u3_layer(central_pauli, qubits)
    q = central_pauli.copy()

    quasi_inverse_circ = pygsti.circuits.Circuit(line_labels=circ.line_labels, editable=True)

    for i in range(d):
        
        layer = circuit_to_mirror.layer_label(d - i - 1).components
        quasi_inverse_layer = [gate_inverse(gate_label) for gate_label in layer]
        
        # Update the u3 gates.
        if len(layer) == 0 or layer[0].name == 'Gu3':
            # update_u3_parameters now twirls/adds label for empty qubits, so don't prepad for speed
            #padded_layer = pad_layer(quasi_inverse_layer, qubits)
            #quasi_inverse_layer = update_u3_parameters(padded_layer, q, q, qubits)
            quasi_inverse_layer = update_u3_parameters(quasi_inverse_layer, q, q, qubits)
            
        # Update q based on the CNOTs in the layer.
        else:
            for g in layer:
                if g.name == 'Gcnot':
                    (control, target) = g.qubits
                    q[qubits.index(control)] = (q[qubits.index(control)] + q[qubits.index(target)]) % 4
                    q[n + qubits.index(target)] = (q[n + qubits.index(control)] + q[n + qubits.index(target)]) % 4
                else:
                    raise ValueError("Circuit can only contain Gcnot and Gu3 gates in separate layers!")

        quasi_inverse_circ.insert_layer_inplace(quasi_inverse_layer, i)

    mc = circuit_to_mirror + pygsti.circuits.Circuit([central_pauli_layer], line_labels=circ.line_labels) + quasi_inverse_circ
    mc.done_editing()

    bs = ''.join([str(b // 2) for b in q[n:]])

    return mc, bs

def pauli_vector_to_u3_layer(p, qubits):
    
    n = len(qubits)
    layer = []
    for i, q in enumerate(qubits):

        if p[i] == 0 and p[i+n] == 0:  # I
            theta = 0.0
            phi = 0.0
            lamb = 0.0
        if p[i] == 2 and p[i+n] == 0:  # Z
            theta = 0.0
            phi = np.pi / 2
            lamb = np.pi / 2
        if p[i] == 0 and p[i+n] == 2:  # X
            theta = np.pi
            phi = 0.0
            lamb = np.pi
        if p[i] == 2 and p[i+n] == 2:  # Y
            theta = np.pi
            phi = np.pi / 2
            lamb = np.pi / 2

        layer.append(pygsti.baseobjs.Label('Gu3', q, args=(theta, phi, lamb)))

    return pygsti.baseobjs.Label(layer)

def haar_random_u3_layer(qubits, rand_state=None):
    
    return pygsti.baseobjs.Label([haar_random_u3(q, rand_state) for q in qubits])

def haar_random_u3(q, rand_state=None):
    if rand_state is None:
        rand_state = np.random.RandomState()

    a, b = 2 * np.pi * np.random.rand(2)
    theta = mod_2pi(2 * np.arcsin(np.sqrt(np.random.rand(1)))[0])
    phi = mod_2pi(a - b + np.pi)
    lamb = mod_2pi(-1 * (a + b + np.pi))
    return pygsti.baseobjs.Label('Gu3', q, args=(theta, phi, lamb))

def mod_2pi(theta):
    while (theta > np.pi or theta <= -1 * np.pi):
        if theta > np.pi:
            theta = theta - 2 * np.pi
        elif theta <= -1 * np.pi:
            theta = theta + 2 * np.pi
    return theta

def inverse_u3(args):
    theta_inv = mod_2pi(-float(args[0]))
    phi_inv = mod_2pi(-float(args[2]))
    lambda_inv = mod_2pi(-float(args[1]))
    return (theta_inv, phi_inv, lambda_inv)

def gate_inverse(label):
    if label.name == 'Gcnot':
        return label
    elif label.name == 'Gu3':
        return pygsti.baseobjs.label.LabelTupWithArgs.init('Gu3', label.qubits, args=inverse_u3(label.args))

def pad_layer(layer, qubits):

    padded_layer = list(layer)
    used_qubits = []
    for g in layer:
        for q in g.qubits:
            used_qubits.append(q)

    for q in qubits:
        if q not in used_qubits:
            padded_layer.append(pygsti.baseobjs.label.LabelTupWithArgs.init('Gu3', (q,), args=(0.0, 0.0, 0.0)))

    return padded_layer


def update_u3_parameters(layer, p, q, qubits):
    """
    Takes a layer containing u3 gates, and finds a new layer containing
    u3 gates that implements p * layer * q (p followed by layer followed by
    q), where p and q are vectors  describing layers of paulis.

    """
    used_qubits = []

    new_layer = []
    n = len(qubits)

    for g in layer:
        assert(g.name == 'Gu3')
        (theta, phi, lamb) = (float(g.args[0]), float(g.args[1]), float(g.args[2]))
        qubit_index = qubits.index(g.qubits[0])
        if p[qubit_index] == 2:   # Z gate preceeding the layer
            lamb = lamb + np.pi
        if q[qubit_index] == 2:   # Z gate following the layer
            phi = phi + np.pi
        if p[n + qubit_index] == 2:  # X gate preceeding the layer
            theta = theta - np.pi
            phi = phi
            lamb = -lamb - np.pi
        if q[n + qubit_index] == 2:  # X gate following the layer
            theta = theta - np.pi
            phi = -phi - np.pi
            lamb = lamb

        new_args = (mod_2pi(theta), mod_2pi(phi), mod_2pi(lamb))
        new_label = pygsti.baseobjs.label.LabelTupWithArgs.init('Gu3', g.qubits[0], args=new_args)
        new_layer.append(new_label)
        used_qubits.append(g.qubits[0])
    
#     for qubit_index, qubit in enumerate(qubits):
#         if qubit in used_qubits:
#             continue

#         # Insert twirled idle on unpadded qubit
#         (theta, phi, lamb) = (0.0, 0.0, 0.0)
#         if p[qubit_index] == 2:   # Z gate preceeding the layer
#             lamb = lamb + np.pi
#         if q[qubit_index] == 2:   # Z gate following the layer
#             phi = phi + np.pi
#         if p[n + qubit_index] == 2:  # X gate preceeding the layer
#             theta = theta - np.pi
#             phi = phi
#             lamb = -lamb - np.pi
#         if q[n + qubit_index] == 2:  # X gate following the layer
#             theta = theta - np.pi
#             phi = -phi - np.pi
#             lamb = lamb
        
#         new_args = (mod_2pi(theta), mod_2pi(phi), mod_2pi(lamb))
#         new_label = pygsti.baseobjs.label.LabelTupWithArgs.init('Gu3', qubit, args=new_args)
#         new_layer.append(new_label)
#         used_qubits.append(qubit)
#
#    assert(set(used_qubits) == set(qubits))

    return new_layer

def hamming_distance_counts(data_dict, circ, idealout):
    nQ = len(circ.line_labels)  # number of qubits
    assert(nQ == len(idealout))
    total = np.sum(list(data_dict.values()))
    hamming_distance_counts = np.zeros(nQ + 1, float)
    for outcome, counts in data_dict.items():
        hamming_distance_counts[pygsti.tools.rbtools.hamming_distance(outcome, idealout)] += counts
    return hamming_distance_counts

def adjusted_success_probability(hamming_distance_counts):
    if np.sum(hamming_distance_counts) == 0.: 
        return 0.
    else:
        hamming_distance_pdf = np.array(hamming_distance_counts) / np.sum(hamming_distance_counts)
        adjSP = np.sum([(-1 / 2)**n * hamming_distance_pdf[n] for n in range(len(hamming_distance_pdf))])
        return adjSP

def effective_polarization(hamming_distance_counts):
    n = len(hamming_distance_counts) - 1 
    asp = adjusted_success_probability(hamming_distance_counts)
    
    return (4**n * asp - 1)/(4**n - 1)

def polarization_to_fidelity(p, n): 
    return 1 - (4**n - 1)*(1 - p)/4**n

def fidelity_to_polarization(f, n):
    return 1 - (4**n)*(1 - f)/(4**n - 1)

def predicted_process_fidelity(mirror_circuit_effective_pols, reference_effective_pols, n):
    a = np.mean(mirror_circuit_effective_pols)
    c = np.mean(reference_effective_pols)
    if c <= 0.:
        return np.nan  # raise ValueError("Reference effective polarization zero or smaller! Cannot estimate the process fidelity")
    elif a <= 0:
        return 0.
    else:
        return polarization_to_fidelity(np.sqrt(a / c), n)
    
def predicted_process_fidelity_with_spam_error(mirror_circuit_effective_pols, reference_effective_pols, n):
    a = np.mean(mirror_circuit_effective_pols)
    c = np.mean(reference_effective_pols)
    if c <= 0.:
        return np.nan  # raise ValueError("Reference effective polarization zero or smaller! Cannot estimate the process fidelity")
    elif a <= 0:
        return 0.
    else:
        return polarization_to_fidelity(np.sqrt(c * a), n)

# Below code created by Sonny Rappaport based off of the above code provided by Tim Proctor. 

def pygsti_string(qc):
    """
    Takes a qiskit circuit (that has been transpiled to the u3 and cx basis), and returns a pygsti-readable string. 
    """

    pygstr = ""

    for gate in qc.data:

        gate_str = "[" 

        gate_name = gate[0].name
        gate_qubits = [] 

        for qubit in gate[1]:

            # qubits look something like Qubit(QuantumRegister(4,'q1'),1)
            gate_qubits.append(qubit._index)
        
        gate_parameters = gate[0].params

        if gate_name == "cx":
            gate_str = gate_str + "Gcnot"
        elif gate_name == "u3":
            gate_str = gate_str + "Gu3"
        else:
            # print(f"Passed {gate_name}")

            continue

        for parameter in gate_parameters:

            gate_str = gate_str + f";{parameter}" 

        for qubit in gate_qubits:

            gate_str = gate_str + f":Q{qubit}"

        gate_str = gate_str + "]"

        pygstr = pygstr + gate_str

    pygstr = pygstr + "@(" + f"{''.join(f'Q{i},' for i in range(qc.num_qubits))[:-1]}" + ")"

    return pygstr.strip()

def convert_to_mirror_circuit(qc, random_pauli, init_state):
    """
    Takes an arbitrary quantum circuit, transpiles it to CX and u3, and eventually returns a mirror circuit version of the quantum circuit. 

    The mirror circuit will have a random initial state in addition to random paulis. 
    """

    # if no initial state is provided, use a ranmdom one
    random_init_flag = True  

    if init_state: 
        random_init_flag = False
        init_state.append(qc, range(qc.num_qubits))
        qc = init_state

    qc = transpile(qc, basis_gates=["cx","u3"])

    pygstr = pygsti_string(qc)

    pygsti_circuit = pygsti.circuits.Circuit(pygstr)

    mcs, bss = sample_mirror_circuits(pygsti_circuit, num_mcs=1, random_pauli = random_pauli, randomized_state_preparation=random_init_flag)

    qasm_string = mcs[0].convert_to_openqasm()

    # use regex magic to remove lines that begin with Delay gates
    delays_removed_string = re.sub(r'^delay.*\n?', '', qasm_string, flags=re.MULTILINE) 

    qiskit_circuit = QuantumCircuit.from_qasm_str(delays_removed_string)
    
    # this circuit is made out of u3 and cx gates by pygsti default
    # the bitstring is reversed to account for qiskit ordering
    return qiskit_circuit , bss[0][::-1]
