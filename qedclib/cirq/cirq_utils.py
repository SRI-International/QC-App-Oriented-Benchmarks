import cirq

class to_gate(cirq.Gate):
    def __init__(self, num_qubits, circ, name="G"):
        self.num_qubits=num_qubits
        self.circ = circ
        self.name = name
        
    def _num_qubits_(self):
        return self.num_qubits
    
    def _decompose_(self, qubits):
        # `sorted()` needed to correct error in `all_qubits()` not returning a reasonable order for all of the qubits
        qbs = sorted(list(self.circ.all_qubits()))
        mapping = {}
        for t in range(self.num_qubits):
            mapping[qbs[t]] = qubits[t]
        def f_map(q):
            return mapping[q]
        
        circ_new = self.circ.transform_qubits(f_map)
        return circ_new.all_operations()
    
    def _circuit_diagram_info_(self, args):
        return [self.name] * self._num_qubits_()
