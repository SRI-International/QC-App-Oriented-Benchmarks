# HA algorithm

This repository contains the implementation of our Hardware-Aware mapping algorithm.

## Installation

The repository is a Python package. You can install by cloning with `git` and then using Python's package manager `pip`:

``` shell
git clone https://github.com/peachnuts/HA.git
python -m pip install HA
```

## Example

``` python
from qiskit import QuantumCircuit

from hamap import (
    ha_mapping,  # Bridge selection performed with a
                 # different algorithm than the one described in the paper.
    ha_mapping_paper_compliant,  # Bridge selection using the exact same algorithm
                                 # described in the paper.
    IBMQHardwareArchitecture,
)

# Create a Quantum Circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

hardware = IBMQHardwareArchitecture("ibmq_16_melbourne")
initial_mapping = {qubit: i for i, qubit in enumerate(circuit.qubits)}

# Map the circuit with our hardware-aware heuristic and using SWAP & Bridge gates.
# Replace "ha_mapping" with "ha_mapping_paper_compliant" to use the version 100% 
# compliant with the paper.
mapped_circuit, final_mapping = ha_mapping(
    circuit, initial_mapping, hardware
)

print(mapped_circuit.draw())
```

## How to use?

The main function that should be the entry point for any user is [`hamap.ha_mapping`](https://github.com/peachnuts/HA/blob/master/src/hamap/mapping.py#L69).
This function takes as parameter:

1. An instance of `qiskit.QuantumCircuit` representing the circuit to map.
2. An initial mapping, given as a dictionnary that maps the instances of `qiskit.Qubit` contained in the quantum circuit given as first parameter to the physical qubit identifier, i.e. an integer representing a physical qubit on the hardware we map the circuit to.
3. An instance of `hamap.IBMQHardwareArchitecture` that wraps Qiskit's API to retrieve calibration data and hardware information.
4. A function `swap_cost_heuristic` that takes as parameters
   1. An instance of `hamap.IBMQHardwareArchitecture`.
   2. An instance of `hamap.layer.QuantumLayer` representing the current "first layer".
   3. A list of `qiskit.dagcircuit.dagcircuit.DAGNode` that contains the nodes of the `DAGCircuit` (i.e. quantum gates) that are not mapped yet, sorted in a topological order (i.e. the first node is guaranteed to to have all its predecessors already mapped).
   4. The index of the current "first" gate, i.e. the first gate of the `DAGNode` list that have not already been mapped.
   5. The current mapping as a dictionnary that maps instances of `qiskit.Qubit` to hardware qubits indices.
   6. A `numpy` array that stores the distance between each pair of hardware qubits.
   7. An instance of `hamap.gates.TwoQubitGate` (or a derived class such as `SwapTwoQubitGate` or `BridgeTwoQubitGate`that represents a potential 2-qubit gate for which we want to compute the cost.
   
   and returns the cost of the `TwoQubitGate` given.
5. A function `get_candidates` that takes as parameters
   1. The current `QuantumLayer` containing the first layer of gates that are not already executed.
   2. The hardware architecture we are mapping the circuit to as an instance of `hamap.IBMQHardwareArchitecture`.
   3. The current mapping as a dictionnary that maps instances of `qiskit.Qubit` to hardware qubits indices.
   4. A set of strings representing the already explored mappings. In order to prevent the mapping algorithm being stuck in an infinite loop in some unusual conditions, the `get_candidate` function needs to know which mapping have already been explored and to avoid exploring them again. This parameter can be ignored safely, but be aware that in some rare cases the algorithm might end up in an infinite loop.
6. A function `get_distance_matrix` that takes an instance of `qubit_mapping_optimiser.IBMQHardwareArchitecture` and returns the distance matrix associated with the hardware. Note that by changing this parameter to a hand-crafted function you can define any distance you want.


## Notes on the implementation

The default implementation `ha_mapping` uses a slightly different method to chose between inserting a `SWAP` or a `Bridge` gate.
The algorithm described in the scientific paper first computes the best `SWAP` and then determine if it is worth changing the `SWAP` into a `Bridge` gate.
The `hamap.ha_mapping` function in this repository evaluates `Bridge` gates along `SWAP` ones, and pick the best gate according to the internal metric.
An implementation of the algorithm described in paper is given in `hamap.ha_mapping_paper_compliant`.



