# Using TKET Transformers

`pytket` is a python module for interfacing with CQC TKET, a set of quantum 
programming tools. These tools include an extensive collection of 
compilation and optimisation passes, which are made use of in 
`tket_optimiser.py`. 

`pytket` and TKET are [open source](https://github.com/CQCL/tket), 
and `pytket` is available for Python 3.8, 3.9 and 3.10, on Linux, MacOS and 
Windows. To install, run:
```
pip install pytket
```
The transformers available here will run with `pytekt>=0.19`. 
Further details about `pytket` can be found in the
[Documentation](https://cqcl.github.io/tket/pytket/api/index.html) and
[Manual](https://cqcl.github.io/pytket/manual/index.html).

The `pytket` ecosystem also includes a collection of 
[extensions](https://cqcl.github.io/pytket-extensions/api/index.html) which 
enable CQC pytket to be used in conjunction with other platforms. 
Each extension adds either new methods to the pytket package to convert 
between circuit representations, or new backends to which pytket 
circuits can be submitted. To make use of the transformers here, 
`pytket-qiskit` will need to be installed as follows:
```
pip install pytket-qiskit
```
`pytekt-qiskit>=0.22` are appropriate for the transformers 
available here.

To use these TKET transformers, simply pass then as a transformer in the
exec_options:
```
import _common.transformers.tket_optimiser as tket_optimiser
import sys
import dj_benchmark

exec_options = { "optimization_level":0, "layout_method":'sabre', "routing_method":'sabre', "transformer": tket_optimiser.quick_optimisation }

sys.path.insert(1, "deutsch-jozsa/qiskit")
dj_benchmark.run(min_qubits=min_qubits, max_qubits=max_qubits, max_circuits=max_circuits, num_shots=num_shots,
                backend_id=backend_id, provider_backend=provider_backend,
                hub=hub, group=group, project=project, exec_options=exec_options)
```
Here we have used `tket_optimiser.quick_optimisation` which provides basic
functionality to ensure that the device constraints are met. `tket_optimiser`
also provides `high_optimisation`, which is slower but performs more extensive
optimisations.