
## Executing the Application Benchmark Programs via the Qiskit Benchmark Runner

It is possible to run the benchmarks from the top level directory in a generalized way on the command line.

For instance, here is an example of running the default QFT algorithm from the
`quantum-fourier-transform` directory:

```
python _common/qiskit/benchmark_runner.py --algorithm quantum-fourier-transform
```

### Optional Arguments to All Benchmarks

Alternatively, one can supply custom arguments to this call if they differ from the defaults. These options are optional, and execution will proceed with default values if they are not provided.

One of the arguments permits the specification of a custom noise model function or None (no noise model) used only when the 'qasm_simulator' is specified as the 'backend_id'. When using this argument the function must be defined in a python file located within the _custom folder at the top level of the repository.

If the 'backend_id' argument is specified it is used to direct execution to either the Qiskit Aer simulator ('qasm_simulator', the default) or one of the backend systems available in the IBM Quantum Computing service (e.g. 'ibms_jakarta').  See the section below about arguments related to backend providers for more information about the 'backend_id' argument and its use with other providers.

For instance:

```
 python _common/qiskit/benchmark_runner.py 
    --algorithm "quantum-fourier-transform"
    --min_qubits 2 
    --max_qubits 8 
    --max_circuits 3 
    --num_shots 100 
    --method 2 
    --backend_id "qasm_simulator"  
    --noise_model "custom_qiskit_noise_model.my_noise_model()"
```

### Optional Arguments for Backend Providers

In addition to the arguments above there are arguments used to specify the hardware backend which on which to execute the benchmarks. These need to be specified to select a harware backend other than those provided by IBM. To do this, both the 'provider_module_name" (the python module) and the 'provider_class_name" must be sepecified. The runner will make an instance of the specified provider class using the backend_id argument and use this backend instance to execute the benchmarks.

The 'hub', ' group', and 'project' arguments are optional and typically only apply to IBM systems.

```
 python _common/qiskit/benchmark_runner.py 
    --algorithm "quantum-fourier-transform"
    --min_qubits 2 
    --max_qubits 8 
    --max_circuits 3 
    --num_shots 100 
    --method 2 
    --backend_id " backend id "
    --provider_module_name " Provider module name " 
    --provider_class_name  " Provider class name "
    --hub "open" 
    --group "open" 
    --project "main" 
```

Note:- that any arguments of type `dict` when using --exec_options must be provided as a string where both
the key and value pair are also string values.

### Optional Benchmark-Specific Arguments

In general, the arguments one can supply for a given benchmark are defined by the associated `run` method are described when executing the runner with only the algorithm argument:

```
python _common/qiskit/benchmark_runner.py ALGORITHM_NAME -h
```

