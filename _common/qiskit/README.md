
## Executing the Application Benchmark Programs via the Qiskit Runner

It is possible to run the benchmarks from the top level directory in a generaized way on the command line.

For instance, here is an example of running the default QFT algorithm from the
`quantum-fourier-transform` directory:

```
python _common/qiskit/benchmark_runner.py --algorithm quantum-fourier-transform
```

Alternatively, one can supply custom arguments to this call if they differ from
the defaults.This options are optional 

You also have an Option to use custom Noise Model or No Noise Model along with existing custom Arguments.

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

In addition to Simulator Arguments their is an option to choose Hardware backend which needs to be provided if we want to differ the default values. In order to run on  provider backend choice of provider details shoulde be updated like provider module name & provider class name according to your choice of provider

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

In general, the arguments one can supply for a given algorithm are defined by
the associated `run` method are described when executing the runner with only the algorithm argument:

```
python _common/qiskit/benchmark_runner.py ALGORITHM_NAME -h
```

