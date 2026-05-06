# Prototype Benchmarks Project Design Notes
This folder contains information about the design of the program code in this repository.

The code is designed to be easy for non-experts to understand and execute.
Program code for the benchmarks themselves is designed to contain only what is relevant to that particular benchmark.
There are several supporting modules that are shared by all benchmark programs that provide services to store metrics and to execute quantum programs on specific simualtion and hardware targets.

## The Benchmark Pattern

All of the benchmark programs in this repository are structured using a semi-standard pattern of execution designed to provide uniformity across benchmarks.
Of course, there are unique aspects to each of the benchmarks and they will vary from this pattern in some cases.

In general, each benchmark program is centered around a specific circuit type, for example a Hidden Shift algorithm.
The program structure involves an outer loop over a range of different qubit sizes, and an inner loop that loops over some number of variations of that circuit type for that number of qubits. 

The circuits are created by calling the circuit definition routine with parameters based on the qubit size and circuit variations. 
Multiple circuits may be submitted for batched execution. At various points a batch of circuits is executed on a target system, either a simulator or quantum hardware. As execution proceeds, various benchmark metrics are accumulated and circuit fidelity calculated.
At the end of circuit execution, a series of bar charts is produced comparing the various metrics averged for all circuits of a particular size.

```
   *** More to come in this section later, including a data flow diagram ***
```


## The Execute Module: execute.py

The code for executing a quantum program on a selected target is contained in a module called **execute.py**.
This module provides a way to submit a series of circuits to be executed in a batch.
When the batch is executed, each circuit is launched as a 'job' to be executed on the target system.
Upon completion, the results from each job are processed in a custom 'result handler' function in order to calculate metrics such as fidelity.
Relevant benchmark metrics are stored for each circuit execution, so they can be aggregated and presented to the user.

The following are the important methods that may be called from a benchmark program:

#### init_execution (handler)
```  
  Initialize the execution module and pass to it a custom results handler.
```
#### submit_circuit (qc, group_id, circuit_id)
```  
  Submit a circuit for execution as part of the current batch.
  Arguments are the circuit handle 'qc', group identifier 'group_id', and circuit identifier 'circuit_id'.  
  The group and circuit ids are passed to the result handler to be used as keys to control
  aggregation of metrics for reporting.
```
#### execute_circuits ()
```
  Execute the current batch of circuits.
  As each circuit completes execution, call the user-provided 'result_handler()' function.
  Arguments passed to the results handler are the circuit handle 'qc', the result data structure 'result',
  group identifier 'group_id', and circuit identifier 'circuit_id'.
```

Below is a data flow diagram that illustrates the implementation of the first two steps in execution of a benchmark: 1) submitting one or more circuits in a 'batch' and 2) launching execution of the entire batch.

![Execute Module - Steps 1 and 2](./images/execute_module_1_2.png)

Below is a data flow diagram that illustrates the implementation of the job completion step. 
The job id for the completed job ised used to find the associated circuit information from the dictionary of active circuits.
From the submit and launch times, the elapsed and execute times are calculaated and stored to the metrics store, using the 'metrics' module.

Additionally, a user-defined 'result_handler()' function is invoked and passed the result data from the completed job.
From this, the fidelity is calculated  in a benchmark-specific way and passed to metric store.

![Execute Module - Step 3](./images/execute_module_3.png)

## The Metrics Module: metrics.py

The code for managing and reporting on a collection of metrics is provided in the module 'metrics.py'.
This module contains methods to create a 2 level table of metrics, indexed by group and circuit id.
Each group/circuit entry can contain any number of metric / value pairs.

In addition, the module provides methods to report the metrics aggregated over each group, as well 
as plot a bar chart for these aggregated metrics across all groups.
The plot function generates a bar chart for the 'create_time', 'exec_time', and 'fidelity' metrics.

The following are the important methods that may be called from a benchmark program:

#### init_metrics ()
```  
  Initialize the metrics module, creating an empty table of metrics.
```
#### store_metric (group, circuit, metric, value)
```  
  Store one metric / value pair into the table, indexed by a group id and circuit id.
```
#### aggregate_metrics_for_group (group)
```
  Aggregate metrics for a specific group, creating average across circuits in group.
```
#### aggregate_metrics ()
```
  Aggregate all metrics by group.
```
#### report_metrics_for_group (group)
```
  Report metrics for a specific group.
```
#### report_metrics ()
```
  Report all metrics for all groups.
```
#### plot_metrics ()
```
  Plot bar charts for each metric over all groups.
```

Below is a data flow diagram (in two parts) that illustrates the implementation of the functions contained in the metrics module.
All metrics are stored uniquely into the circuit_metrics dictionary, and indexed by the group_id, circuit_id, and metric name.

![Metrics Module - 1](./images/metrics_module_1.png)

Reporting can be done either on a per group basis or for all groups, as the metrics are aggregated within each group.
When plotting, each bar chart shows the aggregated metric values for all groups.
The data model for the group_metrics is designed to support both of these functions, implemented as a dictionary of arrays.

![Metrics Module - 2](./images/metrics_module_2.png)



