# Prototype Docker container files 

To build and run a container for the Qiskit platform:

cd  to  the parent folder of QC-Proto-Benchmarks and then run the following commands:

docker image rm qiskitbenchmark:qiskitbenchmarktag;./QC-Proto-Benchmarks/_containerbuildfiles/qiskit/build.sh
 
docker run --rm -it -p 8888:8888 --name qedc_benchmark_qiskit qiskitbenchmark:qiskitbenchmarktag


## TODO: 
	consolidate platform references in build and run commands.  Potentially create scripts
	