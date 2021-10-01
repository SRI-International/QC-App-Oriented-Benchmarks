# Prototype Docker container files 

To build and run a container for the Cirq platform:

cd  to  the parent folder of QC-Proto-Benchmarks and then run the following commands:

docker image rm cirqbenchmark:cirqbenchmarktag;./QC-Proto-Benchmarks/_containerbuildfiles/cirq/build.sh
 
docker run --rm -it -p 8886:8886 --name qedc_benchmark_cirq cirqbenchmark:cirqbenchmarktag


## TODO: 
	consolidate platform references in build and run commands.  Potentially create scripts
	