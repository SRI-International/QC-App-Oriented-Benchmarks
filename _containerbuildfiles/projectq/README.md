# Prototype Docker container files 

To build and run a container for the ProjectQ platform:

cd  to  the parent folder of QC-Proto-Benchmarks and then run the following commands:

docker image rm projectqbenchmark:projectqbenchmarktag;./QC-Proto-Benchmarks/_containerbuildfiles/projectq/build.sh
 
docker run --rm -it -p 8887:8887 --name qedc_benchmark_projectq projectqbenchmark:projectqbenchmarktag


## TODO: 
	consolidate platform references in build and run commands.  Potentially create scripts
	