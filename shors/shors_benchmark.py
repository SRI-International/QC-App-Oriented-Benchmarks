'''
Shor's Order Finding Algorithm Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.

This is a thin wrapper that delegates to the qiskit implementation.
'''

# Add benchmark home dir to path, so the benchmark can be run without pip installing.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from _common.qedc_init import qedc_benchmarks_init

# Benchmark Name
benchmark_name = "Shor's Order Finding"


def run(min_qubits=3, max_circuits=1, max_qubits=18, num_shots=100, method=1,
        backend_id=None, provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None,
        context=None, api=None, get_circuits=False):

    # Configure the QED-C Benchmark package for use with the given API
    # Note: Shors primarily uses qiskit implementation
    qedc_benchmarks_init(api if api else "qiskit", "shors", ["shors_benchmark"])

    # Import the actual benchmark module (now available after qedc_init)
    import shors_benchmark as shors_impl

    # Delegate to the implementation
    shors_impl.run(
        min_qubits=min_qubits, max_circuits=max_circuits, max_qubits=max_qubits,
        num_shots=num_shots, method=method,
        backend_id=backend_id, provider_backend=provider_backend,
        hub=hub, group=group, project=project, exec_options=exec_options,
        context=context, api=api, get_circuits=get_circuits
    )


#######################
# MAIN

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Shor's Order Finding Benchmark")
    parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits (min = max = N)", type=int)
    parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=18, help="Maximum number of qubits", type=int)
    parser.add_argument("--max_circuits", "-c", default=1, help="Maximum circuit repetitions", type=int)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    if args.num_qubits > 0:
        args.min_qubits = args.max_qubits = args.num_qubits

    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        max_circuits=args.max_circuits, num_shots=args.num_shots,
        method=args.method,
        backend_id=args.backend_id,
        exec_options={"noise_model": None} if args.nonoise else {},
        api=args.api
    )
