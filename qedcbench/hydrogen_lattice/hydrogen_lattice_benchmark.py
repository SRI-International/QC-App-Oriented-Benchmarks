'''
Hydrogen Lattice Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.

This is a thin wrapper that delegates to the qiskit implementation.
'''

# Add benchmark home dir to path, so the benchmark can be run without pip installing.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from qedclib import initialize

# Benchmark Name
benchmark_name = "Hydrogen Lattice"


def run(**kwargs):
    """Create circuits, execute, and plot. Delegates to qiskit implementation.
    See hydrogen_lattice/qiskit/hydrogen_lattice_benchmark.py for detailed parameter documentation."""

    # Configure the QED-C Benchmark package for use with the given API
    # Note: hydrogen_lattice only has qiskit implementation
    initialize(kwargs.pop('api', None) or "qiskit", "hydrogen_lattice", ["hydrogen_lattice_benchmark"])

    # Import the actual benchmark module (now available after qedc_init)
    import hydrogen_lattice_benchmark as hl_impl

    # Use default backend_id if None passed
    if kwargs.get('backend_id') is None:
        kwargs['backend_id'] = "qasm_simulator"

    # Delegate to the implementation
    return hl_impl.run(**kwargs)


#######################
# MAIN

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Hydrogen Lattice Benchmark")
    parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits (min = max = N)", type=int)
    parser.add_argument("--min_qubits", "-min", default=2, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=4, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=2, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--method", "-m", default=2, help="Algorithm Method", type=int)
    parser.add_argument("--max_batch_size", "-mbs", default=None, help="Max circuits per execution batch", type=int)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--noplot", "-nop", action="store_true", help="Do not plot results")
    parser.add_argument("--nodraw", "-nod", action="store_true", help="Do not draw circuit diagram")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    if args.num_qubits > 0:
        args.min_qubits = args.max_qubits = args.num_qubits

    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        method=args.method,
        backend_id=args.backend_id,
        exec_options={"noise_model": None} if args.nonoise else {},
        api=args.api,
        max_batch_size=args.max_batch_size,
        draw_circuits=not args.nodraw, plot_results=not args.noplot
    )
