"""QED-C Application-Oriented Quantum Computing Benchmarks."""

import qedclib
from pathlib import Path

# Register this package's root so qedclib's dynamic loader can find benchmark kernels
qedclib.set_benchmark_root(Path(__file__).parent)
