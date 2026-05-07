"""
QED-C Quantum Computing Library

Provides execution infrastructure, metrics collection, and backend abstraction
for quantum computing applications.

Usage:
    import qedclib

    qedclib.set_api("qiskit")
    kernel = qedclib.get_kernel("hs_kernel", api="qiskit")

    # Metrics always available
    qedclib.metrics.plot_metrics(...)
"""

from qedclib.api import (
    qedc_benchmarks_init as initialize,
    qedc_set_api as set_api,
    qedc_get_api as get_api,
    qedc_get_kernel as get_kernel,
    qedc_is_leader as is_leader,
)
from qedclib import metrics
from qedclib import qcb_mpi
from qedclib import job_store
from qedclib.batched import batched_run

__version__ = "2.0.0"

# Benchmark root registration for dynamic kernel loading
_benchmark_root = None

def set_benchmark_root(path):
    """Register the root directory where benchmark packages live."""
    global _benchmark_root
    _benchmark_root = path
