"""Compatibility module — keeps 'from qedclib.qedc_init import ...' working."""
from qedclib._init_engine import (
    qedc_benchmarks_init,
    qedc_set_api,
    qedc_get_api,
    qedc_get_kernel,
    qedc_is_leader,
    reset_module_caches,
)
