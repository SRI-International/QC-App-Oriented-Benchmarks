"""Backwards-compatibility shim. Use 'import qedclib' for new code."""
from qedclib import initialize as qedc_benchmarks_init
from qedclib import get_kernel as qedc_get_kernel
from qedclib import set_api as qedc_set_api
from qedclib import get_api as qedc_get_api
from qedclib import is_leader as qedc_is_leader
