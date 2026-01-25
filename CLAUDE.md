# QED-C Benchmarks Restructuring - Context Document

## Project Overview

This document captures the context of restructuring the QED-C Application Oriented Benchmarks project to resolve Python namespace collision issues and enable cleaner imports.

### Primary Goals
1. Remove `__init__.py` files from API subdirectories (qiskit/, cirq/, cudaq/, ocean/) to avoid conflicts with vendor packages (e.g., importing `qiskit` would find the local directory instead of the installed package)
2. Use `qedc_benchmarks_init` for dynamic module loading based on API
3. Create top-level benchmark entry points that work with `python -m benchmark_name.benchmark_name`
4. Enable testing via a standard test script that runs method 1 (fidelity benchmark) on all benchmarks

### Branch Information
- Working branch: `main` (qedcappbms variant)
- Target: Eventually merge to master after testing

---

## Patterns Established

### 1. Directory Structure Rules
- **Benchmark directories** (amplitude_estimation/, monte_carlo/, etc.): **KEEP** `__init__.py`
- **API subdirectories** (qiskit/, cirq/, cudaq/, ocean/): **DELETE** `__init__.py`
- **Local `_common` directories**: **KEEP** `__init__.py`

### 2. Thin Wrapper Pattern
For complex benchmarks (VQE, HHL, Shors, hydrogen_lattice, maxcut, image_recognition), create a thin wrapper at the top level that delegates to the API-specific implementation.

```python
'''
Benchmark Program
(C) Quantum Economic Development Consortium (QED-C) 2024.

This is a thin wrapper that delegates to the API-specific implementation.
'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from _common.qedc_init import qedc_benchmarks_init

benchmark_name = "Benchmark Name"

def run(...parameters...):
    # Configure the QED-C Benchmark package for use with the given API
    qedc_benchmarks_init(api if api else "qiskit", "benchmark_name", ["benchmark_file"])

    # Import the actual benchmark module (now available after qedc_init)
    import benchmark_file as impl

    # Handle None backend_id
    if backend_id is None:
        backend_id = "qasm_simulator"

    # Delegate to the implementation
    impl.run(...parameters...)

def load_data_and_plot(folder=None, backend_id=None, **kwargs):
    """Relay function - assumes run() was already called."""
    import benchmark_file as impl
    impl.load_data_and_plot(folder=folder, backend_id=backend_id, **kwargs)
```

### 3. Import Pattern in API Implementations
Keep original imports in the qiskit/cirq/etc implementations:
```python
from benchmark._common import common  # package-style for local _common
from _common.qiskit import execute as ex  # global _common with API
from _common import metrics  # global _common (MUST use this style, not "import metrics")
```

### 4. Handling backend_id=None
When argparse passes `backend_id=None`, it overrides function defaults. Handle in wrapper:
```python
if backend_id is None:
    backend_id = "qasm_simulator"
```

### 5. Relative Paths in Implementations
Change relative paths to absolute paths based on file location:
```python
# Bad (breaks when run from different directory):
thetas_array_path = "../_common/instances/"

# Good:
thetas_array_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common", "instances") + os.sep
```

---

## Benchmarks Modified

### Fully Restructured (Full Pattern)
- **amplitude_estimation**: Top-level `ae_benchmark.py` with kernel in `qiskit/ae_kernel.py`
- **monte_carlo**: Top-level `mc_benchmark.py` with kernel in `qiskit/mc_kernel.py`

### Thin Wrapper Pattern
| Benchmark | Wrapper File | Implementation | Notes |
|-----------|--------------|----------------|-------|
| vqe | `vqe/vqe_benchmark.py` | `vqe/qiskit/vqe_benchmark.py` | |
| hhl | `hhl/hhl_benchmark.py` | `hhl/qiskit/hhl_benchmark.py` | |
| shors | `shors/shors_benchmark.py` | `shors/qiskit/shors_benchmark.py` | Inlined QFT functions |
| hydrogen_lattice | `hydrogen_lattice/hydrogen_lattice_benchmark.py` | `hydrogen_lattice/qiskit/hydrogen_lattice_benchmark.py` | Has `load_data_and_plot` |
| maxcut | `maxcut/maxcut_benchmark.py` | `maxcut/qiskit/maxcut_benchmark.py`, `maxcut/ocean/maxcut_benchmark.py` | Multi-API, has `load_data_and_plot` |
| image_recognition | `image_recognition/image_recognition_benchmark.py` | `image_recognition/qiskit/image_recognition_benchmark.py` | Fixed relative path issue, has `load_data_and_plot` |

### Already Correct Structure
- **quantum_reinforcement_learning**: Main benchmark already at top level with its own `qedc_benchmarks_init`. Just added `__init__.py` for package imports.

---

## Files Deleted (API __init__.py files)

```
vqe/qiskit/__init__.py
hhl/qiskit/__init__.py
shors/qiskit/__init__.py
hydrogen_lattice/qiskit/__init__.py
maxcut/qiskit/__init__.py
maxcut/ocean/__init__.py
image_recognition/qiskit/__init__.py
```

---

## Key Technical Issues Encountered

### 1. Module Instance Mismatch
**Problem**: Changing `from _common import metrics` to `import metrics` created a different module instance than what `execute.py` uses, causing `circuit_metrics["subtitle"]` to be None.
**Solution**: Always use `from _common import metrics` to match execute.py's import style.

### 2. Cross-Benchmark Dependencies
**Problem**: Some benchmarks import functions from other benchmarks (e.g., Shors imports QFT).
**Solution**: Inline the required functions rather than using sys.path manipulation.

### 3. Notebook Import Issues
**Problem**: Notebooks in qiskit/ directories can't find `_common` without path setup.
**Solution**: Either:
- Add `sys.path.insert(0, "../..")` at notebook start
- Run `pip install -e .` from benchmark root
- Start Jupyter from benchmark root directory

### 4. Relative Path Issues
**Problem**: Paths like `"../_common/instances/"` break when running from different directories.
**Solution**: Use absolute paths based on `__file__` location.

---

## Testing

### Test Command Pattern
From benchmark root:
```bash
python benchmark_name/benchmark_name.py
```

Or as module:
```bash
python -m benchmark_name.benchmark_name
```

### Test Script Location
`_tests/` directory contains test scripts for running benchmarks.

---

## Next Steps

1. **Merge to master branch copy for testing**: Create a copy of master, merge these changes, test comprehensively
2. **Update notebooks**: Some notebooks may need import updates from `from benchmark.qiskit import module` to `from benchmark import module`
3. **Handle notebook path issues**: Decide on standard approach for notebooks (sys.path vs pip install)
4. **Test all benchmarks**: Run method 1 (fidelity) on all benchmarks to verify functionality
5. **Ocean implementation**: maxcut/ocean needs testing when D-Wave access is available
6. **Documentation**: Update README files to reflect new import patterns

---

## File Locations

- Benchmark root: `C:\Dropbox\Common\github-tl\qcwork\apps\QC-App-Oriented-Benchmarks-qedcappbms\`
- Global _common: `_common/`
- qedc_init module: `_common/qedc_init.py`
- Test scripts: `_tests/`

---

## Commands Reference

```bash
# Run a benchmark
python hydrogen_lattice/hydrogen_lattice_benchmark.py

# Run with specific parameters
python maxcut/maxcut_benchmark.py --min_qubits 3 --max_qubits 6 --method 1

# Install package in development mode (enables imports from anywhere)
pip install -e .
```
