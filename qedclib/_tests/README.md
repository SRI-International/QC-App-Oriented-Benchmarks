# qedclib Developer Tests

Internal test scripts for exercising the qedclib execution API during development.
These are not user-facing examples — see [qedclib-examples](https://github.com/quantumcomputingdata/qedclib-examples) for that.

## Prerequisites

From the repo root: `pip install -e .`

## Scripts

| Script | Purpose |
|--------|---------|
| `01_basic_api.py` | Smoke test: import, configure, execute, inspect results |
| `02_parameter_sweep.py` | Batch execution with parameterized circuits (Ising model) |
| `03_backend_switching.py` | Backend switching via `-b` flag (simulator, IBM, IonQ, IQM) |
| `04_batch_scaling.py` | Batch size and qubit count scaling, serial vs batch comparison |

## Usage

```bash
cd qedclib/_tests
python 01_basic_api.py
python 03_backend_switching.py -b qasm_simulator
python 03_backend_switching.py -b ibm
```
