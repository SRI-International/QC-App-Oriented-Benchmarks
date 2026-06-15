# CUDA-Q Multi-GPU (mgpu) Known Issues

This page documents specific issues encountered with the CUDA-Q multi-GPU statevector simulator (`nvidia-mgpu` / `cusvsim`) when running on real multi-GPU hardware. These issues are in the CUDA-Q runtime itself, not in qedclib code.

## Gate Grouping Failure on Multi-GPU with Certain Circuit Structures

**Date discovered**: June 8, 2026  
**CUDA-Q version**: 0.13.0  
**Affects**: Distributed statevector execution (`nvidia-mgpu`) on real multi-GPU hardware  
**Does NOT affect**: Single-GPU execution, or parallel mode (`-pm`) which uses single-GPU per rank  

### Summary

When running Hamiltonian observable estimation circuits on real multi-GPU hardware (e.g., 4x A100 on Perlmutter), the `cusvsim` multi-GPU statevector simulator fails with `RuntimeError: requested size is too big` and/or `gateGrouping.cpp:245: targets and/or controls are not included in wireIdOrdering` for circuits containing multi-qubit Pauli basis rotation gates (YY, XXX, YYXXX, etc.).

Circuits containing only single-qubit X/Z Pauli terms execute successfully. The issue appears to be in how the mgpu gate-grouping algorithm partitions certain gate patterns across GPUs.

### Reproduction

**Hamiltonian**: FH_D-1 (Fermi-Hubbard), 20 qubits, simple grouping  
**Produces 5 measurement circuits** — circuit 1 has only single-qubit X/Z terms; circuits 2-5 have multi-qubit terms (YY, XXX, YYXXX, etc.)

```bash
# This FAILS on real multi-GPU (Perlmutter 4x A100):
srun -n 4 python -m mpi4py -m qedcbench.hamlib.hamlib_simulation_benchmark \
    -a cudaq -obs -nop -nod -n 20 -gm simple -ham FH_D-1 -s 10000 -v
```

### Systematic Test Results

| Test Configuration | MPI | `-ds` | `-pm` | Result |
|---|---|---|---|---|
| Local, no MPI | No | No | No | **Works** |
| Local, no MPI | No | Yes | No | **Works** |
| Local, no MPI | No | Yes | Yes | **Works** (warning, sequential fallback) |
| Local, 4 MPI ranks on 1 GPU | Yes | Yes | No | **Works** |
| Local, 4 MPI ranks on 1 GPU | Yes | Yes | Yes | **Works** |
| Perlmutter, 4x A100 | Yes | No | No | **FAILS** — mgpu errors on circuits 2-5 |
| Perlmutter, 4x A100 | Yes | Yes | No | **FAILS** — identical errors |
| Perlmutter, 4x A100 | Yes | Yes | Yes | **Works** (switches to single-GPU per rank) |

**Key observation**: The `-ds` (distribute_shots) flag is irrelevant — the same circuits fail with or without it. The failure is purely related to the mgpu statevector simulator on real multi-GPU hardware.

### Why `-pm` (parallel mode) works

When `--parallel` (`-pm`) is enabled, qedclib's `_execute_parallel_mpi()` and `_execute_groups_parallel_mpi()` explicitly switch the target:

```python
cudaq.set_target("nvidia", option="fp32")  # single-GPU per rank
```

This bypasses the mgpu gate-grouping code entirely. Each rank uses its own GPU independently, avoiding the buggy code path.

Without `-pm`, the target remains `nvidia-mgpu` and all ranks cooperate on each `cudaq.sample()` call, which triggers the gate-grouping algorithm that fails on certain circuit structures.

### Environment Comparison

| | Local (CUDA-Q container) | Perlmutter (NERSC) |
|---|---|---|
| **CUDA** | 12.6 | 12.9 |
| **CUDA-Q** | 0.13.0 | 0.13.0 |
| **Real GPUs** | 1 (4 fake MPI ranks share it) | 4x NVIDIA A100 |
| **GPU fabric** | none | `CUDAQ_GPU_FABRIC=NVL` (NVLink) |
| **mgpu behavior** | All ranks share 1 physical GPU | Actual multi-GPU statevector partitioning |

The critical difference is **real multi-GPU statevector partitioning**. Locally with 4 MPI ranks on 1 GPU, the mgpu target doesn't actually partition across separate devices, so the gate-grouping algorithm doesn't encounter the same constraints.

### Error Messages

Errors appear during `cudaq.sample()` calls for circuits 2-5 (multi-qubit Pauli terms):

```
RuntimeError: requested size is too big
RuntimeError: /builds/nvhpc/cudaq_mgmn_svsim/cusvsim/ubackend/circuit/gateGrouping/gateGrouping.cpp:245: targets and/or controls are not included in wireIdOrdering
```

The first circuit (20 single-qubit X/Z terms only) always succeeds. The warmup circuit (1 qubit) also succeeds.

### Circuits That Work vs Fail

**Works on mgpu**:
- TFIM Hamiltonians (primarily ZZ and X terms — nearest-neighbor, low gate complexity)
- QFT circuits
- Warmup circuits
- Circuit 1 of FH_D-1 (single-qubit X/Z Pauli terms only)

**Fails on mgpu**:
- FH_D-1 circuits 2-5 (multi-qubit YY, XXX, YYXXX terms with wide qubit spans)
- Likely affects other complex Hamiltonians with long-range multi-qubit Pauli terms

The failing circuits involve basis rotation gates for multi-qubit Pauli measurements (e.g., Y-basis rotations, multi-qubit X chains) that span wide ranges of qubits. When the statevector is partitioned across GPUs, these gates may span GPU boundaries in ways the gate-grouping algorithm cannot handle.

### Workaround

Use `--parallel` (`-pm`) when running with MPI on multi-GPU systems:

```bash
# Workaround: add -pm to use single-GPU per rank (parallel mode)
srun -n 4 python -m mpi4py -m qedcbench.hamlib.hamlib_simulation_benchmark \
    -a cudaq -obs -nop -nod -n 20 -gm simple -ham FH_D-1 -ds -pm -s 10000 -v
```

This sacrifices the ability to simulate larger-than-single-GPU circuits but avoids the gate-grouping bug. For observable estimation workloads (many moderate-width circuits), parallel mode is typically the better choice anyway.

### Open Questions

1. Was this working in earlier CUDA-Q versions (pre-0.13.0)? Previous papers include results from distributed statevector runs on Perlmutter, but it is unclear whether those runs used Hamiltonians with the same multi-qubit gate patterns that trigger this bug.

2. Is the issue specific to the BK (Bravyi-Kitaev) encoding, which produces longer-range Pauli terms? Would JW (Jordan-Wigner) encoding produce circuits that work on mgpu?

3. Does the number of GPUs matter? (e.g., does it work with 2 GPUs but fail with 4?)

4. Is this a known issue with cuStateVec / cusvsim, or should it be reported to NVIDIA?

<br>
&copy; 2025 Quantum Economic Development Consortium (QED-C). All Rights Reserved.
