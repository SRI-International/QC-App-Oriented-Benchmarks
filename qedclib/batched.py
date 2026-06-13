"""
Batched circuit creation and execution.

When max_batch_size is set, alternates between creating circuits (one qubit
width at a time) and executing them, keeping at most max_batch_size circuits
in memory.  Groups are never split — a new group is only started if it fits
within the remaining batch budget.
"""

import inspect
from qedclib import metrics


def batched_run(get_circuits_fn, run_circuits_fn, plot_results_fn, **kwargs):
    """Alternate circuit creation and execution in group-sized batches.

    Accepts the same kwargs as run() — they are routed to get_circuits,
    run_circuits, and plot_results by inspecting each function's signature.
    """
    max_batch_size = kwargs.get('max_batch_size')
    min_qubits = kwargs.get('min_qubits', 2)
    max_qubits = kwargs.get('max_qubits', 8)
    skip_qubits = kwargs.get('skip_qubits', 1)
    verbose = kwargs.get('verbose', False)

    # Partition incoming arguments to the function that accepts them
    def _for(func):
        return {k: kwargs[k] for k in kwargs
                if k in inspect.signature(func).parameters}

    gc_args = _for(get_circuits_fn)
    rc_args = _for(run_circuits_fn)

    metrics.init_metrics()

    max_circuits = kwargs.get('max_circuits', 3)
    accumulated_qcs = {}
    accumulated_count = 0
    accumulated_widths = []

    for w in range(min_qubits, max_qubits + 1, skip_qubits):
        # If next group would likely exceed batch, execute what we have first
        if accumulated_count > 0 and accumulated_count + max_circuits > max_batch_size:
            #if verbose:
            print(f"... batched_run: executing widths {accumulated_widths} ({accumulated_count} circuits)")
            run_circuits_fn(accumulated_qcs, **rc_args)
            accumulated_qcs = {}
            accumulated_count = 0
            accumulated_widths = []

        # Generate circuits for one qubit width
        qcs, _ = get_circuits_fn(**{**gc_args, 'min_qubits': w, 'max_qubits': w})
        if not qcs:
            continue

        group_count = sum(len(v) for v in qcs.values() if isinstance(v, dict))
        accumulated_qcs.update(qcs)
        accumulated_count += group_count
        accumulated_widths.append(w)

    # Execute any remaining circuits
    if accumulated_qcs:
        print(f"... batched_run: executing widths {accumulated_widths} ({accumulated_count} circuits)")
        run_circuits_fn(accumulated_qcs, **rc_args)

    metrics.end_metrics()
    plot_results_fn(**_for(plot_results_fn))
