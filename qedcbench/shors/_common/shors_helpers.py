"""Shared analytical helpers for Shor's order-finding benchmark."""

import numpy as np


def expected_shor_dist(num_bits, order, num_shots):
    """Return the exact finite-register order-finding distribution."""
    qubits_measured = 2 * num_bits
    r = int(order)
    if r < 1:
        raise ValueError(f"order must be positive, got {order}")

    q = 1 << qubits_measured
    short_length, long_count = divmod(q, r)
    outcomes = np.arange(q, dtype=np.int64)
    residues = (r * outcomes) % q
    half_angles = np.pi * residues / q
    denominators = np.sin(half_angles)

    def geometric_magnitude(length):
        magnitudes = np.empty(q, dtype=np.float64)
        exact_peak = residues == 0
        exact_zero = (~exact_peak) & (((length * residues) % q) == 0)
        regular = ~(exact_peak | exact_zero)

        magnitudes[exact_peak] = float(length * length)
        magnitudes[exact_zero] = 0.0
        magnitudes[regular] = (
            np.sin(length * half_angles[regular]) / denominators[regular]
        ) ** 2
        return magnitudes

    probabilities = (
        long_count * geometric_magnitude(short_length + 1)
        + (r - long_count) * geometric_magnitude(short_length)
    ) / float(q * q)

    probabilities /= probabilities.sum()
    scale = float(num_shots)
    return {
        format(outcome, f"0{qubits_measured}b"): float(probability * scale)
        for outcome, probability in enumerate(probabilities)
        if probability > 0.0
    }
