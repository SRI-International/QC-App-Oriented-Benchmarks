"""Regression tests for Shor's order and ideal-output fidelity."""

import math
import sys
from pathlib import Path


benchmark_root = str(Path(__file__).resolve().parents[1])
repo_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, benchmark_root)
sys.path.insert(0, repo_root)

from qedcbench.shors._common.shors_helpers import expected_shor_dist
from qedcbench.shors._common.shors_utils import multiplicative_order
from qedclib import metrics


def test_multiplicative_order_uses_smallest_exponent():
    # 9**105 == 1 (mod 196), but 105 is not the multiplicative order.
    assert pow(9, 105, 196) == 1
    assert multiplicative_order(9, 196) == 21


def test_exact_distribution_for_power_of_two_order():
    distribution = expected_shor_dist(num_bits=8, order=2, num_shots=1000)
    assert distribution == {
        "0000000000000000": 500.0,
        "1000000000000000": 500.0,
    }


def test_finite_qpe_distribution_includes_off_peak_probability():
    distribution = expected_shor_dist(num_bits=2, order=3, num_shots=100)

    assert len(distribution) == 16
    assert math.isclose(sum(distribution.values()), 100.0, rel_tol=1e-12)
    assert math.isclose(distribution["0000"], 33.59375, rel_tol=1e-12)
    assert distribution["0110"] > 0.0


def test_distribution_has_unit_fidelity_with_itself():
    distribution = expected_shor_dist(num_bits=4, order=21, num_shots=1000)
    fidelity = metrics.polarization_fidelity(distribution, distribution)

    assert math.isclose(fidelity["hf_fidelity"], 1.0, abs_tol=1e-12)
    assert math.isclose(fidelity["fidelity"], 1.0, abs_tol=1e-12)


if __name__ == "__main__":
    test_multiplicative_order_uses_smallest_exponent()
    test_exact_distribution_for_power_of_two_order()
    test_finite_qpe_distribution_includes_off_peak_probability()
    test_distribution_has_unit_fidelity_with_itself()
    print("Shor fidelity tests passed")
