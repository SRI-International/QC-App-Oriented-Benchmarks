# ======================================================================
# MIT License
#
# Copyright (c) [2020] [LIRMM]
# Contributor: Siyuan Niu (<siyuan.niu@lirmm.fr>)
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ======================================================================
from hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
import typing as ty
import logging

logger = logging.getLogger("submit2")


def zi_iz_measure_contribution(zi_and_iz_counts, shots):
    zi_contribution = (zi_and_iz_counts.get('00', 0) + zi_and_iz_counts.get('10', 0)
                       - zi_and_iz_counts.get('11', 0) - zi_and_iz_counts.get('01', 0)) / shots
    iz_contribution = (zi_and_iz_counts.get('00', 0) + zi_and_iz_counts.get('01', 0)
                       - zi_and_iz_counts.get('10', 0) - zi_and_iz_counts.get('11', 0)) / shots
    return zi_contribution, iz_contribution

def xx_yy_measure_contribution(xx_and_yy_counts, shots):
    xx_contribution = (xx_and_yy_counts.get('00', 0) + xx_and_yy_counts.get('10', 0)
                       - xx_and_yy_counts.get('11', 0) - xx_and_yy_counts.get('01', 0)) / shots
    zz_contribution = (xx_and_yy_counts.get('00', 0) + xx_and_yy_counts.get('01', 0)
                       - xx_and_yy_counts.get('10', 0) - xx_and_yy_counts.get('11', 0)) / shots
    yy_contribution = -xx_contribution * zz_contribution
    return xx_contribution, yy_contribution


def submit_circuits(hardware: IBMQHardwareArchitecture,
                   initial_layouts: ty.List,
                   final_circuits: ty.List,
                   ):
    print("[INFO] Hardware execution skipped (no IBMQ account). Returning None.")
    return None


def pauli_measure_multiprogram(counts, start, end):
    result_00, result_01, result_10, result_11 = 0, 0, 0, 0
    for result, count in counts.items():
        if result[start:end] == "00":
            result_00 += count
        elif result[start:end] == "01":
            result_01 += count
        elif result[start:end] == "10":
            result_10 += count
        elif result[start:end] == "11":
            result_11 += count
    return result_00, result_01, result_10, result_11


def result_fidelity(hardware: IBMQHardwareArchitecture,
                   initial_layouts: ty.List,
                   final_circuits: ty.List,
                   partitions: ty.List[ty.List]):
    print(f"\n[Routing complete] {len(final_circuits)} circuit(s) mapped.")
    for i, circuit in enumerate(final_circuits):
        cx_count = circuit.count_ops().get('cx', 0)
        swap_count = circuit.count_ops().get('swap', 0)
        print(f"  Circuit {i+1}: {circuit.num_qubits} qubits, "
              f"{cx_count} CX gates, {swap_count} SWAP gates added")
    print("[INFO] Hardware execution skipped (no IBMQ account).")


def energy_result(
        hardware: IBMQHardwareArchitecture,
        initial_layouts: ty.List,
        final_circuits: ty.List,
        partitions: ty.List[ty.List],
        ansatz_parameter: ty.List,
        shots = 8192,
):
    print(f"\n[Routing complete] {len(final_circuits)} circuit(s) mapped.")
    for i, circuit in enumerate(final_circuits):
        cx_count = circuit.count_ops().get('cx', 0)
        swap_count = circuit.count_ops().get('swap', 0)
        print(f"  Circuit {i+1}: {circuit.num_qubits} qubits, "
              f"{cx_count} CX gates, {swap_count} SWAP gates added")
    print("[INFO] Hardware execution skipped (no IBMQ account).")
