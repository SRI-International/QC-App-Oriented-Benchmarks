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
from qiskit import Aer, IBMQ
from qiskit.providers.jobstatus import JobStatus
from tools.IBMQSubmitter import IBMQSubmitter

import logging
logger = logging.getLogger("submit")

def get_jobs(backend, tags):
    jobs = backend.jobs(
        job_tags=tags, job_tags_operator="AND", limit=1, descending=False
    )
    return jobs

def submit_circuits(hardware: IBMQHardwareArchitecture,
                   initial_layouts: ty.List,
                   final_circuits: ty.List,
                   circuit_tag: ty.List):

    print("Loading account...")
    IBMQ.load_account()
    provider = IBMQ.get_provider(
        hub="ibm-q-france", group="univ-montpellier", project="default"
    )

    backend = provider.get_backend(f"{hardware.name}")
    print(f"Running on {backend.name()}.")
    #backend = provider.get_backend("ibmq_qasm_simulator")

    submitter = IBMQSubmitter(backend, tags=circuit_tag)

    for index, circuit in enumerate(final_circuits):
        circuit.measure_active()
        submitter.add_circuit(
            circuit, initial_layouts[index], backend
        )
    print(f"Submitting {len(submitter)} circuits...")
    submitter.submit()
    jobs = get_jobs(backend, circuit_tag)
    return jobs

def result_fidelity(
        hardware: IBMQHardwareArchitecture,
        initial_layouts: ty.List,
        final_circuits: ty.List,
        partitions: ty.List[ty.List],
        circuit_tag: ty.List,
):
    fidelities = []
    jobs = submit_circuits(hardware, initial_layouts, final_circuits, circuit_tag)
    for i, job in enumerate(jobs):
        if job.status() == JobStatus.ERROR:
            logger.error("job error")
            exit(1)
        qasm_simulator = Aer.get_backend('qasm_simulator')
        qobj = job.qobj()
        ideal_results = qasm_simulator.run(qobj).result()
        ideal_counts = ideal_results.get_counts()
        hardware_results = job.result()
        hardware_counts = hardware_results.get_counts()
        circuit_number = len(qobj.experiments)
        for num in range(circuit_number):
            ideal_result = list(ideal_counts[num].keys())[0]
            if partitions[num] is None or not isinstance(partitions[num], list):
                fidelities.append(
                    hardware_counts[num].get(ideal_result, 0) / ideal_counts[num][ideal_result]
                )
            else:
                fidelity_merge = []
                start = len(ideal_result)
                for partition in (partitions[num]):
                    start -=  len(partition)
                    end = start + len(partition)
                    fidelity = 0
                    for result, count in hardware_counts[num].items():
                        if result[start:end] == ideal_result[start:end]:
                            fidelity += count
                    fidelity = fidelity / ideal_counts[num][ideal_result]
                    fidelity_merge.append(fidelity)
                fidelities.append(fidelity_merge)
        print(fidelities)



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

def energy_result(
        hardware: IBMQHardwareArchitecture,
        initial_layouts: ty.List,
        final_circuits: ty.List,
        partitions: ty.List[ty.List],
        circuit_tag: ty.List,
        ansatz_parameter = [5.9, 0.22, -6.1, -2.14, -2.14],
        shots = 8192,
):
    jobs = submit_circuits(hardware, initial_layouts, final_circuits, circuit_tag)
    final_result_simultaneous = []
    energy_result_multiprogram = []

    for i, job in enumerate(jobs):
        if job.status() == JobStatus.ERROR:
            logger.error("job error")
            exit(1)
        qobj = job.qobj()
        hardware_results = job.result()
        hardware_counts = hardware_results.get_counts()
        circuit_number = len(qobj.experiments)
        for num in range(circuit_number // 2):
            zi_contribution, iz_contribution = zi_iz_measure_contribution(hardware_counts[num], shots)
            xx_contribution, yy_contribution = xx_yy_measure_contribution(hardware_counts[num + circuit_number // 2], shots)
            result_hardware = ansatz_parameter[0] + ansatz_parameter[1] * zi_contribution + ansatz_parameter[
                2] * iz_contribution + \
                              ansatz_parameter[3] * xx_contribution + ansatz_parameter[4] * yy_contribution

            final_result_simultaneous.append(result_hardware)
        print("Simultaneous measurement result:", final_result_simultaneous)

        start1 = len(list(hardware_counts[-1].keys())[0])
        for i in range(len(partitions) // 2):
            end1 = start1
            start1 -= len(partitions[i])
            result_00, result_01, result_10, result_11 = pauli_measure_multiprogram(hardware_counts[-1], start1, end1)
            zi_contribution = (result_00 + result_10 - result_01 - result_11) / shots
            iz_contribution = (result_00 + result_01 - result_10 - result_11) / shots

            start2 = start1 - len(partitions[i]) * (len(partitions) // 2)
            end2 = start2 + len(partitions[i])
            result_00_2, result_01_2, result_10_2, result_11_2 = pauli_measure_multiprogram(hardware_counts[-1], start2, end2)
            xx_contribution = (result_00_2 + result_10_2 - result_01_2 - result_11_2) / shots
            zz_contribution = (result_00_2 + result_01_2 - result_10_2 - result_11_2) / shots
            yy_contribution = -xx_contribution * zz_contribution

            result_hardware_multiprogram = ansatz_parameter[0] + ansatz_parameter[1] * zi_contribution + \
                                           ansatz_parameter[2] * iz_contribution + \
                                           ansatz_parameter[3] * xx_contribution + ansatz_parameter[4] * yy_contribution
            energy_result_multiprogram.append(result_hardware_multiprogram)

        print("Multiprogramming - Simultaneous measurement result:", energy_result_multiprogram)
