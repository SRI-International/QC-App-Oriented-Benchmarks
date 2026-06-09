# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (05/2020)
# Contributor: Adrien Suau (<adrien.suau@cerfacs.fr>
#                           <adrien.suau@lirmm.fr>)
#              Siyuan Niu (<siyuan.niu@lirmm.fr>)
# This software is governed by the CeCILL-B license under French law and
# abiding  by the  rules of  distribution of free software. You can use,
# modify  and/or  redistribute  the  software  under  the  terms  of the
# CeCILL-B license as circulated by CEA, CNRS and INRIA at the following
# URL "http://www.cecill.info".
#
# As a counterpart to the access to  the source code and rights to copy,
# modify and  redistribute granted  by the  license, users  are provided
# only with a limited warranty and  the software's author, the holder of
# the economic rights,  and the  successive licensors  have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using, modifying and/or  developing or reproducing  the
# software by the user in light of its specific status of free software,
# that  may mean  that it  is complicated  to manipulate,  and that also
# therefore  means that  it is reserved for  developers and  experienced
# professionals having in-depth  computer knowledge. Users are therefore
# encouraged  to load and  test  the software's  suitability as  regards
# their  requirements  in  conditions  enabling  the  security  of their
# systems  and/or  data to be  ensured and,  more generally,  to use and
# operate it in the same conditions as regards security.
#
# The fact that you  are presently reading this  means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.
# ======================================================================

from copy import deepcopy
import time
import pickle

from qiskit.providers.ibmq import IBMQBackend
from qiskit import QuantumCircuit, execute, transpile, assemble

from qiskit.transpiler.passes.layout import SetLayout, ApplyLayout
from qiskit.transpiler.passes.basis.unroller import Unroller
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler import Layout
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers import level_0_pass_manager
from qiskit.transpiler import CouplingMap

class IBMQSubmitter:
    def __init__(self, backend: IBMQBackend, tags=None):
        self._circuits = []
        self._backend = backend
        backend_configuration = backend.configuration()
        self._maximum_batch_job = backend_configuration.max_experiments
        self._maximum_shots = backend_configuration.max_shots

        self._circuits_results = []
        self._job_ids = []
        if tags is None:
            tags = []
        self._tags = tags

    def __len__(self):
        return len(self._circuits)

    def unroll_and_map_circuit(
        self, circuit: QuantumCircuit, mapping, backend
    ) -> QuantumCircuit:
        layout = Layout({q: i for q, i in zip(circuit.qubits, mapping)})
        pm = PassManager(
            [
                SetLayout(layout),
                ApplyLayout(),
                Unroller(self._backend.configuration().basis_gates),
            ]
        )
        # pm = level_0_pass_manager(PassManagerConfig(
        #     initial_layout = layout,
        #     basis_gates=backend.configuration().basis_gates,
        #     coupling_map=CouplingMap(backend.configuration().coupling_map),
        #     backend_properties=backend.properties()
        # ))
        return pm.run(circuit)

    def add_circuit(self, circuit: QuantumCircuit, initial_mapping, backend):
        self._circuits.append(self.unroll_and_map_circuit(circuit, initial_mapping, backend))

    def _wait_for_first_job_to_complete(self, jobs):
        print("Waiting for job... ", end="", flush=True)
        jobs[0].wait_for_final_state()
        print("Done!")

    def submit(self):
        running_jobs = []
        configuration = self._backend.configuration()
        max_circuits_per_job = configuration.max_experiments
        for i in range(0, len(self._circuits), max_circuits_per_job):
            # Submit some jobs
            down = i * max_circuits_per_job
            up = max(len(self._circuits), (i + 1) * max_circuits_per_job)
            qobj = assemble(
                self._circuits[down:up],
                backend=self._backend,
                shots=configuration.max_shots,
            )
            job = self._backend.run(qobj, job_tags=self._tags)
            running_jobs.append(job)
            # If we have too much submitted jobs, wait for the first one to finish.
            if len(running_jobs) == self._maximum_batch_job:
                self._wait_for_first_job_to_complete(running_jobs)
                self._circuits_results.append(running_jobs[0].result())
                self._job_ids.append(running_jobs.pop(0).job_id())
        # Wait for the last jobs to finish
        while running_jobs:
            self._wait_for_first_job_to_complete(running_jobs)
            self._circuits_results.append(running_jobs[0].result())
            self._job_ids.append(running_jobs.pop(0).job_id())

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump([self._job_ids, self._circuits_results], f)
            print(f"Saved in '{filename}'.")
