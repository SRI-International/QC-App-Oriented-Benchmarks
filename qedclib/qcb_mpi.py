# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##########################
# QCB MPI Module
#
# This module allows for MPI support when the mpi4py module is loaded from
# the python command-line. When the module is not loaded, the same 
# data and functions are available for single task execution without MPI
#

import os
import sys

rank = 0
size = 1
initialized = False

# QPU group state — populated once by init_qpus()
qpu_id = None          # which QPU group this rank belongs to
qpu_rank = None        # rank within the QPU group
num_qpus = None        # total number of QPU groups
is_qpu_leader = False  # True if qpu_rank == 0
_qpu_comm = None       # subcommunicator for this QPU group (internal — use get_qpu_handle())
leaders_comm = None    # subcommunicator across QPU leaders (mpi4py Comm or None)

##########################################################################
# MPI is enabled by adding by loading the mpi4py module
# - This can be accomplished, for example, with
# - "-m mpy4py" as an argument to python
# If MPI is not enabled, wrapper functions provide the same
# functionality without any MPI calls
##########################################################################
if "mpi4py" not in sys.modules:
    def enabled():
        return False

    # Return true if this task is rank 0
    def leader():
        return True
    
    # Synchronize all MPI Tasks
    def barrier():
        return

    # Broadcast data from leader
    def bcast(data):
        return data

    # Gather data from all ranks to root (no-op without MPI)
    def gather(data, root=0):
        return [data]

    # Scatter data from root to all ranks (no-op without MPI)
    def scatter(data, root=0):
        return data[0] if isinstance(data, list) and len(data) > 0 else data

    def init_qpus(gpus_per_circuit):
        """No-op stub — hybrid QPU mode requires MPI."""
        pass

    def get_qpu_handle():
        """No-op stub — returns None without MPI."""
        return None

    # Initialize this module
    # -- stdout is redirected to null if not the leader
    def init():
        global rank, size, leader, initialized
        if initialized:
            return

        #print("Initializing MPI...No MPI Module Loaded",flush=True)
        rank = 0
        size = 0
        initialized = True

else:
    from mpi4py import MPI
    import atexit

    def enabled():
        return True

    def leader():
        global initialized, rank
        if initialized is False:
            raise Exception("MPI call before init")
        if rank == 0:
            return True
        else:
            return False

    def barrier():
        global initialized
        if initialized is False:
            raise Exception("MPI call before init")
        MPI.COMM_WORLD.barrier()
        return

    def bcast(data):
        return MPI.COMM_WORLD.bcast(data, root=0)

    def gather(data, root=0):
        """Gather data from all ranks to root."""
        global initialized
        if initialized is False:
            raise Exception("MPI call before init")
        return MPI.COMM_WORLD.gather(data, root=root)

    def scatter(data, root=0):
        """Scatter data from root to all ranks."""
        global initialized
        if initialized is False:
            raise Exception("MPI call before init")
        return MPI.COMM_WORLD.scatter(data, root=root)

    def init_qpus(gpus_per_circuit):
        """
        Split COMM_WORLD into groups of `gpus_per_circuit` ranks (QPUs).

        Called once after mpi.init(). Idempotent — subsequent calls with the
        same gpus_per_circuit are no-ops. Raises ValueError on mismatch.
        Does nothing when gpus_per_circuit is None or < 1.
        """
        global qpu_id, qpu_rank, num_qpus, is_qpu_leader, _qpu_comm, leaders_comm

        if gpus_per_circuit is None or gpus_per_circuit < 1:
            return

        # Already initialized — validate consistency
        if _qpu_comm is not None:
            expected = MPI.COMM_WORLD.Get_size() // gpus_per_circuit
            if num_qpus != expected:
                raise ValueError(
                    f"init_qpus called with gpus_per_circuit={gpus_per_circuit} "
                    f"but was previously initialized for num_qpus={num_qpus}")
            return

        world_comm = MPI.COMM_WORLD
        world_rank = world_comm.Get_rank()
        world_size = world_comm.Get_size()

        num_qpus = world_size // gpus_per_circuit
        qpu_id = world_rank // gpus_per_circuit
        _qpu_comm = world_comm.Split(color=qpu_id, key=world_rank)
        qpu_rank = _qpu_comm.Get_rank()
        is_qpu_leader = (qpu_rank == 0)

        # key=qpu_id ensures leaders_comm.gather() returns blocks in QPU order
        leaders_comm = world_comm.Split(
            color=0 if is_qpu_leader else MPI.UNDEFINED,
            key=qpu_id,
        )

        if world_rank == 0:
            print(f"... init_qpus: {world_size} ranks → {num_qpus} QPUs "
                  f"× {gpus_per_circuit} GPUs each", flush=True)

    def get_qpu_handle():
        """Return the raw C integer handle for the QPU subcommunicator."""
        return MPI._addressof(_qpu_comm)

    def finalize():
        global rank, initialized

        # Close null file used for capturing stdout
        if rank > 0 and initialized:
            sys.stdout.close()
        initialized = False

    def init():
        global rank, size, initialized
        if initialized:
            return
        
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            print("Initializing MPI...",flush=True,end='')
        size = MPI.COMM_WORLD.Get_size()
        MPI.COMM_WORLD.barrier()
        initialized = True

        # Capture duplicate output from non-leader ranks
        if rank == 0:
            print("Using",size,"tasks",flush=True)
        else:
            f = open(os.devnull, 'w')
            sys.stdout = f

        atexit.register(finalize)
        return
