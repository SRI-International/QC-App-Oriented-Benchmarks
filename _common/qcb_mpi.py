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

    # Initialize this module
    # -- stdout is redirected to null if not the leader
    def init():
        global rank, size, leader, initialized
        if initialized:
            return
        
        print("Initializing MPI...No MPI Module Loaded",flush=True)        
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
