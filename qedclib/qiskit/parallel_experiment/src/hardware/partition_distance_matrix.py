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
import numpy as np

def adj_matrix_construct(
        hardware: IBMQHardwareArchitecture,
        original_distance_matrix: ty.Callable[
            [IBMQHardwareArchitecture], np.ndarray,
        ],
        partition: ty.List = None,
        ) -> np.ndarray:
    """
    Construct the adjacent matrix according to the given original distance matrix
    :param hardware: selected hardware
    :param partition: allocated partition
    :param original_distance_matrix: distance matrix calculated according to the architecture of the hardware,
           there are different types of distance matrices, such as swap number, error rate or the combination
           of them.
    :return: The constructed adjacent matrix
    """
    if partition == None:
        partition = [i for i in range(hardware.qubit_number)]
    partition_pairs = []
    for i in partition:
        for j in partition:
            if (i,j) in hardware.edges:
                partition_pairs.append((i,j))
    qubit_num = hardware.qubit_number
    adj_mat = np.zeros((qubit_num, qubit_num))

    distance_matrix = original_distance_matrix(hardware)
    for (i,j) in partition_pairs:
        adj_mat[i][j] = distance_matrix.item((i,j))
    return adj_mat


def partition_distance_matrix(qubit_num: int,
          adj_mat: np.ndarray
          ) -> np.ndarray:
    """
    Calculate the distance matrix after the partition by Floyd-Warshall algorithm.
    """
    distance_mat = np.zeros((qubit_num, qubit_num))

    for i in range(qubit_num):
        for j in range(qubit_num):
            if adj_mat.item((i,j)) != 0:
                distance_mat[i][j] = adj_mat.item((i,j))
            else:
                distance_mat[i][j] = 1000000000
        distance_mat[i][i] = 0

    for k in range(qubit_num):
        for i in range(qubit_num):
            for j in range(qubit_num):
                if distance_mat[i][j] > distance_mat[i][k] + distance_mat[k][j]:
                    distance_mat[i][j] = distance_mat[i][k] + distance_mat[k][j]

    return distance_mat

