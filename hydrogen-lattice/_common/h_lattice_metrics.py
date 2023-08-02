
###############################################################################
# (C) Quantum Economic Development Consortium (QED-C) 2021.
# Technical Advisory Committee on Standards and Benchmarks (TAC)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
##########################
# H-Lattice Metrics Module
#
# This module contains methods to initialize, store, aggregate and
# plot metrics collected in the benchmark programs for the H-Lattice
#
# Metrics are indexed by group id (e.g. circuit_id size), circuit_id id (e.g. secret string)
# and metric name.
# Multiple metrics with different names may be stored, each with a single metric_value.
#
# The 'aggregate' method accumulates all circuit_id-specific metrics for each group
# and creates an average that may be reported and plotted across all groups
#

import os
import json
import time
from time import gmtime, strftime
from datetime import datetime
import traceback
import matplotlib.cm as cm
import copy
import matplotlib.pyplot as plt
import numpy as np

# add to the path variable the path to the metrics module
import sys

sys.path[1:1] = [ "_common", "_common/qiskit", "hydrogen-lattice/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit", "../../hydrogen-lattice/_common/" ]


# import the metrics module
import metrics as metrics

h_lattice_metrics = metrics.circuit_metrics_detail


# save plot images flag
save_plot_images = True

# function to take input the title of plot, the x and y axis labels, and the data to plot as a line plot
def plot_line_metric(suptitle="Circuit Width (Number of Qubits)", metric_name="energy", x_val='cumulative_exec_time'):
    
    # get subtitle from metrics
    subtitle = metrics.circuit_metrics['subtitle']

    # get backend id from subtitle
    backend_id = subtitle[9:]

    # set the full title
    fulltitle = suptitle + f" " + metrics.known_score_labels[metric_name] + f"\nDevice={backend_id}  {metrics.get_timestr()}"
    fulltitle = f"{metrics.known_score_labels[metric_name]} vs. {metrics.known_x_labels[x_val]} "

    

    # TODO: add error handling for invalid metric_name and x_val inputs

    # check if x_val starts with cumulative and turn cumulatifve flag on
    if x_val.startswith('cumulative'):
        x_metric_name = x_val[11:]
        cumulative_flag = True

    else:
        x_metric_name = x_val
        cumulative_flag = False
    

    # iterate over all groups
    for group in h_lattice_metrics:

        num_qubits = int(group)
        # iterate over all circuits

        circuit_ids = [int(x) for x in (h_lattice_metrics[group])]


        total_instances = int(np.floor(circuit_ids[-1]/1000))
        

        for instance in range(1, total_instances + 1):

            # define the figure and axis
            fig, ax = plt.subplots(1,1)
        
            y_data = []
            x_data = []

            current_radius = 0

            for circuit_id in h_lattice_metrics[group]:
                if np.floor(int(circuit_id)/1000) == instance:
                    # get the metric value
                    metric_value = h_lattice_metrics[group][circuit_id][metric_name]
                    # set the y label 
                    y_label = metrics.known_score_labels[metric_name]

                    # append the metric value to the y_data list
                    y_data.append(metric_value)

                    # get the x metric value
                    x_metric_value = h_lattice_metrics[group][circuit_id][x_metric_name]
                    # set the x label
                    x_label = metrics.known_x_labels[x_val]

                    # append the x metric value to the x_data list
                    x_data.append(x_metric_value)

                    doci_energy = h_lattice_metrics[group][circuit_id]['doci_energy']
                    fci_energy = h_lattice_metrics[group][circuit_id]['fci_energy']
                    current_radius = h_lattice_metrics[group][circuit_id]['radius']

                else:
                    continue


        #if x_val not in known_x_labels:
        # for circuit_id in metrics.circuit_metrics[group]:
            

        #     # get x_metric_value from metrics module
        #     x_metric_value = metrics.circuit_metrics[group][circuit_id][x_metric_name]
        #     # set the x label from metrics module
        #     x_label = metrics.known_x_labels[x_val]

        #     # append the x metric value to the x_data list
        #     x_data.append(x_metric_value)

        # else:
            
            

            #print("y metric name is " + str(y_label) + " and the y data is " + str(y_data) + "\n and the x metric name is " + str(x_label) + " and the x data is " + str(x_data))

            # make the x_data cumulative if the cumulative flag is on
            if cumulative_flag:
                x_data = cumulative_sum(x_data)

            # plot the data as a line with the color of line depending on the y value
            #ax.plot(x_data, y_data, linestyle='solid', linewidth=2, markersize=12)

            # plot the data as a scatter plot with the color of the point depending on the y value, the scatter points are connected with a line
            # plot if the metric is solution quality invert the color map
            if metric_name == 'solution_quality':
                ax.scatter(x_data, y_data, c=y_data, cmap=cm.coolwarm_r)
            else:
                ax.scatter(x_data, y_data, c=y_data, cmap=cm.coolwarm)

            ax.plot(x_data, y_data, linestyle='-.', linewidth=2, markersize=12)


            # set the title
            ax.set_title(fulltitle)

            ax.set_xlabel(x_label)

            ax.set_ylabel(y_label)

            # if the x metric is iteration count, set x ticks to be integers
            if x_metric_name == 'iteration_count':
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # if the score metric is energy or solution quality, plot the FCI and DOCI energy lines
            if metric_name == 'energy':
                ax.axhline(y=doci_energy, color='r', linestyle='--', label='DOCI Energy for given Hamiltonian')
                ax.axhline(y=fci_energy, color='g', linestyle='solid', label='FCI Energy for given Hamiltonian')
            energy_text = f'DOCI Energy: {doci_energy:.2f} | FCI Energy: {fci_energy:.2f} | Num of Qubits: {num_qubits} | Radius: {current_radius}'
            ax.annotate(energy_text, xy=(0.5, 0.97), xycoords='figure fraction', ha='center', va='top')
            
            

            ax.legend()
            fig.show()
            

            if save_plot_images:
                metrics.save_plot_image(plt, os.path.join(f"Hydrogen-Lattice-line-"
                                                    + str(group) + '-'
                                                    + str(instance) + '-'
                                                    + str(metric_name) + '-'
                                                    + str(x_metric_name) + '-'), backend_id)
    plt.show(block=True)



# function to plot all line metrics
def plot_all_line_metrics(score_metrics="energy", x_vals='cumulative_exec_time'):
    # if score_metrics and x_val are strings, convert to lists
    if type(score_metrics) is str:
        score_metrics = [score_metrics]
    if type(x_vals) is str:
        x_vals = [x_vals]
    
    # iterate over score metrics and plot each
    for score_metric in score_metrics:
        for x_val in x_vals:
            plot_line_metric(suptitle="Hydrogen Lattice (Number of Qubits)", metric_name=score_metric, x_val=x_val)

# function to input a list of float and return a list of cumulative sums
def cumulative_sum(input_list: list):
    output_list = []
    sum = 0
    for item in input_list:
        sum += item
        output_list.append(sum)
    return output_list