
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
import matplotlib, matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# add to the path variable the path to the metrics module
import sys

sys.path[1:1] = [ "_common", "_common/qiskit", "hydrogen-lattice/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit", "../../hydrogen-lattice/_common/" ]


# import the metrics module
import metrics as metrics

h_lattice_metrics = metrics.circuit_metrics_detail

# save plot images flag
save_plot_images = True

#################################################

# function to plot all line metrics
def plot_all_line_metrics(score_metrics=["energy", "solution_quality", "accuracy_volume"],
        x_vals=["iteration_count", "cumulative_exec_time"],
        subplot=True,
        backend_id="UNKNOWN", options=None):
    '''
    Function to plot all line metrics (energy, solution quality, accuracy volume) vs different x values (iteration count, cumulative execution time)

    parameters:
    ----------
    score_metrics: list
        list of score metrics to plot

    x_vals: list
        list of x values to plot

    subplot: bool
        flag to plot all metrics on the same figure or on different figures
    '''

    # if score_metrics and x_val are strings, convert to lists
    if type(score_metrics) is str:
        score_metrics = [score_metrics]
    if type(x_vals) is str:
        x_vals = [x_vals]

    global h_lattice_metrics

    average_exec_time_per_iteration = []
    average_exec_time_per_iteration_error = []
    average_solution_quality = []
    average_accuracy_ratio = []
    average_solution_quality_error = []
    average_accuracy_ratio_error = []
    qubit_counts = []
    
    # Create standard title for all plots
    method = 2
    toptitle = f"Benchmark Results - Hydrogen Lattice ({method}) - Qiskit" + \
                f"\nDevice={backend_id}  {metrics.get_timestr()}"
    
    subtitle = ""
    
    '''
    if options != None:
        options_str = ''
        for key, value in options.items():
            if len(options_str) > 0: options_str += ', '
            options_str += f"{key}={value}"
        subtitle += f"\n{options_str}"
    '''
    
    # iterate over number of qubits and score metrics and plot each
    for qubit_count in h_lattice_metrics:

        num_qubits = qubit_count
        
        circuit_ids = [int(x) for x in (h_lattice_metrics[qubit_count])]
        total_instances = int(np.floor(circuit_ids[-1]/1000))

        exec_time_array = []
        sol_quality_array = []
        acc_ratio_array = []
    
        for instance in range(1, total_instances + 1):

            # search metrics store for final metrics for this group
            current_radius, doci_energy, fci_energy, energy, solution_quality, accuracy_ratio = find_last_metrics_for_group(qubit_count, instance)
            
            '''
            energy_text = f'Num Qubits: {num_qubits} \u00B7 Radius: {current_radius}  \u00B7\u00B7  DOCI Energy: {doci_energy:.3f} \u00B7 FCI Energy: {fci_energy:.3f} \u00B7 Energy: {energy:.3f} '
            
            energy_text = f'DOCI Energy={doci_energy:.3f}, FCI Energy={fci_energy:.3f}, Energy={energy:.3f}, , Accuracy={accuracy_ratio:.3f}'
            '''
            
            # create common title
            suptitle = toptitle + f"\nqubits={num_qubits}, radius={current_radius}, shots={options['shots']}"
            
            # create a figure for all plots in this group
            
            if subplot:
                fig, axs = plt.subplots(2, 2, figsize=(12, 9))
            else:
                fig, axs1 = plt.subplots(1, 1, figsize=(12, 9))
                axs = [axs1]
                
            #fig.suptitle(f"--- {qubit_count} qubit group ---" + "\n" + energy_text, fontsize=14)
            fig.suptitle(suptitle, fontsize=13, backgroundcolor='aliceblue')
                    
            plot_line_metric(suptitle=suptitle, subtitle=subtitle,
                metric_name="energy", x_val="iteration_count", num_qubits=qubit_count, instance=instance, ax=axs[0, 0], subplot=subplot)
            
            exec_time, _, __ = plot_line_metric(suptitle=suptitle, subtitle=subtitle,
                metric_name="energy", x_val="cumulative_exec_time", num_qubits=qubit_count, instance=instance, ax=axs[0, 1], subplot=subplot)
            
            _, sol_quality, __ = plot_line_metric(suptitle=suptitle, subtitle=subtitle,
                metric_name="solution_quality", x_val="cumulative_exec_time", num_qubits=qubit_count, instance=instance, ax=axs[1, 0], subplot=subplot)
            
            #plot_line_metric(suptitle=suptitle, subtitle=subtitle, metric_name="accuracy_volume", x_val="cumulative_exec_time", num_qubits=qubit_count, instance=instance, ax=axs[1, 1], subplot=subplot)
            
            _, __, acc_ratio = plot_line_metric(suptitle=suptitle, subtitle=subtitle,
                metric_name="accuracy_ratio", x_val="cumulative_exec_time", num_qubits=qubit_count, instance=instance, ax=axs[1, 1], subplot=subplot)
            
            # this appears to be unneeded
            #plt.subplots_adjust(top=0.88, hspace=0.10)
      
            # add padding below suptitle, and between plots, due to multi-line titles
            fig.tight_layout(pad=1.5, h_pad=2.0, w_pad=3.0)
            
            group = qubit_count
            
            image_name = (f"Hydrogen-Lattice-(2)-line-{group}-{instance}") + \
                            ("-all" if subplot else f"-{metric_name}-{x_metric_name}")  
            
            if save_plot_images:
                metrics.save_plot_image(plt, image_name, backend_id)
                                            
            if not subplot:
                plt.show(block=True)
            
            exec_time_array.append(exec_time)
            sol_quality_array.append(1 - sol_quality)
            acc_ratio_array.append(1 - acc_ratio)
            
        average_et = np.average(exec_time_array)
        error_et = np.std(exec_time_array)/np.sqrt(len(exec_time_array))
        average_ar = np.average(acc_ratio_array)
        error_ar = np.std(acc_ratio_array)/np.sqrt(len(acc_ratio_array))
        average_sq = np.average(sol_quality_array)
        error_sq = np.std(sol_quality_array)/np.sqrt(len(sol_quality_array))

        average_exec_time_per_iteration.append(average_et)
        average_exec_time_per_iteration_error.append(error_et)
        average_accuracy_ratio.append(average_ar * 100)
        average_accuracy_ratio_error.append(error_ar * 100)
        average_solution_quality.append(average_sq * 100)
        average_solution_quality_error.append(error_sq * 100)

        qubit_counts.append(qubit_count)

        if subplot:
            plt.show(block=True)

    # plot the cumulative execution time per iteration vs. number of qubits
    
    plot_cumulative_metrics(suptitle=suptitle, subtitle=subtitle,
            x_data1=qubit_counts,
            y_data1=average_exec_time_per_iteration,
            y_err1=average_exec_time_per_iteration_error,
            x_data2=qubit_counts,
            y_data2=average_accuracy_ratio,
            y_err2=average_accuracy_ratio_error)


# function to take input the title of plot, the x and y axis labels, and the data to plot as a line plot
def plot_line_metric(suptitle:str="Title", subtitle:str="",
        metric_name:str="energy", x_val:str='cumulative_exec_time',
        num_qubits:int=None, instance:str=None,
        ax=None, subplot:bool=False):
    
    # get subtitle from metrics
    m_subtitle = metrics.circuit_metrics['subtitle']

    # get backend id from metrics subtitle
    backend_id = m_subtitle[9:]
        
    # set the full title
    #fulltitle = f"{metrics.known_score_labels[metric_name]} vs. {metrics.known_x_labels[x_val]} "

    # TODO: add error handling for invalid metric_name and x_val inputs

    # check if x_val starts with cumulative and turn cumulatifve flag on
    if x_val.startswith('cumulative'):
        x_metric_name = x_val[11:]
        cumulative_flag = True

    else:
        x_metric_name = x_val
        cumulative_flag = False
    
    group = str(num_qubits)

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
            energy = h_lattice_metrics[group][circuit_id]['energy']

        else:
            continue
         
    # make the x_data cumulative if the cumulative flag is on
    if cumulative_flag:
        x_data = cumulative_sum(x_data)

    # calculate the return parameters here

    # calculate the exec time per iteration
    if x_val == 'cumulative_exec_time':
        exec_time_per_iteration = x_data[-1]/len(x_data)
    else:
        exec_time_per_iteration = 0

    # final solution quality after all iterations
    if metric_name == 'solution_quality':
        final_solution_quality = y_data[-1]
    else:
        final_solution_quality = 0

    # final accuracy ratio after all iterations
    if metric_name == 'accuracy_ratio':
        final_accuracy_ratio = y_data[-1]
    else:
        final_accuracy_ratio = 0

    # plot the data as a line with the color of line depending on the y value
    #ax.plot(x_data, y_data, linestyle='solid', linewidth=2, markersize=12)

    # plot the data as a scatter plot with the color of the point depending on the y value, the scatter points are connected with a line
    # plot if the metric is solution quality invert the color map
    if metric_name == 'solution_quality' or 'accuracy_ratio':
        ax.scatter(x_data, y_data, c=y_data, cmap=cm.coolwarm_r)
    else:
        ax.scatter(x_data, y_data, c=y_data, cmap=cm.coolwarm)

    ax.plot(x_data, y_data, color='darkblue', linestyle='-.', linewidth=2, markersize=12)

    # set the title   
    #ax.set_title(suptitle + "\n" + subtitle + f", qubits={num_qubits}", fontsize=12)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # if the x metric is iteration count, set x ticks to be integers
    if x_metric_name == 'iteration_count':
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # if the score metric is energy or solution quality, plot the FCI and DOCI energy lines
    if metric_name == 'energy':
        ax.axhline(y=doci_energy, color='r', linestyle='--', label=f'DOCI Energy = {doci_energy:.3f}')
        ax.axhline(y=fci_energy, color='g', linestyle='-.', label=f'FCI Energy    = {fci_energy:.3f}')
        metric_legend_label = f'Solution Energy = {energy:.3f}'
        
    # add a horizontal line at y=1 for solution quality
    elif metric_name == 'solution_quality':
        ax.axhline(y=1, color='r', linestyle='--', label='Ideal Solution')
        metric_legend_label = f'Solution Quality = {final_solution_quality:.3f}'
        
        # start the y-ticks of solution quality at 0 and end at 1
        ax.set_ylim([0, 1.1])

    # add a horizontal line at y=0 for accuracy ratio
    elif metric_name == 'accuracy_ratio':
        ax.axhline(y=1, color='r', linestyle='--', label='Ideal Solution')
        metric_legend_label = f'Accuracy Ratio = {final_accuracy_ratio:.3f}'

    # add a horizontal line at y=0 for accuracy volume
    elif metric_name == 'accuracy_volume':
        ax.axhline(y=0, color='r', linestyle='--', label='Ideal Solution')
        metric_legend_label = f'Accuracy Volume = {final_accuracy_volume:.3f}'
        
    #energy_text = f'DOCI Energy: {doci_energy:.2f} | FCI Energy: {fci_energy:.2f} | Num of Qubits: {num_qubits} | Radius: {current_radius}'
    #ax.annotate(energy_text, xy=(0.5, 0.97), xycoords='figure fraction', ha='center', va='top')
    
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    '''
    import copy
    print(handles[0])
    newhandle = copy.deepcopy(handles[0])
    newhandle.set_color('blue')
    handles.append(newhandle)
    '''
    import matplotlib
    newhandle = matplotlib.lines.Line2D([0,1],[0,1])
    newhandle.update_from(handles[0])
    newhandle.set_color('darkblue')
    handles.append(newhandle)
    labels.append(metric_legend_label)

    ax.legend(handles, labels)
  
    return exec_time_per_iteration, final_solution_quality, final_accuracy_ratio


# function to input a list of float and return a list of cumulative sums
def cumulative_sum(input_list: list):
    output_list = []
    sum = 0
    for item in input_list:
        sum += item
        output_list.append(sum)
    return output_list

# Find the last energy and radius associated with specific group of circuits
def find_last_metrics_for_group(num_qubits, instance):

    group = str(num_qubits)

    current_radius = 0
    doci_energy = 0
    fci_energy = 0

    # DEVNOTE: would it be easier to just get the last entry of array, rather than this loop?
    for circuit_id in h_lattice_metrics[group]:
        if np.floor(int(circuit_id)/1000) == instance:
        
            energy = h_lattice_metrics[group][circuit_id]['energy']
            doci_energy = h_lattice_metrics[group][circuit_id]['doci_energy']
            fci_energy = h_lattice_metrics[group][circuit_id]['fci_energy']
            current_radius = h_lattice_metrics[group][circuit_id]['radius']
            accuracy_ratio = h_lattice_metrics[group][circuit_id]['accuracy_ratio']
            solution_quality = h_lattice_metrics[group][circuit_id]['solution_quality']

        else:
            continue
            
    return current_radius, doci_energy, fci_energy, energy, accuracy_ratio, solution_quality
    
#################################################

# method to plot cumulative execution time vs. number of qubits
def plot_cumulative_metrics(suptitle="", subtitle="",
            x_data1:list=None, y_data1:list=None, y_err1:list=None, 
            x_data2:list=None, y_data2:list=None, y_err2:list=None):
    '''
    Function to plot cumulative metrics (accuracy ratio, execution time per iteration) over different number of qubits

    parameters:
    ----------
    x_data1: list
        list of x data for plot 1

    y_data1: list
        list of y data for plot 1

    y_err1: list
        list of y error data for plot 1

    x_data2: list
        list of x data for plot 2

    y_data2: list   
        list of y data for plot 2

    y_err2: list    
        list of y error data for plot 2
    '''

    # get subtitle from metrics
    m_subtitle = metrics.circuit_metrics['subtitle']

    # get backend id from subtitle
    backend_id = m_subtitle[9:]

    # create a figures for the plot
    fig1 = plt.figure()
    
    # plot xdata1 vs ydata1 on fig1
    ax1 = fig1.gca()
    ax1.plot(x_data1, y_data1, linestyle='solid', linewidth=2, markersize=12, marker='x')

    # set the title
    #ax1.set_title("Cumulative Execution Time per iteration vs. Number of Qubits")
    ax1.set_title(suptitle + "\n" + subtitle, fontsize=12)
    ax1.set_xlabel("Number of Qubits")
    ax1.set_ylabel("Cumulative Execution Time/ iteration (s)")

    # also plot a bar plot on the same figure
    ax1.bar(x_data1, y_data1, 0.75, alpha = 0.8, zorder = 3)
    ax1.grid(True, axis = 'y', color='silver', zorder = 0)

    # error bars for the bar plot
    ax1.errorbar(x_data1, y_data1, yerr=y_err1, ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5,ls='', marker = "D", markersize = 8, mfc = 'c', mec = 'k', mew = 0.5,label = 'Error', alpha = 0.75, zorder = 5)

    # save the plot image
    if save_plot_images:
        metrics.save_plot_image(plt, os.path.join(f"Hydrogen-Lattice-(2)" +
                                            "-" + "avg_exec_time_per_iteration"),
                                            backend_id)                                         
    # create a figures for the plot
    fig2 = plt.figure()
    
    # plot xdata2 vs ydata2 on fig2
    ax2 = fig2.gca()
    ax2.plot(x_data2, y_data2, linestyle='solid', linewidth=2, markersize=12, marker='x')

    #ax2.set_title("Accuracy Ratio Error vs. Number of Qubits")
    ax2.set_title(suptitle + "\n" + subtitle, fontsize=12)
    ax2.set_xlabel("Number of Qubits")
    ax2.set_ylabel("Error in Accuracy Ratio (%)")

    # also plot a bar plot on the same figure
    ax2.bar(x_data2, y_data2, 0.75, alpha = 0.8, zorder = 3)
    ax2.grid(True, axis = 'y', color='silver', zorder = 0) 

    # error bars for the bar plot
    ax2.errorbar(x_data2, y_data2, yerr=y_err2, ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5,ls='', marker = "D", markersize = 8, mfc = 'c', mec = 'k', mew = 0.5,label = 'Error', alpha = 0.75, zorder = 5)

    # save the plot image
    if save_plot_images:                                
        metrics.save_plot_image(plt, os.path.join(f"Hydrogen-Lattice-(2)" +
                                            "-" + "accuracy_ratio_error"),
                                            backend_id)
                                            
    # show the plot(s)
    plt.show(block=True)
