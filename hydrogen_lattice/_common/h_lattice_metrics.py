
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
#################################
# Hydrogen-Lattice Metrics Module
#
# This module contains methods to initialize, store, aggregate and
# plot metrics collected in the benchmark programs for the Hydrogen-Lattice
#
# This module primatily provides custom plotting functions
#

import os
import matplotlib, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

import numpy as np
import math

# add to the path variable the path to the metrics module
import sys

sys.path[1:1] = [ "_common", "_common/qiskit", "hydrogen-lattice/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit", "../../hydrogen-lattice/_common/" ]

# import the metrics module
import metrics as metrics

h_lattice_metrics = metrics.circuit_metrics_detail

# save plot images flag
save_plot_images = True

# chemical accuracy in Hartrees
CHEM_ACC_HARTREE = 0.0016
 
# used for testing error bars in cumulative plots by faking data
testing_error_bars = False

# Toss out elapsed times for any run if the initial value is this factor of the second value 
omit_initial_elapsed_time_factor = 10

#################################################
# ADDITIONAL METRIC FUNCTIONS

# Find the last energy and radius associated with specific group of circuits
def find_last_metrics_for_group(group, instance):

    current_radius = 0
    doci_energy = 0
    fci_energy = 0
    energy = 0
    accuracy_ratio = 0
    solution_quality = 0
    
    # DEVNOTE: would it be easier to just get the last entry of array, rather than this loop?
    for circuit_id in h_lattice_metrics[group]:
        if np.floor(int(circuit_id)/1000) == instance:
        
            metrics_for_circuit = h_lattice_metrics[group][circuit_id]
            
            energy = metrics_for_circuit['energy']
            fci_energy = metrics_for_circuit['fci_energy']
            
            doci_energy = metrics_for_circuit['doci_energy'] if 'doci_energy' in metrics_for_circuit else fci_energy
            
            current_radius = metrics_for_circuit['radius'] if 'radius' in metrics_for_circuit else 0.75
            
            solution_quality = metrics_for_circuit['solution_quality']
            
            # recent additions (backwards compat)
            accuracy_ratio = metrics_for_circuit['accuracy_ratio'] if 'accuracy_ratio' in metrics_for_circuit else solution_quality
            random_energy = metrics_for_circuit['random_energy'] if 'random_energy' in metrics_for_circuit else 0

        else:
            continue
            
    return current_radius, doci_energy, fci_energy, random_energy, energy, accuracy_ratio, solution_quality

# Find the array of metrics associated with specific group of circuits
def find_metrics_array_for_group(group, instance, metric_name, x_val, x_metric_name):   

    y_data = []
    x_data = []
    x_label = ''
    y_label = ''

    '''
    # DEVNOTE: temporary backwards compatibility, remove later
    # (this is so we can display older runs that do not have accuracy ratio)
    if metric_name == 'accuracy_ratio': metric_name = 'solution_quality'
    '''
    
    for circuit_id in h_lattice_metrics[group]:
        if np.floor(int(circuit_id)/1000) == instance:
        
            # get the metric value (0 if not there)
            if metric_name in h_lattice_metrics[group][circuit_id]:
                metric_value = h_lattice_metrics[group][circuit_id][metric_name]
            else:
                metric_value = 0
                
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

        else:
            continue
    
    return x_data, x_label, y_data, y_label

# function to input a list of float and return a list of cumulative sums
def cumulative_sum(input_list: list):
    output_list = []
    sum = 0
    for item in input_list:
        sum += item
        output_list.append(sum)
        
    return output_list  


#################################################
# PLOT LINE METRICS

# maximum number of instances for which to plot line metrics
MAX_INSTANCES_FOR_LINE_PLOTS = 3

# function to plot all line metrics
def plot_all_line_metrics(suptitle=None, options=None,
        line_x_metrics=['iteration_count', 'cumulative_exec_time'],
        line_y_metrics=['energy', 'solution_quality_error'],
        plot_layout_style='grid',
        backend_id="UNKNOWN"):
    '''
    Function to plot all line metrics (energy, solution quality) vs different x values (iteration count, cumulative execution time)

    parameters:
    ----------
    suptitle: str   
        first line of the title of the figure
    plot_layout_style : str, optional
        Style of plot layout, 'grid', 'stacked', or 'individual', default = 'grid'
    line_y_metrics: list
        list of y metrics to plot
    line_x_metrics: list
        list of x values to plot
    backend_id: str 
        identifier for the backend system used for execution
    options: dict
        dictionary of options used for execution
    '''

    # if no metrics to plot, just return
    if line_x_metrics is None or line_y_metrics is None:
        return
        
    # if metrics are strings, convert to lists
    if type(line_x_metrics) is str:
        line_x_metrics = [line_x_metrics]
    if type(line_y_metrics) is str:
        line_y_metrics = [line_y_metrics]
    
    # Create standard title for all plots
    method = 2
    toptitle = suptitle + metrics.get_backend_title() 
    subtitle = ""
    
    # get group keys, sorted by qubit number
    igroup_keys = sorted([int(key) for key in h_lattice_metrics.keys()])
    group_keys = [str(key) for key in igroup_keys]
    
    # iterate over number of qubits and score metrics and plot each
    for qubit_count in group_keys:

        num_qubits = qubit_count
        group = str(num_qubits)
        
        circuit_ids = [int(x) for x in (h_lattice_metrics[qubit_count])]
        total_instances = int(np.floor(circuit_ids[-1]/1000))
        
        # generate a set of plots for each instance (radius) in data set
        for instance in range(1, total_instances + 1):
        
            # limit the line plots to a reasonable number
            if instance > MAX_INSTANCES_FOR_LINE_PLOTS:
                # print(f"... skipping line plot instance: {instance}")
                continue

            # search metrics store for final metrics for this group
            current_radius, doci_energy, fci_energy, random_energy, \
                    energy, solution_quality, accuracy_ratio = \
                            find_last_metrics_for_group(group, instance)

            # create common title (with hardcoded list of options, for now)
            suptitle = toptitle + f"\nqubits={num_qubits}, shots={options['shots']}, radius={current_radius}, restarts={options['restarts']}"
            
            # if drawing plots individually, get plot count from metrics length
            if plot_layout_style == 'individual':
                plot_count = min(4, len(line_y_metrics))
                subplot_count = 1
                
            # otherwise is it just one plot to draw with multiple subplots
            else:
                plot_count = 1
                subplot_count = min(4, len(line_y_metrics))
             
            # individual and stacked plots need an indication of the grouping
            if plot_layout_style != 'grid':
                print(f"----- Line Plots for the {qubit_count} qubit group -----")

            subplot_index = 0
            
            for jj in range(plot_count):
                
                # create individual figure for each plot, smaller if only one plot at a time
                if plot_layout_style == 'individual':
                    fig, axs1 = plt.subplots(1, 1, figsize=(6, 4.2))
                    axs = [axs1]
                    axes = [ axs1, axs1, axs1, axs1]
                    padding = 0.8 
                
                # create figure for stacked layout, all in one column
                elif plot_layout_style == 'stacked':
                    fig, axs = plt.subplots(subplot_count, 1, figsize=(6, subplot_count * 3.8 + 0.4))
                    axes = axs
                    padding = 0.8 
                    
                # create figure for grid layout, add rows as needed
                else:
                    if min(4, len(line_y_metrics)) <= 2:
                        fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
                        axes = axs
                        padding = 0.6
                    else:
                        fig, axs = plt.subplots(2, 2, figsize=(12, 8.0))
                        axes = [ axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1] ]
                        padding = 1.4
                    
                #fig.suptitle(suptitle, fontsize=13, backgroundcolor='whitesmoke')
                fig.suptitle(suptitle, fontsize=13, x=(0.5 if plot_layout_style=='grid' else 0.54))
                   
                #### Generate a subplot for all each metric combination   
                for kk in range(subplot_count):
                    
                    metric_name=line_y_metrics[subplot_index]
                    x_val=line_x_metrics[subplot_index]
                    
                    # draw a single subplot
                    plot_line_metric(ax=axes[subplot_index], subtitle=subtitle,
                        metric_name=metric_name, x_val=x_val,
                        num_qubits=qubit_count, instance=instance)
                    
                    subplot_index += 1
                        
                # this appears to be unneeded
                #plt.subplots_adjust(top=0.88, hspace=0.10)
          
                # add padding below suptitle, and between plots, due to multi-line titles
                fig.tight_layout(pad=padding, h_pad=2.0, w_pad=3.0)
                
                if save_plot_images:
                    image_name = (f"Hydrogen-Lattice-({method})-line-{group}-{instance}") + \
                        ("-all" if plot_layout_style != 'individual' else f"-{metric_name}-{x_val}")
                                
                    metrics.save_plot_image(plt, image_name, backend_id)
                                                
                if plot_layout_style == 'individual':
                    plt.show(block=True)

        if plot_layout_style != 'individual':
            plt.show(block=True)

# function to create a single subplot
def plot_line_metric(ax=None, subtitle:str="",
        metric_name:str="energy", x_val:str='cumulative_exec_time',
        num_qubits:int=None, instance:str=None):
    '''
    Function to create subplot for one line metrics (energy, solution quality, accuracy volume)
    vs different x values (iteration count, cumulative execution time)

    parameters:
    ----------
    ax: Axis Object
        axis on which to draw the plot
    subtitle: str   
        title of the subplot (unused now)
    metric_name: str
        list of score metrics to plot
    x_val: str
        list of x values to plot
    num_qubits: int
        number of qubits, or the group identifier
    instance: str
        instance identifier
    
    TODO: add error handling for invalid metric_name and x_val inputs
    '''

    # Get the data required to plot this metric

    # check if x_val starts with cumulative and turn cumulatifve flag on
    if x_val.startswith('cumulative'):
        x_metric_name = x_val[11:]
        cumulative_flag = True

    else:
        x_metric_name = x_val
        cumulative_flag = False
  
    group = str(num_qubits)
 
    # find the metrics_array for this group and instance, along with the final metrics 
    # if metric name ends with "_error", get value for base name
    x_data, x_label, y_data, y_label = \
            find_metrics_array_for_group(group, instance,
                    metric_name if not metric_name.endswith("_error") else metric_name[0:-6],
                    x_val, x_metric_name)

    current_radius, doci_energy, fci_energy, random_energy, \
            energy, solution_quality, accuracy_ratio = \
                    find_last_metrics_for_group(group, instance)
    
    # special case for solution_quality and solution_quality_error
    if metric_name.startswith("solution_quality"):
        metric_name_2 = metric_name.replace("solution_quality", "accuracy_ratio")
        
        x_data_2, x_label_2, y_data_2, y_label_2 = \
            find_metrics_array_for_group(group, instance,
                    metric_name_2 if not metric_name_2.endswith("_error") else metric_name_2[0:-6],
                    x_val, x_metric_name)
        
        y_data_2 = [1 - y for y in y_data_2]
        
    # special case for accuracy_ratio and accuracy_ratio_error
    if metric_name.startswith("accuracy_ratio"):
        metric_name_2 = metric_name.replace("accuracy_ratio", "solution_quality")
        
        x_data_2, x_label_2, y_data_2, y_label_2 = \
            find_metrics_array_for_group(group, instance,
                    metric_name_2 if not metric_name_2.endswith("_error") else metric_name_2[0:-6],
                    x_val, x_metric_name)
        
        y_data_2 = [1 - y for y in y_data_2]
    
    # always get the std_error data for energy 
    _, _, standard_error_data, standard_error_label = \
        find_metrics_array_for_group(group, instance,
                "standard_error",
                x_val, x_metric_name)
        
    # make the x_data cumulative if the cumulative flag is on
    if cumulative_flag:
        x_data = cumulative_sum(x_data)
        
    # calc range of the y data
    y_min = min(y_data); y_max = max(y_data)
    y_range = y_max - y_min

    # calculate the return parameters here

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

    # final accuracy ratio after all iterations
    if metric_name == 'accuracy_volume':
        final_accuracy_volume = y_data[-1]
    else:
        final_accuracy_volume = 0
        
    # if this is an 'error' metric, subtract all values from optimal (1.0)
    # DEVNOTE: may want to add 'energy_error' later, where optimal is not 1
    if metric_name == 'solution_quality_error' or metric_name == 'accuracy_ratio_error':
        y_data = [1 - y for y in y_data]
        
    ###################
    
    # set the title --- not currently shown for these subplots
    # fulltitle = f"{metrics.known_score_labels[metric_name]} vs. {metrics.known_x_labels[x_val]} "
    # ax.set_title(fulltitle, fontsize=12)
    
    # DEVNOTE: decided to eliminate the markers for now
    # plot the data as a scatter plot where the color of the point depends on the y value
    # if the metric is solution quality or accuracy ratio invert the color map   
    '''
    if metric_name == 'solution_quality' or metric_name == 'accuracy_ratio':
        ax.scatter(x_data, y_data, c=y_data, cmap=cm.coolwarm_r)
    else:
        ax.scatter(x_data, y_data, c=y_data, cmap=cm.coolwarm)
    ### ax.scatter(x_data, y_data, c='darkblue')
    '''
    # draw a line plot for the primary data
    #ax.plot(x_data, y_data, color='darkblue', linestyle='-.', linewidth=2, markersize=12)
    ax.plot(x_data, y_data, color='darkblue', linewidth=1.5, markersize=12)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # if the x metric is iteration count, set x ticks to be integers
    if x_metric_name == 'iteration_count':
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # if the score metric is energy or solution quality, plot the FCI and DOCI energy lines
    if metric_name == 'energy':
        ax.axhline(y=fci_energy, color='r', linestyle='--', label=f'FCI Energy    = {fci_energy:.4f}')
        ax.axhline(y=doci_energy, color='g', linestyle='-.', label=f'DOCI Energy = {doci_energy:.4f}')
        if random_energy != 0:
            ax.axhline(y=random_energy, color='darkblue', linestyle=':', label=f'Random Energy = {random_energy:.4f}')
        metric_legend_label = f'Solution Energy = {energy:.4f}'
        add_legend_item(ax, metric_legend_label, 'darkblue', '-')
        
        # start the y-ticks at 0 and end at y max
        ax.set_ylim([fci_energy-0.08*y_range, y_max+0.08*y_range])
        
        # error bars for the energy plot
        ax.errorbar(x_data, y_data, yerr=standard_error_data, ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5, ls='', marker = "D", markersize = 3, mfc = 'c', mec = 'k', mew = 0.5,label = 'Error', alpha = 0.75, zorder = 5)

    # solution quality
    elif metric_name == 'solution_quality':
        ax.axhline(y=1, color='r', linestyle='--', label='Ideal Solution = 1.0000')
        metric_legend_label = f'Solution Quality = {final_solution_quality:.4f}'
        add_legend_item(ax, metric_legend_label, 'darkblue', '-')
        
        # start the y-ticks at 0 and end at 1.1
        ax.set_ylim([0.0, 1.08])
        
    # accuracy ratio
    elif metric_name == 'accuracy_ratio':
        ax.axhline(y=1, color='r', linestyle='--', label='Ideal Solution = 1.0000')
        metric_legend_label = f'Accuracy Ratio = {final_accuracy_ratio:.4f}'
        add_legend_item(ax, metric_legend_label, 'darkblue', '-')
        
        # start the y-ticks at just below min and just above 1.0
        ax.set_ylim([y_min-0.08*y_range, 1.0+0.08*y_range])

    # solution quality error
    elif metric_name == 'solution_quality_error':
        ax.set_ylabel("Solution Quality Error")
        
        # compute chemical accuracy relative to exact energy (FCI)
        # and using solution quality formula
        #precision = 0.5
        #chacc = -math.atan((CHEM_ACC_HARTREE/fci_energy) * precision) / (math.pi/2)
        chacc = get_solution_quality_value(CHEM_ACC_HARTREE, fci_energy)
        
        # define the lowest y value a fraction below the chemical accuracy line
        y_base = chacc/3

        # if lowest value is below this base, make base a little lower
        if min(y_data) < y_base:
            y_base = 0.75 * min(y_data)
        
        # draw a shaded rectangle from bottom of plot to chem accuracy level
        rect = Rectangle((0.0, y_base), x_data[-1], chacc-y_base, color='lightgrey')
        ax.add_patch(rect)
        
        # draw horiz line representing level of error that would be chemical accuracy
        ax.axhline(y=chacc, color='r', linestyle='--', label=f'Chem. Accuracy = {1 - chacc:.4f}')
        
        # add legend item for solution quality
        final_solution_quality = 1.0 - y_data[-1]
        metric_legend_label = f'Solution Quality = {final_solution_quality:.4f}'
        
        # make this a log axis from base to 1.0
        ax.set_ylim(y_base, 1.0)
        ax.set_yscale('log') 
        
        # draw the accuracy ratio error plot 
        ax.plot(x_data, y_data_2, color='darkblue', linestyle='-.', linewidth=1, markersize=12)
        
        # scale the standard error data to the range between random_energy and fci_energy
        standard_error_data = [get_accuracy_ratio_value(err, fci_energy, random_energy) for err in standard_error_data]
        
        # error bars for the accuracy ratio error plot 
        ax.errorbar(x_data, y_data, yerr=standard_error_data, ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5, ls='', marker = "D", markersize = 3, mfc = 'c', mec = 'k', mew = 0.5,label = None, alpha = 0.75, zorder = 5)

        # add legend item for accuracy ratio
        final_accuracy_ratio = 1.0 - y_data_2[-1]
        metric_legend_label_2 = f'Accuracy Ratio = {final_accuracy_ratio:.4f}'
        
        add_two_legend_items(ax, metric_legend_label_2, "darkblue", "-.",
                            metric_legend_label, "darkblue", "-")
        
    # accuracy ratio error
    elif metric_name == 'accuracy_ratio_error':
        ax.set_ylabel("Accuracy Ratio Error")
        
        # compute chemical accuracy relative to exact energy (FCI)
        # and using solution quality formula
        precision = 0.5
        chacc = get_accuracy_ratio_value(CHEM_ACC_HARTREE, fci_energy, random_energy)
        
        # define the lowest y value a fraction below the chemical accuracy line
        y_base = chacc/3
        
        # if lowest value is below this base, make base a little lower
        if min(y_data) < y_base:
            y_base = 0.75 * min(y_data)
            
        # draw a shaded rectangle from bottom of plot to chem accuracy level
        rect = Rectangle((0.0, y_base), x_data[-1], chacc-y_base, color='lightgrey')
        ax.add_patch(rect)
        
        # draw horiz line representing level of error that would be chemical accuracy
        ax.axhline(y=chacc, color='r', linestyle='--', label=f'Chem. Accuracy = {1 - chacc:.4f}')
        
        # draw the solution_quality error plot 
        ax.plot(x_data, y_data_2, color='darkblue', linestyle='-.', linewidth=1, markersize=12)
        
        # scale the standard error data to the range between random_energy and fci_energy
        standard_error_data = [get_accuracy_ratio_value(err, fci_energy, random_energy) for err in standard_error_data]
        
        # error bars for the accuracy ratio error plot 
        ax.errorbar(x_data, y_data, yerr=standard_error_data, ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5, ls='', marker = "D", markersize = 3, mfc = 'c', mec = 'k', mew = 0.5, label = None, alpha = 0.75, zorder = 5)
        
        # add legend item for solution quality
        final_accuracy_ratio = 1.0 - y_data[-1]
        metric_legend_label = f'Accuracy Ratio = {final_accuracy_ratio:.4f}'
        add_legend_item(ax, metric_legend_label, 'darkblue', '-')
        
        # add legend item for accuracy ratio
        final_solution_quality = 1.0 - y_data_2[-1]
        metric_legend_label_2 = f'Solution Quality = {final_solution_quality:.4f}'
        
        # make this a log axis from base to 1.0
        ax.set_ylim(y_base, 1.0)
        ax.set_yscale('log') 
        
        add_two_legend_items(ax,
                    metric_legend_label, "darkblue", "-",
                    metric_legend_label_2, "darkblue", "-.")
        
    # add a horizontal line at y=0 for accuracy volume
    elif metric_name == 'accuracy_volume':
        ax.axhline(y=0, color='r', linestyle='--', label='Ideal Solution')
        metric_legend_label = f'Accuracy Volume = {final_accuracy_volume:.3f}' 
        add_legend_item(ax, metric_legend_label, 'darkblue', '-')
    
    ax.grid(True)

    return

# Returns a value scaled between infinity and fci_energy
def get_solution_quality_value(value, fci_energy):
    precision = 0.5
    return -math.atan((value/fci_energy) * precision) / (math.pi/2)
    
# Returns a value scaled between random_energy and fci_energy
def get_accuracy_ratio_value(value, fci_energy, random_energy):
    return abs(value / (random_energy - fci_energy))  
    
# add a new legend item
# add a copy of first legend item and change to blue with the metric value shown
def add_legend_item(ax, item_label, item_color, item_linestyle):

    handles, labels = ax.get_legend_handles_labels()
    
    newhandle = matplotlib.lines.Line2D([0,1],[0,1])
    newhandle.update_from(handles[0])
    newhandle.set_color(item_color)
    newhandle.set_linestyle(item_linestyle)
    handles.append(newhandle)
    labels.append(item_label)
    
    ax.legend(handles, labels)
  
# add two new legend items (can only do this once, it seems)
# should make this an array function later
def add_two_legend_items(ax,
                item_label, item_color, item_linestyle,
                item_label2, item_color2, item_linestyle2):

    handles, labels = ax.get_legend_handles_labels()
   
    newhandle = matplotlib.lines.Line2D([0,1],[0,1])
    newhandle.update_from(handles[0])
    newhandle.set_color(item_color)
    newhandle.set_linestyle(item_linestyle)
    handles.append(newhandle)
    labels.append(item_label)
    
    newhandle2 = matplotlib.lines.Line2D([0,1],[0,1])
    newhandle2.update_from(handles[0])
    newhandle2.set_color(item_color2)
    newhandle2.set_linestyle(item_linestyle2)
    handles.append(newhandle2)
    labels.append(item_label2)
    
    ax.legend(handles, labels)
    
    
#################################################
# PLOT CUMULATIVE METRICS

# function to plot all cumulative/final metrics
def plot_all_cumulative_metrics(suptitle=None,
        bar_y_metrics=["average_exec_times", "accuracy_ratio_error"],
        bar_x_metrics=["num_qubits"],
        show_elapsed_times=True,
        use_logscale_for_times=False,
        plot_layout_style='grid',
        backend_id="UNKNOWN", options=None):
    '''
    Function to plot all cumulative metrics (average_iteration_time, final_accuracy_ratio)

    parameters:
    ----------
    suptitle: str   
        first line of the title of the figure
    bar_y_metrics: list
        list of metrics to plot
    bar_x_metrics: list
        list of x values to plot
    plot_layout_style : str, optional
        Style of plot layout, 'grid', 'stacked', or 'individual', default = 'grid'
    backend_id: str 
        identifier for the backend system used for execution
    options: dict
        dictionary of options used for execution
    '''

    # if no metrics, just return
    if bar_y_metrics is None or bar_x_metrics is None:
        return
        
    if type(bar_y_metrics) is str:
        bar_y_metrics = [bar_y_metrics]
    if type(bar_x_metrics) is str:
        bar_x_metrics = [bar_x_metrics]
    
    average_exec_time_per_iteration = []
    average_exec_time_per_iteration_error = []
    average_elapsed_time_per_iteration = []
    average_elapsed_time_per_iteration_error = []
    average_solution_quality = []
    average_accuracy_ratio = []
    average_solution_quality_error = []
    average_accuracy_ratio_error = []
    qubit_counts = []
    
    # iterate over number of qubits and instances to compute averages
    for num_qubits in h_lattice_metrics:
        group = str(num_qubits)
        
        circuit_ids = [int(x) for x in (h_lattice_metrics[group])]
        total_instances = int(np.floor(circuit_ids[-1]/1000))

        exec_time_array = []
        elapsed_time_array = []
        sol_quality_array = []
        acc_ratio_array = []
        
        # generate a set of plots for each instance (radius) in data set
        for instance in range(1, total_instances + 1):

            # search metrics store for final metrics for this group
            current_radius, doci_energy, fci_energy, random_energy, \
                    energy, accuracy_ratio, solution_quality = \
                        find_last_metrics_for_group(group, instance)
            
            ###### find the execution time array for "energy" metric         
            x_data, x_label, y_data, y_label = \
                find_metrics_array_for_group(group, instance, "energy", "cumulative_exec_time", "exec_time")
                
            # make the x_data cumulative if the cumulative flag is on
            x_data = cumulative_sum(x_data)
            
            # and compute average execution time
            exec_time_per_iteration = x_data[-1]/len(x_data)

            ###### find the elapsed execution time array for "energy" metric         
            x_data, x_label, y_data, y_label = \
                find_metrics_array_for_group(group, instance, "energy", "cumulative_elapsed_time", "elapsed_time")
            
            # DEVNOTE: A brutally simplistic way to toss out initially long elapsed times
            # that are most likely due to either queueing or system initialization
            if len(x_data) > 1 and omit_initial_elapsed_time_factor > 0 and (x_data[0] > omit_initial_elapsed_time_factor * x_data[1]):
                x_data[0] = x_data[1]
                
            # make the x_data cumulative if the cumulative flag is on
            x_data = cumulative_sum(x_data)
            
            # and compute average elapsed execution time
            elapsed_time_per_iteration = x_data[-1]/len(x_data)
            
            exec_time_array.append(exec_time_per_iteration)
            elapsed_time_array.append(elapsed_time_per_iteration)
            sol_quality_array.append(1 - solution_quality)
            acc_ratio_array.append(1 - accuracy_ratio)
                   
        average_et = np.average(exec_time_array)
        error_et = np.std(exec_time_array)/np.sqrt(len(exec_time_array))
        average_elt = np.average(elapsed_time_array)
        error_elt = np.std(elapsed_time_array)/np.sqrt(len(elapsed_time_array))
        average_ar = np.average(acc_ratio_array)
        error_ar = np.std(acc_ratio_array)/np.sqrt(len(acc_ratio_array))
        average_sq = np.average(sol_quality_array)
        error_sq = np.std(sol_quality_array)/np.sqrt(len(sol_quality_array))

        average_exec_time_per_iteration.append(average_et)
        average_exec_time_per_iteration_error.append(error_et)
        average_elapsed_time_per_iteration.append(average_elt)
        average_elapsed_time_per_iteration_error.append(error_elt)
        average_accuracy_ratio.append(average_ar * 100)
        average_accuracy_ratio_error.append(error_ar * 100)
        average_solution_quality.append(average_sq * 100)
        average_solution_quality_error.append(error_sq * 100)

        qubit_counts.append(num_qubits)

    ##########################
    
    individual=True
    
    # Create standard title for all plots
    method = 2
    toptitle = suptitle + metrics.get_backend_title() 
    subtitle = ""
    
    # create common title (with hardcoded list of options, for now)
    suptitle = toptitle + f"\nqubits={num_qubits}, shots={options['shots']}, radius={current_radius}, restarts={options['restarts']}"
    
    # since all subplots share the same header, give user and indication of the grouping
    if individual:
        print("----- Cumulative Plots for all qubit groups -----")
    
    # draw the average execution time plots
    if "average_exec_times" in bar_y_metrics:
        plot_exec_time_metrics(suptitle=suptitle,
            x_data=qubit_counts,
            x_label="Number of Qubits",
            y_data=average_elapsed_time_per_iteration,
            y_err=average_elapsed_time_per_iteration_error,
            y_data_2=average_exec_time_per_iteration,
            y_err_2=average_exec_time_per_iteration_error,
            y_label="Cumulative Execution Times / iteration (s)",
            show_elapsed_times=show_elapsed_times,
            use_logscale_for_times=use_logscale_for_times,
            suffix="avg_exec_times_per_iteration")
    
    # draw the accuracy ratio error plot
    if "accuracy_ratio_error" in bar_y_metrics:
        plot_cumulative_metrics(suptitle=suptitle,
            x_data=qubit_counts,
            x_label="Number of Qubits",
            y_data=average_accuracy_ratio,
            y_err=average_accuracy_ratio_error,
            y_label="Error in Accuracy Ratio (%)",
            y_lim_min=1.0,
            suffix="accuracy_ratio_error")

#####################################

# method to plot cumulative accuracy ratio vs. number of qubits
def plot_cumulative_metrics(suptitle="",
            x_data:list=None, x_label:str="",
            y_data:list=None, y_err:list=None,
            y_label:str="", y_lim_min=None,
            plot_layout_style='grid', 
            suffix=None):
    '''
    Function to plot cumulative metrics (accuracy ratio, execution time per iteration) over different number of qubits

    parameters:
    ----------
    x_data: list
        list of x data for plot
    x_label: str
        label for x axis
    y_data: list
        list of y data for plot
    y_err: list
        list of y error data for plot
    y_label: str
        label for y axis
    y_lim_min: float    
        minimum value to autoscale y axis 
    '''

    # get subtitle from metrics
    m_subtitle = metrics.circuit_metrics['subtitle']

    # get backend id from subtitle
    backend_id = m_subtitle[9:]

    # create a figure for the plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4.2))
    
    # and add the title (shifted to the right a bit if single column plot)
    fig1.suptitle(suptitle, fontsize=13, x=(0.5 if plot_layout_style=='grid' else 0.54))
    
    ###### DEVNOTE: these arrays should be converted to float (and sorted?) at higher level
    
    # sort the arrays, in case they come out of order
    x_data = [float(x) for x in x_data]
    y_data = [float(y) for y in y_data]
    z = sorted(zip(x_data, y_data))
    x_data = [x for x, y in z]
    y_data = [y for x, y in z]
    
    ######
    
    # check if we have sparse or non-linear axis data and linearize if so
    # (convert irregular x-axis data to linear if any non-linear gaps in the data)
    if metrics.needs_linearize(x_data, gap=2):
        x_data, x_labels = metrics.linearize_axis(x_data, gap=1, outer=0, fill=False) 
        ax1.set_xticks(x_data)
        if x_labels != None:
            plt.xticks(x_data, x_labels)

    # for testing of error bars
    if testing_error_bars:
        y_err = [y * 0.15 for y in y_data]
    
    ##########
    
    # autoscale y axis to user-specified min
    if y_lim_min != None and max(y_data) < y_lim_min:
        ax1.set_ylim(0.0, y_lim_min)
    
    # set the axis labels
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    # add the background grid
    ax1.grid(True, axis = 'y', color='silver', zorder = 0)
    
    # plot a bar plot of the data values
    ax1.bar(x_data, y_data, zorder = 3)
    
    # plot a dotted line to connect the values
    ax1.plot(x_data, y_data, color='darkblue',
            linestyle='dotted', linewidth=1, markersize=6, zorder = 3)
    
    # error bars for the bar plot
    ax1.errorbar(x_data, y_data, yerr=y_err, ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5, ls='', marker = "D", markersize = 5, mfc = 'c', mec = 'k', mew = 0.5,label = 'Error', alpha = 0.75, zorder = 5)
    
    ##########
        
    # add padding below suptitle, and between plots, due to multi-line titles
    padding=0.8
    fig1.tight_layout(pad=padding, h_pad=2.0, w_pad=3.0)
                
    # save the plot image
    if save_plot_images:
        metrics.save_plot_image(plt, os.path.join("Hydrogen-Lattice-(2)" +
                                            "-" + suffix),
                                            backend_id)
    # show the plot(s)
    plt.show(block=True)

#####################################

# method to plot cumulative exec and execution time vs. number of qubits
def plot_exec_time_metrics(suptitle="",
            x_data:list=None, x_label:str="",
            y_data:list=None, y_err:list=None,
            y_data_2:list=None, y_err_2:list=None,
            y_label:str="", y_lim_min=None,
            show_elapsed_times=True,
            use_logscale_for_times=False,
            plot_layout_style='grid', 
            suffix=None):
    '''
    Function to plot execution time metrics (elapsed/execution time per iteration) over different number of qubits

    parameters:
    ----------
    x_data: list
        list of x data for plot
    x_label: str
        label for x axis
    y_data: list
        list of y data for plot
    y_err: list
        list of y error data for plot
    y_data: list
        list of y data for plot 2
    y_err: list
        list of y error data for plot 2
    y_label: str
        label for y axis
    y_lim_min: float    
        minimum value to autoscale y axis 
    '''

    # get subtitle from metrics
    m_subtitle = metrics.circuit_metrics['subtitle']

    # get backend id from subtitle
    backend_id = m_subtitle[9:]

    # create a figure for the plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4.2))
    
    # and add the title (shifted to the right a bit if single column plot)
    fig1.suptitle(suptitle, fontsize=13, x=(0.5 if plot_layout_style=='grid' else 0.54))
    
    ###### DEVNOTE: these arrays should be converted to float (and sorted?) at higher level
    
    # sort the arrays, in case they come out of order
    x_data = [float(x) for x in x_data]
    y_data = [float(y) for y in y_data]
    y_data_2 = [float(y) for y in y_data_2]
    z = sorted(zip(x_data, y_data, y_data_2))
    x_data = [x for x, y, y2 in z]
    y_data = [y for x, y, y2 in z]
    y_data_2 = [y2 for x, y, y2 in z]
    
    #############
    
    # check if we have sparse or non-linear axis data and linearize if so
    # (convert irregular x-axis data to linear if any non-linear gaps in the data)
    if metrics.needs_linearize(x_data, gap=2):
        x_data, x_labels = metrics.linearize_axis(x_data, gap=1, outer=0, fill=False) 
        ax1.set_xticks(x_data)
        if x_labels != None:
            plt.xticks(x_data, x_labels)

    # for testing of error bars
    if testing_error_bars:
        y_err = [y * 0.15 for y in y_data]
        y_err_2 = [y * 0.15 for y in y_data_2]
        
    #############
    
    # set the axis labels
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    
    # add the background grid
    ax1.grid(True, axis = 'y', which='major', color='silver', zorder = 0)
  
    # determine max of both data sets, with a lower limit of 0.1
    # DEVNOTE: we are suppressing range of the first plot if show_elapsed times is False, backwards?
    y_max_0 = 0.1
    y_max_0 = max(y_max_0, max(y_data_2))
    if show_elapsed_times:
        y_max_0 = max(y_max_0, max(y_data))
        
    if y_max_0 > 0.1:
            y_max_0 *= 1.2
     
    # set up log scale if specified
    y_min_0 = 0.0
    if use_logscale_for_times:
        ax1.set_yscale('log') 
        y_min_0 = min(0.01, min(y_data_2) / 2.0)    # include smallest data value
        
        if y_max_0 > 0.1:
            y_max_0 *= 2.0 
   
    ax1.set_ylim([y_min_0, y_max_0])
    
    # elapsed time bar plot
    if show_elapsed_times:
        ax1.bar(x_data, y_data, 0.75, color='skyblue', zorder = 3)
        
        ax1.plot(x_data, y_data, color='darkblue',
            linestyle='dotted', linewidth=1, markersize=6, zorder = 3)
            
        ax1.errorbar(x_data, y_data, yerr=y_err, ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5, ls='', marker = "D", markersize = 5, mfc = 'c', mec = 'k', mew = 0.5, label = 'Error', alpha = 0.75, zorder = 5)
    
    # execution time bar plot
    ax1.bar(x_data, y_data_2, zorder = 3)
    
    ax1.plot(x_data, y_data_2, color='darkblue',
            linestyle='dotted', linewidth=1, markersize=6, zorder = 3)
            
    ax1.errorbar(x_data, y_data_2, yerr=y_err_2, ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5, ls='', marker = "D", markersize = 5, mfc = 'c', mec = 'k', mew = 0.5, label = 'Error', alpha = 0.75, zorder = 5)
    
    # legend
    if show_elapsed_times:
        elapsed_patch = Patch(color='skyblue', label='Elapsed')
        exec_patch = Patch(color='#1f77b4', label='Quantum')
        #ax1.legend(handles=[elapsed_patch, exec_patch], loc='upper left')
        ax1.legend(handles=[elapsed_patch, exec_patch])
        #ax1.legend(['Elapsed', 'Quantum'], loc='upper left')
    #else:
        #ax1.legend(['Quantum'], loc='upper left')

    ##############
    
    # add padding below suptitle, and between plots, due to multi-line titles
    padding=0.8
    fig1.tight_layout(pad=padding, h_pad=2.0, w_pad=3.0)
                
    # save the plot image
    if save_plot_images:
        metrics.save_plot_image(plt, os.path.join("Hydrogen-Lattice-(2)" +
                                            "-" + suffix),
                                            backend_id)
    # show the plot(s)
    plt.show(block=True)
