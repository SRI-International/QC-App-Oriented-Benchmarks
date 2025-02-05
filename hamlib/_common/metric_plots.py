###############################################################################
# (C) Quantum Economic Development Consortium (QED-C) 2025.
# Technical Advisory Committee on Standards and Benchmarks (TAC)
#
#######################################
# HamLib Simulation Metric Plots Module
#
# This module contains methods to initialize, store, aggregate and
# plot metrics collected in the benchmark programs for HamLib simulation.
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

# This module does not currently use the metrics stored in top-level benchmark metrics.
# Instead we collect metrics locally for now and concentrate here on plotting.

sys.path[1:1] = [ "_common" ]
sys.path[1:1] = [ "../_common" ]

# import the metrics module
import metrics as metrics

# h_lattice_metrics = metrics.circuit_metrics_detail

# save plot images flag
save_plot_images = True

# chemical accuracy in Hartrees
#CHEM_ACC_HARTREE = 0.0016
 
# used for testing error bars in cumulative plots by faking data
testing_error_bars = False

# Toss out elapsed times for any run if the initial value is this factor of the second value 
omit_initial_elapsed_time_factor = 10


_markers = [ ".", "s", "*", "h", "P", "X", "d" ]
_colors = [ "coral", "C0", "C2", "C4", "C5", "C6" ]
_styles = [ "dotted" ]

    
#################################################
# PLOT EXPECTATION METRICS

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
def plot_expectation_value_metrics(suptitle="",
            x_data:list=None, x_label:str="",
            y_data:list=None, y_err:list=None,
            y_data_2:list=None, y_err_2:list=None,
            y_label:str="", y_lim_min=None,
            show_elapsed_times=True,
            use_logscale_for_times=False,
            plot_layout_style='grid', 
            
            groups=None,
            expectation_values_exact=None,
            expectation_values_computed=None,
                      
            backend_id=None,
            options=None,
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
       
    # Create standard title for all plots
    #toptitle = suptitle + metrics.get_backend_title()
    toptitle = suptitle + f"\nDevice={backend_id}"
    subtitle = ""
    
    # create common title (with hardcoded list of options, for now)
    suptitle = toptitle + f"\nham={options['ham']}, gm={options['gm']}, shots={options['shots']}, reps={options['reps']}"
    
    print("----- Expectation Value Plot -----")

    # create a figure for the plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4.2))
    
    # and add the title (shifted to the right a bit if single column plot)
    fig1.suptitle(suptitle, fontsize=13, x=(0.5 if plot_layout_style=='grid' else 0.54))
    
    #ax1.set_title('Plot of Two Data Sets')
    
    ###### Plot the metrics
    
    if not groups:
        return

    x_data = groups
    y_data1 = expectation_values_exact
    y_data2 = expectation_values_computed
    
    #############
    
    # set the axis labels
    ax1.set_xlabel("Number of Qubits")
    ax1.set_ylabel("Expectation Value")
    
    # add the background grid
    ax1.grid(True, axis = 'y', which='major', color='silver', zorder = 0)
    
    # Plot the data
    ax1.plot(x_data[:len(y_data1)], y_data1, label='Exact Value', marker='.', color='coral', linestyle='dotted')
    ax1.plot(x_data[:len(y_data2)], y_data2, label='Quantum Value', marker='s', color='C0')

    # Add legend
    ax1.legend()

    # Autoscale the y-axis
    ax1.autoscale(axis='y')
    
    """
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
    """
    ##############
    
    # add padding below suptitle, and between plots, due to multi-line titles
    padding=0.8
    fig1.tight_layout(pad=padding, h_pad=2.0, w_pad=3.0)
                
    # save the plot image
    
    save_plot_images = True
    suffix = ""
    
    if save_plot_images:

        suffix=("-" + suffix) if suffix else ""                                 
        metrics.save_plot_image(plt,
                os.path.join(f"HamLib-Simulation-{options['ham']}-exp-values" + suffix),
                backend_id)
                                            
    # show the plot(s)
    plt.show(block=True)

#####################################

# method to plot cumulative exec and execution time vs. number of qubits
def plot_expectation_value_metrics_2(suptitle="",
            x_data:list=None, x_label:str="",
            y_data:list=None, y_err:list=None,
            y_data_2:list=None, y_err_2:list=None,
            y_label:str="", y_lim_min=None,
            show_elapsed_times=True,
            use_logscale_for_times=False,
            plot_layout_style='grid', 
            
            groups: list = None,
            labels: list = None,
            values: list = None,
                      
            backend_id=None,
            options=None,
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
    
    """ 
    # Create standard title for all plots
    #toptitle = suptitle + metrics.get_backend_title()
    toptitle = suptitle + f"\nDevice={backend_id}"
    subtitle = ""
    
    # create common title (with hardcoded list of options, for now)
    suptitle = toptitle + f"\nham={options['ham']}, gm={options['gm']}, shots={options['shots']}, reps={options['reps']}"
    """
    print("----- Expectation Value Plot -----")

    # create a figure for the plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4.2))
    
    # and add the title (shifted to the right a bit if single column plot)
    fig1.suptitle(suptitle, fontsize=13, x=(0.5 if plot_layout_style=='grid' else 0.54))
    
    #ax1.set_title('Plot of Two Data Sets')
    
    ###### Plot the metrics
    
    if not groups:
        return

    x_data = groups  
    
    #############
    
    # set the axis labels
    ax1.set_xlabel("Number of Qubits")
    ax1.set_ylabel("Expectation Value")
    
    # add the background grid
    ax1.grid(True, axis = 'y', which='major', color='silver', zorder = 0)
    
    # Plot the data
    ax1.plot(x_data[:len(values[0])], values[0], label=labels[0], marker='.', color='coral', linestyle='dotted')
    
    for i in range(1, len(values)):
        color = _colors[i] if i < len(_colors) else _colors[-1]
        marker = _markers[i] if i < len(_markers) else _markers[-1]
        
        ax1.plot(x_data[:len(values[i])], values[i], label=labels[i], marker=marker, color=color)

    # Add legend
    ax1.legend()

    # Autoscale the y-axis
    ax1.autoscale(axis='y')

    ##############
    
    # add padding below suptitle, and between plots, due to multi-line titles
    padding=0.8
    fig1.tight_layout(pad=padding, h_pad=2.0, w_pad=3.0)
                
    # save the plot image
    
    save_plot_images = True
    suffix = ""
    
    if save_plot_images:

        suffix=("-" + suffix) if suffix else ""                                 
        metrics.save_plot_image(plt,
                os.path.join(f"HamLib-Simulation-{options['ham']}-exp-values" + suffix),
                backend_id)
                                            
    # show the plot(s)
    plt.show(block=True)


#####################################

# method to plot cumulative exec and execution time vs. number of qubits
def plot_expectation_time_metrics(suptitle="",
            x_data:list=None, x_label:str="",
            y_data:list=None, y_err:list=None,
            y_data_2:list=None, y_err_2:list=None,
            y_label:str="", y_lim_min=None,
            show_elapsed_times=True,
            use_logscale_for_times=False,
            plot_layout_style='grid', 
            
            groups=None,
            expectation_times_exact=None,
            expectation_times_computed=None,
                      
            backend_id=None,
            options=None,
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
       
    # Create standard title for all plots
    #toptitle = suptitle + metrics.get_backend_title()
    toptitle = suptitle + f"\nDevice={backend_id}"
    subtitle = ""
    
    # create common title (with hardcoded list of options, for now)
    suptitle = toptitle + f"\nham={options['ham']}, gm={options['gm']}, shots={options['shots']}, reps={options['reps']}"
    
    print("----- Expectation Time Plot -----")

    # create a figure for the plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4.2))
    
    # and add the title (shifted to the right a bit if single column plot)
    fig1.suptitle(suptitle, fontsize=13, x=(0.5 if plot_layout_style=='grid' else 0.54))
    
    #ax1.set_title('Plot of Two Data Sets')
    
    ###### Plot the metrics
    
    if not groups:
        return

    x_data = groups
    y_data1 = expectation_times_exact
    y_data2 = expectation_times_computed
    
    #############
    
    # set the axis labels
    ax1.set_xlabel("Number of Qubits")
    ax1.set_ylabel("Expectation Compute Time (sec)")
    
    # add the background grid
    ax1.grid(True, axis = 'y', which='major', color='silver', zorder = 0)
    
    # Plot the data
    ax1.plot(x_data[:len(y_data1)], y_data1, label='Exact Time', marker='.', color='coral', linestyle='dotted')
    ax1.plot(x_data[:len(y_data2)], y_data2, label='Quantum Time', marker='X', color='C0')

    # Add legend
    ax1.legend()

    # Autoscale the y-axis
    ax1.autoscale(axis='y')
    
    ##############
    
    # add padding below suptitle, and between plots, due to multi-line titles
    padding=0.8
    fig1.tight_layout(pad=padding, h_pad=2.0, w_pad=3.0)
                
    # save the plot image
    
    save_plot_images = True
    suffix = ""
    
    if save_plot_images:

        suffix=("-" + suffix) if suffix else ""                                 
        metrics.save_plot_image(plt,
                os.path.join(f"HamLib-Simulation-{options['ham']}-exp-times" + suffix),
                backend_id)
                                            
    # show the plot(s)
    plt.show(block=True)

#####################################

# method to plot cumulative exec and execution time vs. number of qubits
def plot_expectation_time_metrics_2(suptitle="",
            x_data:list=None, x_label:str="",
            y_data:list=None, y_err:list=None,
            y_data_2:list=None, y_err_2:list=None,
            y_label:str="", y_lim_min=None,
            show_elapsed_times=True,
            use_logscale_for_times=False,
            plot_layout_style='grid', 
            
            groups: list = None,
            labels: list = None,
            times: list = None,
                      
            backend_id=None,
            options=None,
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

    """
    # Create standard title for all plots
    #toptitle = suptitle + metrics.get_backend_title()
    toptitle = suptitle + f"\nDevice={backend_id}"
    subtitle = ""
    
    # create common title (with hardcoded list of options, for now)
    suptitle = toptitle + f"\nham={options['ham']}, gm={options['gm']}, shots={options['shots']}, reps={options['reps']}"
    """
    print("----- Expectation Time Plot -----")

    # create a figure for the plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4.2))
    
    # and add the title (shifted to the right a bit if single column plot)
    fig1.suptitle(suptitle, fontsize=13, x=(0.5 if plot_layout_style=='grid' else 0.54))
    
    #ax1.set_title('Plot of Two Data Sets')
    
    ###### Plot the metrics
    
    if not groups:
        return

    x_data = groups
    
    #############
    
    # set the axis labels
    ax1.set_xlabel("Number of Qubits")
    ax1.set_ylabel("Expectation Compute Time (sec)")
    
    # add the background grid
    ax1.grid(True, axis = 'y', which='major', color='silver', zorder = 0)
    
    # Plot the data 
    ax1.plot(x_data[:len(times[0])], times[0], label=labels[0], marker='.', color='coral', linestyle='dotted')

    for i in range(1, len(times)):
        color = _colors[i] if i < len(_colors) else _colors[-1]
        marker = _markers[i] if i < len(_markers) else _markers[-1]
        
        ax1.plot(x_data[:len(times[i])], times[i], label=labels[i], marker=marker, color=color)
    
    # Add legend
    ax1.legend()

    # Autoscale the y-axis
    ax1.autoscale(axis='y')
    
    ##############
    
    # add padding below suptitle, and between plots, due to multi-line titles
    padding=0.8
    fig1.tight_layout(pad=padding, h_pad=2.0, w_pad=3.0)
                
    # save the plot image
    
    save_plot_images = True
    suffix = ""
    
    if save_plot_images:

        suffix=("-" + suffix) if suffix else ""                                 
        metrics.save_plot_image(plt,
                os.path.join(f"HamLib-Simulation-{options['ham']}-exp-times" + suffix),
                backend_id)
                                            
    # show the plot(s)
    plt.show(block=True)
