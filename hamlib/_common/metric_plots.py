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


# This module does not currently use the metrics stored in top-level benchmark metrics.
# Instead we collect metrics locally for now and concentrate here on plotting.

# import the metrics module
from _common import metrics as metrics

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
_colors = [ "coral", "lightseagreen", "C2", "C4", "slateblue", "magenta", "yellowgreen", "royalblue" ]
#_colors = [ "coral", "C0", "C1", "C2", "C3", "C4", "C5", "C6" ]
#_styles = [ "dotted", "solid" ]
_styles = [ "dashed", "solid" ]

    
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


def get_errors(
        hamiltonian_name: str,
        backend_id: str,
        num_qubits: int,
        group_method: str,
        num_shots: int,
        exact_energies,
        computed_energies
):
    """
        return the errors
    Args:
        errors (list or array): List of error values.
    """
    errors = [computed - exact for computed, exact in zip(computed_energies, exact_energies)]
    base_ham_name = os.path.basename(hamiltonian_name)

    title = "Error Distribution with Key Metrics"
    title += f"\nHam={base_ham_name}, qubits={num_qubits}, gm={group_method}, shots={num_shots}"

    errors = np.array(errors)

    # Compute key error metrics
    std_dev = np.std(errors)  # Standard deviation (σ)
    mean_error = np.mean(errors)  # Mean error (detects bias)
    bias_direction = "positive" if mean_error > 0 else "negative"
    if abs(mean_error) > 0.5 * (np.max(errors) - np.min(errors)) / 4:  # Using range-based threshold
        print(f"⚠️ Warning: Significant {bias_direction} bias detected in the error distribution!")
    else:
        print("✅ No significant bias detected.")
    return std_dev, mean_error



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
            options={},
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
        
    suptitle = append_options_to_title(suptitle, options, backend_id)
    
    subtitle = ""
    
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
    #ax1.legend()

    # Autoscale the y-axis
    ax1.autoscale(axis='y')
    
    ###############
    
    # Create secondary y-axis
    ax2 = ax1.twinx()

    # Set its own y-label
    ax2.set_ylabel("Difference from Exact Value", color="black")
     
    # data for second axis is difference between the first two traces
    second_data = []  
    for i in range(len(y_data1)):
        if y_data1[i] == None or y_data2[i] == None:
            delta = None
        else:
            delta = y_data2[i] - y_data1[i]
        second_data.append(delta)
    
    if second_data and len(second_data) > 0 and second_data[0] is not None:
    
        # auto-center and shrink the range of second axis' data
        ymin = min([x for x in second_data if x is not None])
        ymax = max([x for x in second_data if x is not None])
        ydelta = ymax - ymin
        #print(f"min = {ymin}, {ymax}, {ydelta}")
        
        if ymin >= 0 and ymax > 0:
            ymin = 0.0 
            ymax = ymax * 2.0
        elif ymin < 0 and ymax <= 0:        
            ymin = ymin * 2.0 + (ydelta / 2)
            ymax = 0.0 + (ydelta / 2)
        else:
            ymid = (ymin + ymax) / 2.0
            ymin = ymid - ydelta
            ymax = ymid + ydelta 
            
            # move the second plot down by 5 %
            ymin += 0.1 * ydelta
            ymax += 0.1 * ydelta
            
    else:
        ymin = ymax = 0.0
        
    if ymin == ymax:
        ymax += 0.01
    
    # plot the data for second axis (difference)
    """
    color = _alt_colors[j] if j < len(_alt_colors) else _alt_colors[-1]
    marker = _alt_markers[j] if j < len(_alt_markers) else _alt_markers[-1]
    style = _alt_styles[j] if j < len(_alt_styles) else _alt_styles[-1]
    """
    color = "blueviolet"
    style = "dashed"
    marker = "."
    
    ax2.plot(x_data[:len(y_data1)], second_data[:len(y_data1)], label="Difference", #label=second_labels[j],
             linestyle=style, marker=marker, color=color, linewidth=0.5)
    
    ax2.set_ylim(ymin, ymax)
    
    ##############
    
    # Manually merge legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)  # Combine and display
    
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
    fig1, ax1 = plt.subplots(1, 1, figsize=(7.2, 5.0))
    
    # and add the title (shifted to the right a bit if single column plot)
    fig1.suptitle(suptitle, fontsize=13, x=(0.5 if plot_layout_style=='grid' else 0.54))
    
    #ax1.set_title('Plot of Two Data Sets')
    
    ###### Plot the metrics
    
    if not groups:
        return
    
    # set the axis labels
    ax1.set_xlabel("Number of Qubits")
    ax1.set_ylabel("Expectation Value")
    
    # add the background grid
    ax1.grid(True, axis = 'y', which='major', color='silver', zorder = 0)
    
    for i in range(0, len(values)):       
        if len(values[i]) < 1:
            continue

        color = _colors[i] if i < len(_colors) else _colors[-1]
        marker = _markers[i] if i < len(_markers) else _markers[-1]
        style = _styles[i] if i < len(_styles) else _styles[-1]
        
        ax1.plot(groups[i][:len(values[i])], values[i], label=labels[i],
                linestyle=style, marker=marker, color=color)

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
    suptitle = append_options_to_title(suptitle, options, backend_id)
    
    subtitle = ""
    
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
    fig1, ax1 = plt.subplots(1, 1, figsize=(7.2, 5.0))
    
    # and add the title (shifted to the right a bit if single column plot)
    fig1.suptitle(suptitle, fontsize=13, x=(0.5 if plot_layout_style=='grid' else 0.54))
    
    #ax1.set_title('Plot of Two Data Sets')
    
    ###### Plot the metrics
    
    if not groups:
        return
    
    # set the axis labels
    ax1.set_xlabel("Number of Qubits")
    ax1.set_ylabel("Expectation Compute Time (sec)")
    
    # add the background grid
    ax1.grid(True, axis = 'y', which='major', color='silver', zorder = 0)
    
    for i in range(0, len(times)):
        if len(times[i]) < 1:
            continue
        
        color = _colors[i] if i < len(_colors) else _colors[-1]
        marker = _markers[i] if i < len(_markers) else _markers[-1]
        style = _styles[i] if i < len(_styles) else _styles[-1]
        lwidth = 1.5 if i > 0 else 2.0
        
        ax1.plot(groups[i][:len(times[i])], times[i], label=labels[i],
                    linewidth=lwidth, linestyle=style, marker=marker, color=color)
    
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

##########################################################
# SUPPORT FUNCTIONS

def append_options_to_title(suptitle: str, options:list, backend_id:str):

    # these options are required
    hamiltonian_name = options['ham']
    if not hamiltonian_name: hamiltonian_name = "?"
    
    hamiltonian_params = options['params'] if 'params' in options else {}
    if not hamiltonian_params: hamiltonian_params = {}
    ham_params = ",".join([f"{k}:{v}" for k, v in hamiltonian_params.items()])
    
    num_shots = options['shots']
    reps = options['reps']
    K = options['K'] if 'K' in options else '?'
    t = options['t'] if 't' in options else '?'
    t = round(t, 3)
    gm = options['gm'] if 'gm' in options else '?'
    
    # Create standard title for all plots
    suptitle += f"\nHam={hamiltonian_name}"
    suptitle += f" {ham_params}"
    suptitle += f" K={K}, t={t}, gm={gm}"

    suptitle = suptitle + f"\nDevice={backend_id}"
    suptitle += f", shots={num_shots}, reps={reps}"
    
    return suptitle
        

##########################################################
# VALUE ANALYSIS PLOT FUNCTIONS

#from metric_plots import plot_values_scatter, plot_value_counts
#from metric_plots import plot_value_error, visualize_error_distribution

def plot_value_analysis_data(
        hamiltonian_name: str,
        backend_id: str,
        num_qubits: int,
        group_method: str,
        num_shots: int,
        init_values,
        exact_energies,
        computed_energies
    ):

    # Optionally plot raw observable values, both exact and computed, using a scatter plot
    if len(init_values) == len(computed_energies):
        plot_values_scatter(init_values, exact_energies, computed_energies)

    # Plot a spectrum of the observable values (this needs work yet)
    # plot_value_counts(computed_energies)

    # Plot the difference between computed and exact obverable values
    plot_value_error(
            hamiltonian_name,
            backend_id,
            num_qubits,
            group_method,
            num_shots,
            exact_energies,
            computed_energies
        )

    # Plot the distribution of errors and compute mean and sigma
    visualize_error_distribution(
            hamiltonian_name,
            backend_id,
            num_qubits,
            group_method,
            num_shots,
            np.array(exact_energies) - np.array(computed_energies)
        )
    

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def plot_value_error(
        hamiltonian_name: str,
        backend_id: str,
        num_qubits: int,
        group_method: str,
        num_shots: int,
        exact_energies,
        computed_energies
    ):
    """
    Plots the difference (computed - exact) against exact_energies.

    Args:
        exact_energies (list or array): Exact energy values (x-axis).
        computed_energies (list or array): Computed energy values.

    """
    if len(exact_energies) != len(computed_energies):
        raise ValueError("Arrays must have the same length.")

    base_ham_name = os.path.basename(hamiltonian_name)
    
    # Compute the difference (error)
    errors = [computed - exact for computed, exact in zip(computed_energies, exact_energies)]

    # Create scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(exact_energies, errors, color='red', alpha=0.7, edgecolors='black', label='Computed - Exact')

    # Labels and title
    title = "Error in Computed Energy vs Exact Energy"
    title += f"\nHam={base_ham_name}, qubits={num_qubits}, gm={group_method}, shots={num_shots}" 
    plt.title(title)
    
    plt.xlabel("Exact Energy")
    plt.ylabel("Computed - Exact Energy (Error)")
    plt.axhline(0, color='gray', linestyle='--')  # Add horizontal line at 0 for reference
    plt.grid(True)

    ymin = min(errors)
    ymax = max(errors)
    ydelta = ymax - ymin
    ymin -= 0.10 * ydelta
    ymax += 0.10 * ydelta
    
    #plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)  

    plt.legend()

    if group_method: group_method = group_method.replace(":", "-")
    imagename = f"{base_ham_name}-{num_qubits}-{group_method}-value-error"
    metrics.save_plot_image(plt, imagename, backend_id)

    plt.show()


def visualize_error_distribution(
        hamiltonian_name: str,
        backend_id: str,
        num_qubits: int,
        group_method: str,
        num_shots: int,
        errors
    ):
    """
    Plots the histogram of the error distribution and overlays key statistical metrics.
    Works for any distribution (not just Gaussian).

    Args:
        errors (list or array): List of error values.
    """
    base_ham_name = os.path.basename(hamiltonian_name)
    
    title = "Error Distribution with Key Metrics"
    title += f"\nHam={base_ham_name}, qubits={num_qubits}, gm={group_method}, shots={num_shots}" 
    
    errors = np.array(errors)

    # Compute key error metrics
    std_dev = np.std(errors)  # Standard deviation (σ)
    mean_error = np.mean(errors)  # Mean error (detects bias)
    mae = np.mean(np.abs(errors))  # Mean Absolute Error (MAE)
    rmse = np.sqrt(np.mean(errors**2))  # Root Mean Square Error (RMSE)

    # Compute Full Width at Half Maximum (FWHM) estimate
    fwhm = 2.355 * std_dev  # Approximate FWHM if shape is roughly symmetric

    # Create histogram without assuming normality
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=40, color='blue', alpha=0.6, edgecolor='black')

    plt.title(title)
    
    # Plot key statistics
    plt.axvline(mean_error, color='red', linestyle="--", label=f"Mean (Bias) = {mean_error:.4f}")
    plt.axvline(mean_error + std_dev, color='orange', linestyle="--", label=f"σ = {std_dev:.4f}")
    plt.axvline(mean_error - std_dev, color='orange', linestyle="--")
    plt.axvline(mean_error + fwhm/2, color='green', linestyle="--", label=f"FWHM/2 = {fwhm/2:.4f}")
    plt.axvline(mean_error - fwhm/2, color='green', linestyle="--")

    # Labels & legend
    plt.xlabel("Error Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    if group_method: group_method = group_method.replace(":", "-")
    imagename = f"{base_ham_name}-{num_qubits}-{group_method}-value-error-distr"
    metrics.save_plot_image(plt, imagename, backend_id)

    plt.show()

    # Print error bias analysis
    bias_direction = "positive" if mean_error > 0 else "negative"
    print("\n🔹 **Bias Detection**")
    print(f"Mean Error (Bias) = {mean_error:.4f}")
    
    # More general bias detection without assuming Gaussian shape
    if abs(mean_error) > 0.5 * (np.max(errors) - np.min(errors)) / 4:  # Using range-based threshold
        print(f"⚠️ Warning: Significant {bias_direction} bias detected in the error distribution!")
    else:
        print("✅ No significant bias detected.")


# DEVNOTE: These function need title and code to save to imagefile

def plot_values_scatter(x_values, y_values, y_values_2):
    """
    Plots a scatter plot with x_values on the x-axis and y_values on the y-axis.

    Args:
        x_values (list or array): Values for the x-axis.
        y_values (list or array): Values for the y-axis.
    """
    if len(x_values) != len(y_values):
        raise ValueError("Both input arrays must have the same length.")

    plt.figure(figsize=(8, 5))  # Set figure size
    plt.scatter(x_values, y_values, color='coral', alpha=0.7, edgecolors='black', label='Exact')
    plt.scatter(x_values, y_values_2, color='blue', alpha=0.7, edgecolors='black', label='Computed')

    plt.xlabel("Initial State (int)")
    plt.ylabel("Y Values")
    plt.ylabel("Expectation Value")
    plt.title("Expectation Value over Initial States")
    plt.grid(True)

    plt.legend()

    plt.show()
    
def plot_value_counts(x_values):
    """
    Plots a scatter plot where the x-axis represents unique values from x_values,
    and the y-axis represents their frequency (number of occurrences).

    Args:
        x_values (list or array): Values for the x-axis.
    """
    # Count occurrences of each unique value
    value_counts = Counter(x_values)

    # Extract unique values and their counts
    unique_x = list(value_counts.keys())
    counts = list(value_counts.values())

    # Create scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(unique_x, counts, color='blue', alpha=0.7, edgecolors='black')

    # Labels and title
    plt.xlabel("Unique Values")
    plt.ylabel("Frequency")
    plt.title("Scatter Plot of Value Counts")
    plt.grid(True)

    plt.show()



##########################################################
# TIMING ANALYSIS FUNCTIONS    

def plot_timing_analysis_bar_chart(
        hamiltonian_name: str,
        backend_id: str,
        num_qubits: int,
        group_method: str,
        num_shots: int,
        datasets: list, 
        dataset_labels: list, 
        categories: list,
        x_label = "Observable Computation / Grouping Method", 
        error_bar_position = "middle"
        ):
    """
    Plots a stacked bar chart for multiple datasets with standard deviations.

    Args:
        datasets (list of dicts): List of dictionaries containing mean & stddev values for each dataset.
        dataset_labels (list of dicts): List containing metadata (e.g., name, label) for each dataset.
        x_label (str): Label for the x-axis.
        error_bar_position (str): "middle" for stddev markers at the middle, "top" for error bars at the top.
    """
    if not datasets or len(datasets) != len(dataset_labels):
        raise ValueError("Datasets and labels must have the same length.")

    base_ham_name = os.path.basename(hamiltonian_name)
    
    #title = "Error Distribution with Key Metrics"
    title = "Stacked Bar Chart of Timing Metrics"
    title += f"\nHam={base_ham_name}, qubits={num_qubits}, gm={group_method}, shots={num_shots}" 

    # Define categories (excluding stddev fields)
    #categories = [key for key in datasets[0] if not key.endswith("_stddev") and key != "exact_time" and key != 'execute_circuits_time' and key != 'total_time'  ]
    if not categories:
        categories = [key for key in datasets[0] if not key.endswith("_stddev")]

    # Scale bar width dynamically
    num_datasets = len(datasets)
    #bar_width = max(0.2, 0.8 / num_datasets)  # Ensures bars are visible and not too thin
    bar_width = 0.6
    
    xmin = 0.0 - (1.0 - bar_width / 2.0)
    xmax = len(datasets) - bar_width / 2.0

    x_positions = np.arange(num_datasets)  # X positions for each dataset

    # x_positions = np.array([0]) if num_datasets == 1 else np.arange(num_datasets)  # Fix for single dataset
    # bar_width = 0.5 if num_datasets > 1 else 0.3  # Adjust bar width for single dataset

    # Define colors for different categories
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

    # we are currently only using the color by its index, not the key name
    """
    colors = {
        "create_base_time": "#A7C7E7",  # Soft Sky Blue
        "append_measurements_time": "#B5D6B2",  # Light Sage Green
        "execute_circuits_time": "#809BCE",  # Muted Blue
        "observable_compute_time": "#5D6D7E",  # Slate Gray
        "total_time": "#5B84B1"  # Deep Cool Blue
    }
    """
    colors = {
        "create_base_time": "#1B4965",  # Deep Steel Blue
        "append_measurements_time": "#5FA8D3",  # Muted Sky Blue
        "execute_circuits_time": "#9FC6E7",  # Soft Light Blue
        "observable_compute_time": "#5F7367",  # Subtle Green-Gray
        "total_time": "#2E5E4E",  # Deep Forest Green
        "more_1": "#1B4965",  # Deep Steel Blue
        "more_2": "#1B4965",  # Deep Steel Blue
    }

    #print(colors)
    color_list = list(colors.values())  # Convert dictionary values to a list    

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Initialize bottoms for stacking
    bottoms = np.zeros(num_datasets)

    ymax = 0.01
    ymax_cat = 0.0
    all_values = None
    
    for i, category in enumerate(categories):
        values = np.array([dataset[category] for dataset in datasets])
        stddevs = np.array([dataset.get(f"{category}_stddev", 0) for dataset in datasets])

        # Plot stacked bars
        ax.bar(x_positions, values, bar_width, label=category.replace("_", " ").title(), bottom=bottoms, color=color_list[i], align='center')
        
        all_values = values if all_values is None else (all_values + values)
        
        #ymax = max(ymax, max(sum(values)))

        # bar_width = min(0.8, 0.6 / num_datasets)  # Keeps bars visible but prevents full-width issue
        # x_positions = np.linspace(-bar_width * (num_datasets - 1) / 2, bar_width * (num_datasets - 1) / 2, num_datasets)  # Evenly space bars
        # ax.bar(x_positions + i, values, bar_width, label=category.replace("_", " ").title(), bottom=bottoms, color=color_list[i])

        # Plot standard deviation
        if error_bar_position == "middle":
            # Place markers at the middle of each bar segment
            ax.errorbar(x_positions, bottoms + values / 2, yerr=stddevs, fmt='o', color='black', capsize=4, alpha=0.6)
        elif error_bar_position == "top":
            # Place error bars at the top of each bar segment
            ax.errorbar(x_positions, bottoms + values, yerr=stddevs, fmt='none', ecolor='black', capsize=5, alpha=0.8)

        # Update stacking positions
        bottoms += values
   
    ymax = max(ymax, max(all_values))
        
    # X-axis labels
    dataset_labels_formatted = [d["label"] for d in dataset_labels]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dataset_labels_formatted)

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time (seconds)")
    ax.set_title(title)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, ymax * 1.4)

    # Legend
    #ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Add legend, but reverse its order to match the stacked bars
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1))
    ax.legend(handles[::-1], labels[::-1])

    # Grid for readability
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if "execute_circuits_time" in categories:
        imagename = f"{base_ham_name}-{num_qubits}-timing-x-metrics"
    else:
        imagename = f"{base_ham_name}-{num_qubits}-timing-metrics"
    metrics.save_plot_image(plt, imagename, backend_id)
    
    plt.show()


##########################################################
# OTHER ANALYSIS FUNCTIONS - NOT USED CURRENTLY

