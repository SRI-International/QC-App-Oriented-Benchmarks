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
# Metrics Module
#
# This module contains methods to initialize, store, aggregate and
# plot metrics collected in the benchmark programs
#
# Metrics are indexed by group id (e.g. circuit size), circuit id (e.g. secret string)
# and metric name.
# Multiple metrics with different names may be stored, each with a single value.
#
# The 'aggregate' method accumulates all circuit-specific metrics for each group
# and creates an average that may be reported and plotted across all groups
#

import os
import json
import time
from time import gmtime, strftime
from datetime import datetime

# Raw and aggregate circuit metrics
circuit_metrics = {  }
group_metrics = { "groups": [],
    "avg_create_times": [], "avg_elapsed_times": [], "avg_exec_times": [], "avg_fidelities": [],
    "avg_depths": [], "avg_xis": [], "avg_tr_depths": [], "avg_tr_xis": [], "avg_tr_n2qs": [],
    "avg_exec_creating_times": [], "avg_exec_validating_times": [], "avg_exec_running_times": []
}

# Additional properties
_properties = { "api":"unknown", "backend_id":"unknown"}

# times relevant to an application run
start_time = 0 
end_time = 0

##### Options

# Print more detailed metrics info
verbose = False

# Option to save metrics to data file
save_metrics = True 

# Option to save plot images (all of them)
save_plot_images = True

# Option to generate volumetric positioning charts
do_volumetric_plots = True

# Option to include all app charts with vplots at end
do_app_charts_with_all_metrics = False

# Number of ticks on volumetric depth axis
max_depth_log = 22

# Quantum Volume to display on volumetric background
AQ=22
QV = 0
aq_cutoff=0.368

# average transpile factor between base QV depth and our depth based on results from QV notebook
QV_transpile_factor = 12.7     

# Base for volumetric plot logarithmic axes
#depth_base = 1.66  # this stretches depth axis out, but other values have issues:
#1) need to round to avoid duplicates, and 2) trailing zeros are getting removed 
depth_base = 2


##### Initialize methods

# Set a subtitle for the Chart
def set_plot_subtitle (subtitle=None):
    circuit_metrics["subtitle"] = subtitle
    
# Set properties context for this set of metrics
def set_properties ( properties=None ):
    global _properties
    
    if properties == None:
        _properties = { "api":"unknown", "backend_id":"unknown" }
    else:
        _properties = properties
       
# Initialize the metrics module, creating an empty table of metrics
def init_metrics ():
    global start_time
    
    # create empty dictionary for circuit metrics
    circuit_metrics.clear()
    
    # create empty arrays for group metrics
    group_metrics["groups"] = []
    
    group_metrics["avg_create_times"] = []
    group_metrics["avg_elapsed_times"] = []
    group_metrics["avg_exec_times"] = []
    group_metrics["avg_fidelities"] = []
    
    group_metrics["avg_depths"] = []
    group_metrics["avg_xis"] = []
    group_metrics["avg_tr_depths"] = []
    group_metrics["avg_tr_xis"] = []
    group_metrics["avg_tr_n2qs"] = []
    
    group_metrics["avg_exec_creating_times"] = []
    group_metrics["avg_exec_validating_times"] = []
    group_metrics["avg_exec_running_times"] = []
    
    # store the start of execution for the current app
    start_time = time.time()
    print(f'... execution starting at {strftime("%Y-%m-%d %H:%M:%S", gmtime())}')

# End metrics collection for an application
def end_metrics():
    global end_time

    end_time = time.time()
    print(f'... execution complete at {strftime("%Y-%m-%d %H:%M:%S", gmtime())}')
    print("")
    
    
##### Metrics methods

# Store an individual metric associate with a group and circuit in the group
def store_metric (group, circuit, metric, value):
    group = str(group)
    circuit = str(circuit)
    if group not in circuit_metrics:
        circuit_metrics[group] = { }
    if circuit not in circuit_metrics[group]:
        circuit_metrics[group][circuit] = { }
    circuit_metrics[group][circuit][metric] = value
    #print(f'{group} {circuit} {metric} -> {value}')


# Aggregate metrics for a specific group, creating average across circuits in group
def aggregate_metrics_for_group (group):
    group = str(group)
    
    # generate totals, then divide by number of circuits to calculate averages    
    if group in circuit_metrics:
        num_circuits = 0
        group_create_time = 0
        group_elapsed_time = 0
        group_exec_time = 0
        group_fidelity = 0
        group_depth = 0
        group_xi = 0
        group_tr_depth = 0
        group_tr_xi = 0
        group_tr_n2q = 0
        group_exec_creating_time = 0
        group_exec_validating_time = 0
        group_exec_running_time = 0

        # loop over circuits in group to generate totals
        for circuit in circuit_metrics[group]:
            num_circuits += 1
            for metric in circuit_metrics[group][circuit]:
                value = circuit_metrics[group][circuit][metric]
                #print(f'{group} {circuit} {metric} -> {value}')
                if metric == "create_time": group_create_time += value
                if metric == "elapsed_time": group_elapsed_time += value
                if metric == "exec_time": group_exec_time += value
                if metric == "fidelity": group_fidelity += value
                
                if metric == "depth": group_depth += value
                if metric == "xi": group_xi += value
                if metric == "tr_depth": group_tr_depth += value
                if metric == "tr_xi": group_tr_xi += value
                if metric == "tr_n2q": group_tr_n2q += value
                
                if metric == "exec_creating_time": group_exec_creating_time += value
                if metric == "exec_validating_time": group_exec_validating_time += value
                if metric == "exec_running_time": group_exec_running_time += value

        # calculate averages
        avg_create_time = round(group_create_time / num_circuits, 3)
        avg_elapsed_time = round(group_elapsed_time / num_circuits, 3)
        avg_exec_time = round(group_exec_time / num_circuits, 3)
        avg_fidelity = round(group_fidelity / num_circuits, 3)
        
        avg_depth = round(group_depth / num_circuits, 0)
        avg_xi = round(group_xi / num_circuits, 3)
        avg_tr_depth = round(group_tr_depth / num_circuits, 0)
        avg_tr_xi = round(group_tr_xi / num_circuits, 3)
        avg_tr_n2q = round(group_tr_n2q / num_circuits, 3)
        
        avg_exec_creating_time = round(group_exec_creating_time / num_circuits, 3)
        avg_exec_validating_time = round(group_exec_validating_time / num_circuits, 3)
        avg_exec_running_time = round(group_exec_running_time / num_circuits, 3)
        
        # store averages in arrays structured for reporting and plotting by group
        group_metrics["groups"].append(group)
        
        group_metrics["avg_create_times"].append(avg_create_time)
        group_metrics["avg_elapsed_times"].append(avg_elapsed_time)
        group_metrics["avg_exec_times"].append(avg_exec_time)
        group_metrics["avg_fidelities"].append(avg_fidelity)
        
        if avg_depth > 0:
            group_metrics["avg_depths"].append(avg_depth)
        if avg_xi > 0:
            group_metrics["avg_xis"].append(avg_xi)
        if avg_tr_depth > 0:
            group_metrics["avg_tr_depths"].append(avg_tr_depth)
        if avg_tr_xi > 0:
            group_metrics["avg_tr_xis"].append(avg_tr_xi)
        if avg_tr_n2q > 0:
            group_metrics["avg_tr_n2qs"].append(avg_tr_n2q)
        
        if avg_exec_creating_time > 0:
            group_metrics["avg_exec_creating_times"].append(avg_exec_creating_time)
        if avg_exec_validating_time > 0:
            group_metrics["avg_exec_validating_times"].append(avg_exec_validating_time)
        if avg_exec_running_time > 0:
            group_metrics["avg_exec_running_times"].append(avg_exec_running_time)

    
# Aggregate all metrics by group
def aggregate_metrics ():
    for group in circuit_metrics:
        aggregate_metrics_for_group(group)


# Report metrics for a specific group
def report_metrics_for_group (group):
    group = str(group)
    if group in group_metrics["groups"]:
        group_index = group_metrics["groups"].index(group)
        if group_index >= 0:
            avg_xi = 0
            if len(group_metrics["avg_xis"]) > 0:
                avg_xi = group_metrics["avg_xis"][group_index]

            if len(group_metrics["avg_depths"]) > 0:
                avg_depth = group_metrics["avg_depths"][group_index]
                if avg_depth > 0:
                    print(f"Average Depth, \u03BE (xi) for the {group} qubit group = {int(avg_depth)}, {avg_xi}")
            
            avg_tr_xi = 0
            if len(group_metrics["avg_tr_xis"]) > 0:
                avg_tr_xi = group_metrics["avg_tr_xis"][group_index]
                
            avg_tr_n2q = 0
            if len(group_metrics["avg_tr_n2qs"]) > 0:
                avg_tr_n2q = group_metrics["avg_tr_n2qs"][group_index]
            
            if len(group_metrics["avg_tr_depths"]) > 0:
                avg_tr_depth = group_metrics["avg_tr_depths"][group_index]
                if avg_tr_depth > 0:
                    print(f"Average Transpiled Depth, \u03BE (xi), 2q gates for the {group} qubit group = {int(avg_tr_depth)}, {avg_tr_xi}, {avg_tr_n2q}")
                    
            avg_create_time = group_metrics["avg_create_times"][group_index]
            print(f"Average Creation Time for the {group} qubit group = {avg_create_time} secs")
            avg_elapsed_time = group_metrics["avg_elapsed_times"][group_index]
            print(f"Average Elapsed Time for the {group} qubit group = {avg_elapsed_time} secs")
            avg_exec_time = group_metrics["avg_exec_times"][group_index]
            print(f"Average Execution Time for the {group} qubit group = {avg_exec_time} secs")
            
            #if verbose:
            if len(group_metrics["avg_exec_creating_times"]) > 0:
                avg_exec_creating_time = group_metrics["avg_exec_creating_times"][group_index]
                #if avg_exec_creating_time > 0:
                    #print(f"Average Creating Time for group {group} = {avg_exec_creating_time}")
                    
                if len(group_metrics["avg_exec_validating_times"]) > 0:
                    avg_exec_validating_time = group_metrics["avg_exec_validating_times"][group_index]
                    #if avg_exec_validating_time > 0:
                        #print(f"Average Validating Time for group {group} = {avg_exec_validating_time}")
                        
                if len(group_metrics["avg_exec_running_times"]) > 0:
                    avg_exec_running_time = group_metrics["avg_exec_running_times"][group_index]
                    #if avg_exec_running_time > 0:
                        #print(f"Average Running Time for group {group} = {avg_exec_running_time}")
                            
                print(f"Average Transpiling, Validating, Running Times for group {group} = {avg_exec_creating_time}, {avg_exec_validating_time}, {avg_exec_running_time} secs")
            
            avg_fidelity = group_metrics["avg_fidelities"][group_index]
            print(f"Average Fidelity for the {group} qubit group = {avg_fidelity}")
            
            print("")
            return
            
    # if group metrics not found       
    print("")
    print(f"no metrics for group: {group}")
        
# Report all metrics for all groups
def report_metrics ():   
    # loop over all groups and print metrics for that group
    for group in circuit_metrics:
        report_metrics_for_group(group)
        
# Aggregate and report on metrics for the given groups, if all circuits in group are complete
def finalize_group(group):

    #print(f"... finalize group={group}")

    # loop over circuits in group to generate totals
    group_done = True
    for circuit in circuit_metrics[group]:
        #print(f"  ... metrics = {group} {circuit} {circuit_metrics[group][circuit]}")
        
        if "elapsed_time" not in circuit_metrics[group][circuit]:
            group_done = False
            break
    
    #print(f"  ... group_done = {group} {group_done}")
    if group_done:
        aggregate_metrics_for_group(group)
        print("************")
        report_metrics_for_group(group)
        
    # sort the group metrics (sometimes they come back out of order)
    sort_group_metrics()
        
        
# sort the group array as integers, then all metrics relative to it
def sort_group_metrics():

    # get groups as integer, then sort each metric with it
    igroups = [int(group) for group in group_metrics["groups"]]
    for key in group_metrics:
        if key == "groups": continue
        xy = sorted(zip(igroups, group_metrics[key]))
        group_metrics[key] = [y for x, y in xy]
        
    # save the sorted group names when all done 
    xy = sorted(zip(igroups, group_metrics["groups"]))    
    group_metrics["groups"] = [y for x, y in xy]
    

##########################################
# ANALYSIS AND VISUALIZATION

import matplotlib.pyplot as plt
    
# Plot bar charts for each metric over all groups
def plot_metrics (suptitle="Circuit Width (Number of Qubits)", transform_qubit_group = False, new_qubit_group = None, filters=None, suffix=""):
    
    subtitle = circuit_metrics["subtitle"]
    
    # Extract shorter app name from the title passed in by user
    appname = suptitle[len('Benchmark Results - '):len(suptitle)]
    appname = appname[:appname.index(' - ')]
    
    # for creating plot image filenames replace spaces
    appname = appname.replace(' ', '-')
    
    backend_id = subtitle[9:]   

    # save the metrics for current application to the DATA file, one file per device
    if save_metrics:
        #data = group_metrics
        #filename = f"DATA-{subtitle[9:]}.json"
        #title = suptitle

        # If using mid-circuit transformation, convert old qubit group to new qubit group
        if transform_qubit_group:
            original_data = group_metrics["groups"]
            group_metrics["groups"] = new_qubit_group
            store_app_metrics(backend_id, circuit_metrics, group_metrics, suptitle,
                start_time=start_time, end_time=end_time)
            group_metrics["groups"] = original_data
        else:
            store_app_metrics(backend_id, circuit_metrics, group_metrics, suptitle,
                start_time=start_time, end_time=end_time)

        
    if len(group_metrics["groups"]) == 0:
        print(f"\n{suptitle}")
        print(f"     ****** NO RESULTS ****** ")
        return
    
    # sort the group metrics (in case they weren't sorted when collected)
    sort_group_metrics()
    
    # flags for charts to show
    do_creates = True
    do_executes = True
    do_fidelities = True
    do_depths = True
    do_vbplot = True
    
    # check if we have depth metrics to show
    do_depths = len(group_metrics["avg_depths"]) > 0
    
    # if filters set, adjust these flags
    if filters != None:
        if "create" not in filters: do_creates = False
        if "execute" not in filters: do_executes = False
        if "fidelity" not in filters: do_fidelities = False
        if "depth" not in filters: do_depths = False
        if "vbplot" not in filters: do_vbplot = False
    
    # generate one-column figure with multiple bar charts, with shared X axis
    cols = 1
    fig_w = 6.0
    
    numplots = 0
    if do_creates: numplots += 1
    if do_executes: numplots += 1
    if do_fidelities: numplots += 1
    if do_depths: numplots += 1
    
    rows = numplots
    
    # DEVNOTE: this calculation is based on visual assessment of results and could be refined
    # compute height needed to draw same height plots, no matter how many there are
    fig_h = 3.5 + 2.0 * (rows - 1) + 0.25 * (rows - 1)
    #print(fig_h)
    
    # create the figure into which plots will be placed
    fig, axs = plt.subplots(rows, cols, sharex=True, figsize=(fig_w, fig_h))
    
    # append the circuit metrics subtitle to the title
    timestr = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    realtitle = suptitle + f"\nDevice={backend_id}  {timestr} UTC"
    '''
    realtitle = suptitle
    if subtitle != None:
        realtitle += ("\n" + subtitle)
    '''    
    plt.suptitle(realtitle)
    
    axi = 0
    xaxis_set = False
    
    if rows == 1:
        ax = axs
        axs = [ax]
    
    if do_creates:
        if max(group_metrics["avg_create_times"]) < 0.01:
            axs[axi].set_ylim([0, 0.01])
        axs[axi].bar(group_metrics["groups"], group_metrics["avg_create_times"])
        axs[axi].set_ylabel('Avg Creation Time (sec)')
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        plt.setp(axs[axi].get_xticklabels(), visible=False)
        axi += 1
    
    if do_executes:
        if max(group_metrics["avg_exec_times"]) < 0.1:
            axs[axi].set_ylim([0, 0.1])
        axs[axi].bar(group_metrics["groups"], group_metrics["avg_exec_times"])
        axs[axi].set_ylabel('Avg Execution Time (sec)')
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        # none of these methods of sharing the x axis gives proper effect; makes extra white space
        #axs[axi].sharex(axs[2])
        #plt.setp(axs[axi].get_xticklabels(), visible=False)
        #axs[axi].set_xticklabels([])
        axi += 1
    
    if do_fidelities:
        axs[axi].set_ylim([0, 1.0])
        axs[axi].bar(group_metrics["groups"], group_metrics["avg_fidelities"]) 
        axs[axi].set_ylabel('Avg Result Fidelity')
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        axi += 1
    
    if do_depths:
        if max(group_metrics["avg_tr_depths"]) < 20:
            axs[axi].set_ylim([0, 20])  
        axs[axi].bar(group_metrics["groups"], group_metrics["avg_depths"], 0.8)
        axs[axi].bar(group_metrics["groups"], group_metrics["avg_tr_depths"], 0.5, color='C9') 
        axs[axi].set_ylabel('Circuit Depth')
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        axs[axi].legend(['Circuit Depth', 'Transpiled Depth'], loc='upper left')
        axi += 1
    
    # shared x axis label
    axs[rows - 1].set_xlabel('Circuit Width (Number of Qubits)')
     
    fig.tight_layout() 
    
    # save plot image to file
    if save_plot_images:
        save_plot_image(plt, f"{appname}-metrics" + suffix, backend_id) 
            
    # show the plot for user to see
    plt.show()
    
    ###################### Volumetric Plot
    
    timestr = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    
    suptitle = f"Volumetric Positioning - {appname}\nDevice={backend_id}  {timestr} UTC"
    
    global cmap   
    
    # note: if using filters, both "depth" and "vbplot" must be set for this to draw
    
    # generate separate figure for volumetric positioning chart of depth metrics
    # found it difficult to share the x axis with first 3, but have diff axis for this one
    if do_depths and do_volumetric_plots and do_vbplot:
        
        w_data = group_metrics["groups"]
        d_tr_data = group_metrics["avg_tr_depths"]
        f_data = group_metrics["avg_fidelities"]
        
        try:
            #print(f"... {d_data} {d_tr_data}")
            
            vplot_anno_init()
            
            max_qubits = max([int(group) for group in w_data])
            
            ax = plot_volumetric_background(max_qubits, QV, depth_base, suptitle=suptitle)
            
            # determine width for circuit
            w_max = 0
            for i in range(len(w_data)):
                y = float(w_data[i])
                w_max = max(w_max, y)

            cmap = cmap_spectral

            # If using mid-circuit transformation, convert width data to singular circuit width value
            if transform_qubit_group:
                w_data = new_qubit_group
                group_metrics["groups"] = w_data

            plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base, fill=True,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)  
            
            anno_volumetric_data(ax, depth_base,
                label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False)
        
        except Exception as e:
            print(f'ERROR: failure when creating volumetric positioning chart')
            print(f"... exception = {e}")
        
        # save plot image to file
        if save_plot_images:
            save_plot_image(plt, f"{appname}-vplot", backend_id) 
        
        #display plot
        plt.show()       
    
# Plot bar charts for each metric over all groups
def plot_metrics_aq (suptitle="Circuit Width (Number of Qubits)", transform_qubit_group = False, new_qubit_group = None, filters=None, suffix=""):
    
    subtitle = circuit_metrics["subtitle"]
    
    # Extract shorter app name from the title passed in by user
    appname = suptitle[len('Benchmark Results - '):len(suptitle)]
    appname = appname[:appname.index(' - ')]
    
    # for creating plot image filenames replace spaces
    appname = appname.replace(' ', '-')
    
    backend_id = subtitle[9:]   

    # save the metrics for current application to the DATA file, one file per device
    if save_metrics:
        #data = group_metrics
        #filename = f"DATA-{subtitle[9:]}.json"
        #title = suptitle

        # If using mid-circuit transformation, convert old qubit group to new qubit group
        if transform_qubit_group:
            original_data = group_metrics["groups"]
            group_metrics["groups"] = new_qubit_group
            store_app_metrics(backend_id, circuit_metrics, group_metrics, suptitle,
                start_time=start_time, end_time=end_time)
            group_metrics["groups"] = original_data
        else:
            store_app_metrics(backend_id, circuit_metrics, group_metrics, suptitle,
                start_time=start_time, end_time=end_time)

        
    if len(group_metrics["groups"]) == 0:
        print(f"\n{suptitle}")
        print(f"     ****** NO RESULTS ****** ")
        return
    
    # sort the group metrics (in case they weren't sorted when collected)
    sort_group_metrics()
    
    # flags for charts to show
    do_creates = True
    do_executes = True
    do_fidelities = True
    do_depths = True
    do_vbplot = True
    
    # check if we have depth metrics to show
    do_depths = len(group_metrics["avg_depths"]) > 0
    
    # if filters set, adjust these flags
    if filters != None:
        if "create" not in filters: do_creates = False
        if "execute" not in filters: do_executes = False
        if "fidelity" not in filters: do_fidelities = False
        if "depth" not in filters: do_depths = False
        if "vbplot" not in filters: do_vbplot = False
    
    # generate one-column figure with multiple bar charts, with shared X axis
    cols = 1
    fig_w = 6.0
    
    numplots = 0
    if do_creates: numplots += 1
    if do_executes: numplots += 1
    if do_fidelities: numplots += 1
    if do_depths: numplots += 1
    
    rows = numplots
    
    # DEVNOTE: this calculation is based on visual assessment of results and could be refined
    # compute height needed to draw same height plots, no matter how many there are
    fig_h = 3.5 + 2.0 * (rows - 1) + 0.25 * (rows - 1)
    #print(fig_h)
    
    # create the figure into which plots will be placed
    fig, axs = plt.subplots(rows, cols, sharex=True, figsize=(fig_w, fig_h))
    
    # append the circuit metrics subtitle to the title
    timestr = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    realtitle = suptitle + f"\nDevice={backend_id}  {timestr} UTC"
    '''
    realtitle = suptitle
    if subtitle != None:
        realtitle += ("\n" + subtitle)
    '''    
    plt.suptitle(realtitle)
    
    axi = 0
    xaxis_set = False
    
    if rows == 1:
        ax = axs
        axs = [ax]
    
    if do_creates:
        if max(group_metrics["avg_create_times"]) < 0.01:
            axs[axi].set_ylim([0, 0.01])
        axs[axi].bar(group_metrics["groups"], group_metrics["avg_create_times"])
        axs[axi].set_ylabel('Avg Creation Time (sec)')
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        plt.setp(axs[axi].get_xticklabels(), visible=False)
        axi += 1
    
    if do_executes:
        if max(group_metrics["avg_exec_times"]) < 0.1:
            axs[axi].set_ylim([0, 0.1])
        axs[axi].bar(group_metrics["groups"], group_metrics["avg_exec_times"])
        axs[axi].set_ylabel('Avg Execution Time (sec)')
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        # none of these methods of sharing the x axis gives proper effect; makes extra white space
        #axs[axi].sharex(axs[2])
        #plt.setp(axs[axi].get_xticklabels(), visible=False)
        #axs[axi].set_xticklabels([])
        axi += 1
    
    if do_fidelities:
        axs[axi].set_ylim([0, 1.0])
        axs[axi].bar(group_metrics["groups"], group_metrics["avg_fidelities"]) 
        axs[axi].set_ylabel('Avg Result Fidelity')
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        axi += 1
    
    if do_depths:
        if max(group_metrics["avg_tr_depths"]) < 20:
            axs[axi].set_ylim([0, 20])  
        axs[axi].bar(group_metrics["groups"], group_metrics["avg_depths"], 0.8)
        axs[axi].bar(group_metrics["groups"], group_metrics["avg_tr_depths"], 0.5, color='C9') 
        axs[axi].set_ylabel('Circuit Depth')
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        axs[axi].legend(['Circuit Depth', 'Transpiled Depth'], loc='upper left')
        axi += 1
    
    # shared x axis label
    axs[rows - 1].set_xlabel('Circuit Width (Number of Qubits)')
     
    fig.tight_layout() 
    
    # save plot image to file
    if save_plot_images:
        save_plot_image(plt, f"{appname}-metrics" + suffix, backend_id) 
            
    # show the plot for user to see
    plt.show()
    
    ###################### Volumetric Plot
    
    timestr = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    
    suptitle = f"Volumetric Positioning - {appname}\nDevice={backend_id}  {timestr} UTC"
    
    global cmap   
    
    # note: if using filters, both "depth" and "vbplot" must be set for this to draw
    
    # generate separate figure for volumetric positioning chart of depth metrics
    # found it difficult to share the x axis with first 3, but have diff axis for this one
    if do_depths and do_volumetric_plots and do_vbplot:

        aq_metrics={}
        aq_metrics["groups"]=[]
        aq_metrics["tr_n2qs"]=[]
        aq_metrics["fidelities"]=[]
        for group in circuit_metrics:
            if group=='subtitle':
                continue
        
            for key in circuit_metrics[group]:
                aq_metrics["groups"].append(group)
                aq_metrics["tr_n2qs"].append(circuit_metrics[group][key]["tr_n2q"])
                aq_metrics["fidelities"].append(circuit_metrics[group][key]["fidelity"])
        
        w_data = aq_metrics["groups"]
        n2q_tr_data = aq_metrics["tr_n2qs"]
        f_data = aq_metrics["fidelities"]
        
        try:
            #print(f"... {d_data} {d_tr_data}")
            
            vplot_anno_init()
            
            max_qubits = max([int(group) for group in w_data])
            
            ax = plot_volumetric_background_aq(max_qubits=max_qubits, AQ=0, depth_base=depth_base, suptitle=suptitle)
            
            # determine width for circuit
            w_max = 0
            for i in range(len(w_data)):
                y = float(w_data[i])
                w_max = max(w_max, y)

            cmap = cmap_spectral

            # If using mid-circuit transformation, convert width data to singular circuit width value
            if transform_qubit_group:
                w_data = new_qubit_group
                aq_metrics["groups"] = w_data

            plot_volumetric_data_aq(ax, w_data, n2q_tr_data, f_data, depth_base, fill=True,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)  
            
            anno_volumetric_data(ax, depth_base,
                label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False)
        
        except Exception as e:
            print(f'ERROR: failure when creating volumetric positioning chart')
            print(f"... exception = {e}")
        
        # save plot image to file
        if save_plot_images:
            save_plot_image(plt, f"{appname}-vplot", backend_id) 
        
        #display plot
        plt.show()       

    
# Plot metrics over all groups (2)
def plot_metrics_all_overlaid (shared_data, backend_id, suptitle=None, imagename="_ALL-vplot-1"):    
    
    global circuit_metrics
    global group_metrics
    
    subtitle = circuit_metrics["subtitle"]
    
    print("Overlaid Results From All Applications")
    
    # generate separate figure for volumetric positioning chart of depth metrics
    # found it difficult to share the x axis with first 3, but have diff axis for this one
    
    try:
        #print(f"... {d_data} {d_tr_data}")
        
        # determine largest width for all apps
        w_max = 0
        for app in shared_data:
            group_metrics = shared_data[app]["group_metrics"]
            w_data = group_metrics["groups"]
            for i in range(len(w_data)):
                y = float(w_data[i])
                w_max = max(w_max, y)
        
        # allow one more in width to accommodate the merge values below
        max_qubits = int(w_max) + 1     
        #print(f"... {w_max} {max_qubits}")
        
        ax = plot_volumetric_background(max_qubits, QV, depth_base, suptitle=suptitle)
        
        vplot_anno_init()
        
        for app in shared_data:
            #print(shared_data[app])

            # Extract shorter app name from the title passed in by user
            appname = app[len('Benchmark Results - '):len(app)]
            appname = appname[:appname.index(' - ')]
            
            group_metrics = shared_data[app]["group_metrics"]
            #print(group_metrics)
            
            if len(group_metrics["groups"]) == 0:
                print(f"****** NO RESULTS for {appname} ****** ")
                continue

            # check if we have depth metrics
            do_depths = len(group_metrics["avg_depths"]) > 0
            if not do_depths:
                continue
                
            w_data = group_metrics["groups"]
            d_data = group_metrics["avg_depths"]
            d_tr_data = group_metrics["avg_tr_depths"]
            f_data = group_metrics["avg_fidelities"]
    
            plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base, fill=True,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)  

        # do annotation separately, spreading labels for readability
        anno_volumetric_data(ax, depth_base,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False)
    
    except Exception as e:
        print(f'ERROR: failure when creating volumetric positioning chart')
        print(f"... exception = {e}")
    
    # save plot image file
    if save_plot_images:
        save_plot_image(plt, imagename, backend_id) 
    
    #display plot
    plt.show()    


# Plot metrics over all groups (2)
def plot_metrics_all_overlaid_aq (shared_data, backend_id, suptitle=None, imagename="_ALL-vplot-1"):    
    
    global circuit_metrics
    global group_metrics
    
    subtitle = circuit_metrics["subtitle"]
    
    print("Overlaid Results From All Applications")
    
    # generate separate figure for volumetric positioning chart of depth metrics
    # found it difficult to share the x axis with first 3, but have diff axis for this one
    
    try:
        #print(f"... {d_data} {d_tr_data}")
        
        # determine largest width for all apps
        w_max = 0
        for app in shared_data:
            group_metrics = shared_data[app]["group_metrics"]
            w_data = group_metrics["groups"]
            for i in range(len(w_data)):
                y = float(w_data[i])
                w_max = max(w_max, y)
        
        # allow one more in width to accommodate the merge values below
        max_qubits = int(w_max) + 1     
        #print(f"... {w_max} {max_qubits}")
        
        ax = plot_volumetric_background_aq(max_qubits=max_qubits, AQ=AQ, depth_base=depth_base, suptitle=suptitle)
        
        vplot_anno_init()
        
        for app in shared_data:
            #print(shared_data[app])

            # Extract shorter app name from the title passed in by user
            appname = app[len('Benchmark Results - '):len(app)]
            appname = appname[:appname.index(' - ')]
            
            group_metrics = shared_data[app]["group_metrics"]
            #print(group_metrics)
            
            if len(group_metrics["groups"]) == 0:
                print(f"****** NO RESULTS for {appname} ****** ")
                continue

            # check if we have depth metrics
            do_depths = len(group_metrics["avg_depths"]) > 0
            if not do_depths:
                continue
                
            w_data = group_metrics["groups"]
            d_data = group_metrics["avg_depths"]
            d_tr_data = group_metrics["avg_tr_depths"]            
            n2q_tr_data = group_metrics["avg_tr_n2qs"]
            f_data = group_metrics["avg_fidelities"]
    
            plot_volumetric_data(ax, w_data, n2q_tr_data, f_data, depth_base, fill=True,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)  

        # do annotation separately, spreading labels for readability
        anno_volumetric_data(ax, depth_base,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False)
    
    except Exception as e:
        print(f'ERROR: failure when creating volumetric positioning chart')
        print(f"... exception = {e}")
    
    # save plot image file
    if save_plot_images:
        save_plot_image(plt, imagename, backend_id) 
    
    #display plot
    plt.show()    


def plot_metrics_all_merged_individual_aq (shared_data, backend_id, suptitle=None, imagename="_ALL-vplot-2", avail_qubits=0):    
      
    global circuit_metrics
    global aq_metrics
  
    # generate separate figure for volumetric positioning chart of depth metrics
    # found it difficult to share the x axis with first 3, but have diff axis for this one
    
    #print(f"... {max_depth_log}")
    
    #if True:
    try:
        #print(f"... {d_data} {d_tr_data}")
        
        # determine largest width for all apps
        w_max = 0
        for app in shared_data:
            aq_metrics = shared_data[app]["aq_metrics"]
            w_data = aq_metrics["groups"]
            for i in range(len(w_data)):
                y = float(w_data[i])
                w_max = max(w_max, y)
        
        #determine width for AQ
        AQ=w_max
        for app in shared_data:
            aq_metrics = shared_data[app]["aq_metrics"]
            w_data = aq_metrics["groups"]
            n2q_data = aq_metrics["tr_n2qs"]       
            fidelity_data=aq_metrics["fidelities"]
            while True:
                n2q_cutoff=AQ*AQ
                fail_w=[float(w_data[i]) for i,v in enumerate(n2q_data) if (float(v) < n2q_cutoff and float(fidelity_data[i])<aq_cutoff)] 
                if len(fail_w)==0:
                    break        
                AQ-=1
        
        # allow one more in width to accommodate the merge values below
        max_qubits = int(w_max) + 1     
        #print(f"... {w_max} {max_qubits}")
        
        ax = plot_volumetric_background_aq(max_qubits=max_qubits, AQ=AQ, depth_base=depth_base, suptitle=suptitle, avail_qubits=avail_qubits)
        
        # create 2D array to hold merged value arrays with gradations, one array for each qubit size
        num_grads = 4
        depth_values_merged = []
        for w in range(max_qubits):
            depth_values_merged.append([ None ] * (num_grads * max_depth_log))
        
        #print(depth_values_merged)
            
        # run through depth metrics for all apps, splitting cells into gradations
        for app in shared_data:
            #print(shared_data[app])
            
            # Extract shorter app name from the title passed in by user
            appname = app[len('Benchmark Results - '):len(app)]
            appname = appname[:appname.index(' - ')]
            
            aq_metrics = shared_data[app]["aq_metrics"]
            #print(aq_metrics)
            
            if len(aq_metrics["groups"]) == 0:
                print(f"****** NO RESULTS for {appname} ****** ")
                continue

            # check if we have depth metrics
            do_depths = len(aq_metrics["tr_n2qs"]) > 0
            if not do_depths:
                continue
                
            w_data = aq_metrics["groups"]          
            n2q_tr_data = aq_metrics["tr_n2qs"]
            f_data = aq_metrics["fidelities"]
    
            #plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base,
                   #label=appname, labelpos=(0.4, 0.6), labelrot=50, type=1)  

            # aggregate value metrics for each depth cell over all apps
            for i in range(len(w_data)):
                x = depth_index(n2q_tr_data[i], depth_base)
                y = float(w_data[i])
                f = f_data[i]
                
                # accumulate largest width for all apps
                w_max = max(w_max, y)
                
                xp = x * 4
                
                if x > max_depth_log - 1:
                    print(f"... data out of chart range, skipped; w={y} d={n2q_tr_data[i]}")
                    continue;
                    
                for grad in range(num_grads):
                    e = depth_values_merged[int(w_data[i])][int(xp + grad)]
                    if e == None: 
                        e = { "value": 0.0, "count": 0 }
                    e["count"] += 1
                    e["value"] += f
                    depth_values_merged[int(w_data[i])][int(xp + grad)] = e
        
        # Now overlay depth metrics for each app with unfilled rects, to outline each circuit
        
        vplot_anno_init()
        
        for app in shared_data:
        
            # Extract shorter app name from the title passed in by user
            appname = app[len('Benchmark Results - '):len(app)]
            appname = appname[:appname.index(' - ')]
    
            aq_metrics = shared_data[app]["aq_metrics"]
            
            # check if we have depth metrics for group
            if len(aq_metrics["groups"]) == 0:
                continue
            if len(aq_metrics["tr_n2qs"]) == 0:
                continue
                
            w_data = aq_metrics["groups"]
            n2q_tr_data=aq_metrics['tr_n2qs']
            f_data = aq_metrics["fidelities"]            

            # plot data rectangles
            '''
            for i in range(len(d_data)):
                x = depth_index(d_tr_data[i], depth_base)
                y = float(w_data[i])
                f = f_data[i]
                ax.add_patch(box_at(x, y, f, type=1, fill=False))
            '''
            
            #print(f"... plotting {appname}")
                
            plot_volumetric_data_aq(ax, w_data, n2q_tr_data, f_data, depth_base,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=True, w_max=w_max)
        
        # do annotation separately, spreading labels for readability
        anno_volumetric_data(ax, depth_base,
                   label=appname, labelpos=(3.0, 1.5), labelrot=15, type=1, fill=False)
          
        #Final pass to overlay unfilled rects for each cell (this is incorrect, should be removed)
        #plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base, fill=False,
                   #label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)
    
    except Exception as e:
        print(f'ERROR: failure when creating volumetric positioning chart')
        print(f"... exception = {e}")
    
    # save plot image file
    if save_plot_images:
        save_plot_image(plt, imagename, backend_id)

    #display plot
    plt.show()

# Plot metrics over all groups (2), merging data from all apps into smaller cells
def plot_metrics_all_merged (shared_data, backend_id, suptitle=None, imagename="_ALL-vplot-2", avail_qubits=0):    
      
    global circuit_metrics
    global group_metrics
  
    # generate separate figure for volumetric positioning chart of depth metrics
    # found it difficult to share the x axis with first 3, but have diff axis for this one
    
    #print(f"... {max_depth_log}")
    
    #if True:
    try:
        #print(f"... {d_data} {d_tr_data}")
        
        # determine largest width for all apps
        w_max = 0
        for app in shared_data:
            group_metrics = shared_data[app]["group_metrics"]
            w_data = group_metrics["groups"]
            for i in range(len(w_data)):
                y = float(w_data[i])
                w_max = max(w_max, y)
        
        # allow one more in width to accommodate the merge values below
        max_qubits = int(w_max) + 1     
        #print(f"... {w_max} {max_qubits}")
        
        ax = plot_volumetric_background(max_qubits, QV, depth_base, suptitle=suptitle, avail_qubits=avail_qubits)
        
        # create 2D array to hold merged value arrays with gradations, one array for each qubit size
        num_grads = 4
        depth_values_merged = []
        for w in range(max_qubits):
            depth_values_merged.append([ None ] * (num_grads * max_depth_log))
        
        #print(depth_values_merged)
            
        # run through depth metrics for all apps, splitting cells into gradations
        for app in shared_data:
            #print(shared_data[app])
            
            # Extract shorter app name from the title passed in by user
            appname = app[len('Benchmark Results - '):len(app)]
            appname = appname[:appname.index(' - ')]
            
            group_metrics = shared_data[app]["group_metrics"]
            #print(group_metrics)
            
            if len(group_metrics["groups"]) == 0:
                print(f"****** NO RESULTS for {appname} ****** ")
                continue

            # check if we have depth metrics
            do_depths = len(group_metrics["avg_depths"]) > 0
            if not do_depths:
                continue
                
            w_data = group_metrics["groups"]
            d_data = group_metrics["avg_depths"]
            d_tr_data = group_metrics["avg_tr_depths"]
            f_data = group_metrics["avg_fidelities"]
    
            #plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base,
                   #label=appname, labelpos=(0.4, 0.6), labelrot=50, type=1)  

            # aggregate value metrics for each depth cell over all apps
            for i in range(len(d_data)):
                x = depth_index(d_tr_data[i], depth_base)
                y = float(w_data[i])
                f = f_data[i]
                
                # accumulate largest width for all apps
                w_max = max(w_max, y)
                
                xp = x * 4
                
                if x > max_depth_log - 1:
                    print(f"... data out of chart range, skipped; w={y} d={d_tr_data[i]}")
                    continue;
                    
                for grad in range(num_grads):
                    e = depth_values_merged[int(w_data[i])][int(xp + grad)]
                    if e == None: 
                        e = { "value": 0.0, "count": 0 }
                    e["count"] += 1
                    e["value"] += f
                    depth_values_merged[int(w_data[i])][int(xp + grad)] = e
                    
        #print(depth_values_merged)
        
        # compute and plot the average fidelity at each width / depth gradation with narrow filled rects 
        for wi in range(len(depth_values_merged)):
            w = depth_values_merged[wi]
            #print(f"... w = {w}")
            
            for di in range(len(w)):
            
                e = w[di]
                
                if e != None:
                    e["value"] /= e["count"]
                    e["count"] = 1
                
                    x = di / 4
                    
                    # move half cell to left, to account for num grads
                    x -= 0.25
                    
                    y = float(wi)
                    f = e["value"]
                    
                    ax.add_patch(box4_at(x, y, f, type=1, fill=True))
        
        #print("**** merged...")
        #print(depth_values_merged)
        
        # Now overlay depth metrics for each app with unfilled rects, to outline each circuit
        
        vplot_anno_init()
        
        for app in shared_data:
        
            # Extract shorter app name from the title passed in by user
            appname = app[len('Benchmark Results - '):len(app)]
            appname = appname[:appname.index(' - ')]
    
            group_metrics = shared_data[app]["group_metrics"]
            
            # check if we have depth metrics for group
            if len(group_metrics["groups"]) == 0:
                continue
            if len(group_metrics["avg_depths"]) == 0:
                continue
                
            w_data = group_metrics["groups"]
            d_data = group_metrics["avg_depths"]
            d_tr_data = group_metrics["avg_tr_depths"]
            f_data = group_metrics["avg_fidelities"]            

            # plot data rectangles
            '''
            for i in range(len(d_data)):
                x = depth_index(d_tr_data[i], depth_base)
                y = float(w_data[i])
                f = f_data[i]
                ax.add_patch(box_at(x, y, f, type=1, fill=False))
            '''
            
            #print(f"... plotting {appname}")
                
            plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False, w_max=w_max)
        
        # do annotation separately, spreading labels for readability
        anno_volumetric_data(ax, depth_base,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False)
          
        #Final pass to overlay unfilled rects for each cell (this is incorrect, should be removed)
        #plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base, fill=False,
                   #label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)
    
    except Exception as e:
        print(f'ERROR: failure when creating volumetric positioning chart')
        print(f"... exception = {e}")
    
    # save plot image file
    if save_plot_images:
        save_plot_image(plt, imagename, backend_id)

    #display plot
    plt.show()

# Plot metrics over all groups (2), merging data from all apps into smaller cells
def plot_metrics_all_merged_aq (shared_data, backend_id, suptitle=None, imagename="_ALL-vplot-2", avail_qubits=0):    
      
    global circuit_metrics
    global group_metrics
  
    # generate separate figure for volumetric positioning chart of depth metrics
    # found it difficult to share the x axis with first 3, but have diff axis for this one
    
    #print(f"... {max_depth_log}")
    
    #if True:
    try:
        #print(f"... {d_data} {d_tr_data}")
        
        # determine largest width for all apps
        w_max = 0
        for app in shared_data:
            group_metrics = shared_data[app]["group_metrics"]
            w_data = group_metrics["groups"]
            for i in range(len(w_data)):
                y = float(w_data[i])
                w_max = max(w_max, y)
        
        #determine width for AQ
        AQ=w_max
        for app in shared_data:
            group_metrics = shared_data[app]["group_metrics"]
            w_data = group_metrics["groups"]
            n2q_data = group_metrics["avg_tr_n2qs"]       
            fidelity_data=group_metrics["avg_fidelities"]
            while True:
                n2q_cutoff=AQ*AQ
                fail_w=[float(w_data[i]) for i,v in enumerate(n2q_data) if (float(v) < n2q_cutoff and float(fidelity_data[i])<aq_cutoff)] 
                if len(fail_w)==0:
                    break        
                AQ-=1
        
        # allow one more in width to accommodate the merge values below
        max_qubits = int(w_max) + 1     
        #print(f"... {w_max} {max_qubits}")
        
        ax = plot_volumetric_background_aq(max_qubits=max_qubits, AQ=AQ, depth_base=depth_base, suptitle=suptitle, avail_qubits=avail_qubits)
        
        # create 2D array to hold merged value arrays with gradations, one array for each qubit size
        num_grads = 4
        depth_values_merged = []
        for w in range(max_qubits):
            depth_values_merged.append([ None ] * (num_grads * max_depth_log))
        
        #print(depth_values_merged)
            
        # run through depth metrics for all apps, splitting cells into gradations
        for app in shared_data:
            #print(shared_data[app])
            
            # Extract shorter app name from the title passed in by user
            appname = app[len('Benchmark Results - '):len(app)]
            appname = appname[:appname.index(' - ')]
            
            group_metrics = shared_data[app]["group_metrics"]
            #print(group_metrics)
            
            if len(group_metrics["groups"]) == 0:
                print(f"****** NO RESULTS for {appname} ****** ")
                continue

            # check if we have depth metrics
            do_depths = len(group_metrics["avg_depths"]) > 0
            if not do_depths:
                continue
                
            w_data = group_metrics["groups"]
            d_data = group_metrics["avg_depths"]
            d_tr_data = group_metrics["avg_tr_depths"]            
            n2q_tr_data = group_metrics["avg_tr_n2qs"]
            f_data = group_metrics["avg_fidelities"]
    
            #plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base,
                   #label=appname, labelpos=(0.4, 0.6), labelrot=50, type=1)  

            # aggregate value metrics for each depth cell over all apps
            for i in range(len(d_data)):
                x = depth_index(n2q_tr_data[i], depth_base)
                y = float(w_data[i])
                f = f_data[i]
                
                # accumulate largest width for all apps
                w_max = max(w_max, y)
                
                xp = x * 4
                
                if x > max_depth_log - 1:
                    print(f"... data out of chart range, skipped; w={y} d={n2q_tr_data[i]}")
                    continue;
                    
                for grad in range(num_grads):
                    e = depth_values_merged[int(w_data[i])][int(xp + grad)]
                    if e == None: 
                        e = { "value": 0.0, "count": 0 }
                    e["count"] += 1
                    e["value"] += f
                    depth_values_merged[int(w_data[i])][int(xp + grad)] = e
                    
        #print(depth_values_merged)
        
        # compute and plot the average fidelity at each width / depth gradation with narrow filled rects 
        for wi in range(len(depth_values_merged)):
            w = depth_values_merged[wi]
            #print(f"... w = {w}")
            
            for di in range(len(w)):
            
                e = w[di]
                
                if e != None:
                    e["value"] /= e["count"]
                    e["count"] = 1
                
                    x = di / 4
                    
                    # move half cell to left, to account for num grads
                    x -= 0.25
                    
                    y = float(wi)
                    f = e["value"]
                    
                    ax.add_patch(box4_at(x, y, f, type=1, fill=True))
        
        #print("**** merged...")
        #print(depth_values_merged)
        
        # Now overlay depth metrics for each app with unfilled rects, to outline each circuit
        
        vplot_anno_init()
        
        for app in shared_data:
        
            # Extract shorter app name from the title passed in by user
            appname = app[len('Benchmark Results - '):len(app)]
            appname = appname[:appname.index(' - ')]
    
            group_metrics = shared_data[app]["group_metrics"]
            
            # check if we have depth metrics for group
            if len(group_metrics["groups"]) == 0:
                continue
            if len(group_metrics["avg_depths"]) == 0:
                continue
                
            w_data = group_metrics["groups"]
            d_data = group_metrics["avg_depths"]
            d_tr_data = group_metrics["avg_tr_depths"]
            n2q_tr_data=group_metrics['avg_tr_n2qs']
            f_data = group_metrics["avg_fidelities"]            

            # plot data rectangles
            '''
            for i in range(len(d_data)):
                x = depth_index(d_tr_data[i], depth_base)
                y = float(w_data[i])
                f = f_data[i]
                ax.add_patch(box_at(x, y, f, type=1, fill=False))
            '''
            
            #print(f"... plotting {appname}")
                
            plot_volumetric_data(ax, w_data, n2q_tr_data, f_data, depth_base,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False, w_max=w_max)
        
        # do annotation separately, spreading labels for readability
        anno_volumetric_data(ax, depth_base,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False)
          
        #Final pass to overlay unfilled rects for each cell (this is incorrect, should be removed)
        #plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base, fill=False,
                   #label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)
    
    except Exception as e:
        print(f'ERROR: failure when creating volumetric positioning chart')
        print(f"... exception = {e}")
    
    # save plot image file
    if save_plot_images:
        save_plot_image(plt, imagename, backend_id)

    #display plot
    plt.show()


### plot metrics across all apps for a backend_id

def plot_all_app_metrics(backend_id, do_all_plots=False,
        include_apps=None, exclude_apps=None, suffix="", avail_qubits=0):

    global circuit_metrics
    global group_metrics
    global cmap

    # load saved data from file
    api = "qiskit"
    shared_data = load_app_metrics(api, backend_id)
    
    # apply include / exclude lists
    if include_apps != None:
        new_shared_data = {}
        for app in shared_data:
        
            # Extract shorter app name from the title passed in by user
            appname = app[len('Benchmark Results - '):len(app)]
            appname = appname[:appname.index(' - ')]
            
            if appname in include_apps:
                new_shared_data[app] = shared_data[app]
                
        shared_data = new_shared_data
    
    if exclude_apps != None:
        new_shared_data = {}
        for app in shared_data:
        
            # Extract shorter app name from the title passed in by user
            appname = app[len('Benchmark Results - '):len(app)]
            appname = appname[:appname.index(' - ')]
            
            if appname not in exclude_apps:
                new_shared_data[app] = shared_data[app]
                
        shared_data = new_shared_data  
 
    #print(shared_data)
    
    # since the bar plots use the subtitle field, set it here
    circuit_metrics["subtitle"] = f"device = {backend_id}"
    
    timestr = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # show vplots if enabled
    if do_volumetric_plots:
    
        # this is an overlay plot, not very useful; better to merge
        '''
        cmap = cmap_spectral
        suptitle = f"Volumetric Positioning - All Applications (Combined)\nDevice={backend_id}  {timestr} UTC"
        plot_metrics_all_overlaid(shared_data, backend_id, suptitle=suptitle, imagename="_ALL-vplot-2")
        '''
        
        # draw the volumetric plots with two different colormaps, for comparison purposes
        
        #suptitle = f"Volumetric Positioning - All Applications (Merged)\nDevice={backend_id}  {timestr} UTC"
        #cmap = cmap_blues
        #plot_metrics_all_merged(shared_data, backend_id, suptitle=suptitle, imagename="_ALL-vplot-1"+suffix, avail_qubits=avail_qubits)
        
        cmap = cmap_spectral
        suptitle = f"Volumetric Positioning - All Applications (Merged)\nDevice={backend_id}  {timestr} UTC"
        
        plot_metrics_all_merged(shared_data, backend_id, suptitle=suptitle, imagename="_ALL-vplot-2"+suffix, avail_qubits=avail_qubits)
        
    # show all app metrics charts if enabled
    if do_app_charts_with_all_metrics or do_all_plots:
        for app in shared_data:
            #print("")
            #print(app)
            group_metrics = shared_data[app]["group_metrics"]
            plot_metrics(app)
 

def plot_all_app_metrics_aq(backend_id, do_all_plots=False,
        include_apps=None, exclude_apps=None, suffix="", avail_qubits=0, is_individual=False):

    global circuit_metrics
    global group_metrics
    global aq_metrics
    global cmap

    # load saved data from file
    api = "qiskit"
    shared_data = load_app_metrics(api, backend_id)
    
    # apply include / exclude lists
    if include_apps != None:
        new_shared_data = {}
        for app in shared_data:
        
            # Extract shorter app name from the title passed in by user
            appname = app[len('Benchmark Results - '):len(app)]
            appname = appname[:appname.index(' - ')]
            
            if appname in include_apps:
                new_shared_data[app] = shared_data[app]
                
        shared_data = new_shared_data
    
    if exclude_apps != None:
        new_shared_data = {}
        for app in shared_data:
        
            # Extract shorter app name from the title passed in by user
            appname = app[len('Benchmark Results - '):len(app)]
            appname = appname[:appname.index(' - ')]
            
            if appname not in exclude_apps:
                new_shared_data[app] = shared_data[app]
                
        shared_data = new_shared_data  
 
    #print(shared_data)
    
    # since the bar plots use the subtitle field, set it here
    circuit_metrics["subtitle"] = f"device = {backend_id}"
    
    # timestr = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    timestr = strftime("%b %d, %Y", gmtime())

    # show vplots if enabled
    if do_volumetric_plots:
    
        # this is an overlay plot, not very useful; better to merge
        '''
        cmap = cmap_spectral
        suptitle = f"Volumetric Positioning - All Applications (Combined)\nDevice={backend_id}  {timestr} UTC"
        plot_metrics_all_overlaid(shared_data, backend_id, suptitle=suptitle, imagename="_ALL-vplot-2")
        '''
        
        # draw the volumetric plots with two different colormaps, for comparison purposes
        
        #suptitle = f"Volumetric Positioning - All Applications (Merged)\nDevice={backend_id}  {timestr} UTC"
        #cmap = cmap_blues
        #plot_metrics_all_merged(shared_data, backend_id, suptitle=suptitle, imagename="_ALL-vplot-1"+suffix, avail_qubits=avail_qubits)
        
        cmap = cmap_spectral
        suptitle = f"Volumetric Positioning - All Applications (Merged)\nDevice={backend_id}  {timestr} UTC"
        
        if is_individual is False:
            plot_metrics_all_merged_aq(shared_data, backend_id, suptitle=suptitle, imagename="_ALL-vplot-2"+suffix, avail_qubits=avail_qubits)
        else:
            plot_metrics_all_merged_individual_aq(shared_data, backend_id, suptitle=suptitle, imagename="_ALL-vplot-2"+suffix, avail_qubits=avail_qubits)
        
    # show all app metrics charts if enabled
    if do_app_charts_with_all_metrics or do_all_plots:
        for app in shared_data:
            #print("")
            #print(app)
            group_metrics = shared_data[app]["group_metrics"]
            plot_metrics(app)
 
### Plot Metrics for a specific application

def plot_metrics_for_app(backend_id, appname, apiname="Qiskit", filters=None, suffix=""):
    global circuit_metrics
    global group_metrics
    
    # load saved data from file
    api = "qiskit"
    shared_data = load_app_metrics(api, backend_id)
    
    # since the bar plots use the subtitle field, set it here
    circuit_metrics["subtitle"] = f"device = {backend_id}"
    
    timestr = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    
    app = "Benchmark Results - " + appname + " - " + apiname
    
    group_metrics = shared_data[app]["group_metrics"]
    plot_metrics(app, filters=filters, suffix=suffix)

 
##### Data File Methods      
     
# Save the application metrics data to a shared file for the current device
def store_app_metrics (backend_id, circuit_metrics, group_metrics, app, start_time=None, end_time=None):
    # print(f"... storing {title} {group_metrics}")
    
    # don't leave slashes in the filename
    backend_id = backend_id.replace("/", "_")
    
    # load the current data file of all apps
    api = "qiskit"
    shared_data = load_app_metrics(api, backend_id)
    
    # if there are no previous data for this app, init empty dict 
    if app not in shared_data:
        shared_data[app] = { "circuit_metrics":None, "group_metrics":None }
    
    shared_data[app]["backend_id"] = backend_id
    shared_data[app]["start_time"] = start_time
    shared_data[app]["end_time"] = end_time
    
    shared_data[app]["group_metrics"] = group_metrics
    
    aq_metrics={}
    aq_metrics["groups"]=[]
    aq_metrics["tr_n2qs"]=[]
    aq_metrics["fidelities"]=[]
    for group in circuit_metrics:
        if group=='subtitle':
            continue
        
        for key in circuit_metrics[group]:
            aq_metrics["groups"].append(group)
            aq_metrics["tr_n2qs"].append(circuit_metrics[group][key]["tr_n2q"])
            aq_metrics["fidelities"].append(circuit_metrics[group][key]["fidelity"])


    shared_data[app]["aq_metrics"] = aq_metrics
    
    # if saving raw circuit data, add it too
    #shared_data[app]["circuit_metrics"] =circuit_metrics
    
    # be sure we have a __data directory
    if not os.path.exists('__data'): os.makedirs('__data')
    
    # create filename based on the backend_id
    filename = f"__data/DATA-{backend_id}.json"
    
    # overwrite the existing file with the merged data
    with open(filename, 'w+') as f:
        json.dump(shared_data, f, indent=2, sort_keys=True)
        f.close()
 
# Load the application metrics from the given data file
# Returns a dict containing circuit and group metrics
def load_app_metrics (api, backend_id):

    # don't leave slashes in the filename
    backend_id = backend_id.replace("/", "_")

    filename = f"__data/DATA-{backend_id}.json"
    
    shared_data = None
    
    # attempt to load shared_data from file
    if os.path.exists(filename) and os.path.isfile(filename):
        with open(filename, 'r') as f:
            
            # attempt to load shared_data dict as json
            try:
                shared_data = json.load(f)
                
            except:
                pass
            
    # create empty shared_data dict if not read from file
    if shared_data == None:
        shared_data = {}
    
    # temporary: to read older format files ...     
    for app in shared_data:
        if "group_metrics" not in shared_data[app]:
            print(f"... upgrading version of app data {app}")
            shared_data[app] = { "circuit_metrics":None, "group_metrics":shared_data[app] }
 
    return shared_data
            
            
# save plot as image
def save_plot_image(plt, imagename, backend_id):

    # don't leave slashes in the filename
    backend_id = backend_id.replace("/", "_")
     
    # not used currently
    date_of_file = datetime.now().strftime("%d%m%Y_%H%M%S")
    
    if not os.path.exists('__images'): os.makedirs('__images')
    if not os.path.exists(f'__images/{backend_id}'): os.makedirs(f'__images/{backend_id}')
    
    pngfilename = f"{backend_id}/{imagename}"
    pngfilepath = os.path.join(os.getcwd(),"__images", pngfilename + ".jpg")
    
    plt.savefig(pngfilepath)
    
    #print(f"... saving (plot) image file:{pngfilename}.jpg")   
    
    pdffilepath = os.path.join(os.getcwd(),"__images", pngfilename + ".pdf")
    
    plt.savefig(pdffilepath)

## Uniform distribution function commonly used

def uniform_dist(num_state_qubits):
    dist = {}
    for i in range(2**num_state_qubits):
        key = bin(i)[2:].zfill(num_state_qubits)
        dist[key] = 1/(2**num_state_qubits)
    return dist                

### Analysis methods to be expanded and eventually compiled into a separate analysis.py file
import math, functools
import numpy as np

# Compute the fidelity based on Hellinger distance between two discrete probability distributions
def hellinger_fidelity_with_expected(p, q):
    """ p: result distribution, may be passed as a counts distribution
        q: the expected distribution to be compared against

    References:
        `Hellinger Distance @ wikipedia <https://en.wikipedia.org/wiki/Hellinger_distance>`_
        Qiskit Hellinger Fidelity Function
    """
    p_sum = sum(p.values())
    q_sum = sum(q.values())

    p_normed = {}
    for key, val in p.items():
        p_normed[key] = val/p_sum

    q_normed = {}
    for key, val in q.items():
        q_normed[key] = val/q_sum

    total = 0
    for key, val in p_normed.items():
        if key in q_normed.keys():
            total += (np.sqrt(val) - np.sqrt(q_normed[key]))**2
            del q_normed[key]
        else:
            total += val
    total += sum(q_normed.values())
    dist = np.sqrt(total)/np.sqrt(2)
    fidelity = (1-dist**2)**2

    return fidelity
    
def rescale_fidelity(fidelity, floor_fidelity, new_floor_fidelity):
    """
    Linearly rescales our fidelities to allow comparisons of fidelities across benchmarks
    
    fidelity: raw fidelity to rescale
    floor_fidelity: threshold fidelity which is equivalent to random guessing
    new_floor_fidelity: what we rescale the floor_fidelity to 

    Ex, with floor_fidelity = 0.25, new_floor_fidelity = 0.0:
        1 -> 1;
        0.25 -> 0;
        0.5 -> 0.3333;
    """
    rescaled_fidelity = (1-new_floor_fidelity)/(1-floor_fidelity) * (fidelity - 1) + 1
    
    # ensure fidelity is within bounds (0, 1)
    if rescaled_fidelity < 0:
        rescaled_fidelity = 0.0
    if rescaled_fidelity > 1:
        rescaled_fidelity = 1.0
    
    return rescaled_fidelity

def polarization_fidelity(counts, correct_dist, thermal_dist=None):
    """
    Combines Hellinger fidelity and polarization rescaling into fidelity calculation
    used in every benchmark

    counts: the measurement outcomes after `num_shots` algorithm runs
    correct_dist: the distribution we expect to get for the algorithm running perfectly
    thermal_dist: optional distribution to pass in distribution from a uniform
                  superposition over all states. If `None`: generated as 
                  `uniform_dist` with the same qubits as in `counts`

    Polarization from: `https://arxiv.org/abs/2008.11294v1`
    """
    # calculate fidelity via hellinger fidelity between correct distribution and our measured expectation values
    fidelity = hellinger_fidelity_with_expected(counts, correct_dist)

    if thermal_dist == None:
        # get length of random key in counts to find how many qubits measured
        num_measured_qubits = len(list(counts.keys())[0])
        
        # generate thermal dist based on number of qubits
        thermal_dist = uniform_dist(num_measured_qubits)

    # set our fidelity rescaling value as the hellinger fidelity for a depolarized state
    floor_fidelity = hellinger_fidelity_with_expected(thermal_dist, correct_dist)

    # rescale fidelity result so uniform superposition (random guessing) returns fidelity
    # rescaled to 0 to provide a better measure of success of the algorithm (polarization)
    new_floor_fidelity = 0
    fidelity = rescale_fidelity(fidelity, floor_fidelity, new_floor_fidelity)

    return fidelity

##############################################
# VOLUMETRIC PLOT
  
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

import matplotlib.cm as cm

############### Helper functions

# get a color from selected colormap
cmap_spectral = plt.get_cmap('Spectral')
cmap_blues = plt.get_cmap('Blues')
cmap = cmap_spectral

def get_color(value):

    if cmap == cmap_spectral:
        value = 0.05 + value*0.9
    elif cmap == cmap_blues:
        value = 0.05 + value*0.8
        
    return cmap(value)
    
    
# return the base index for a circuit depth value
# take the log in the depth base, and add 1
def depth_index(d, depth_base):
    if d==0:
        return 0
    else:
        return math.log(d, depth_base) + 1


# draw a box at x,y with various attributes   
def box_at(x, y, value, type=1, fill=True):
    size = 1.0
    
    value = min(value, 1.0)
    value = max(value, 0.0)

    fc = get_color(value)
    ec = (0.5,0.5,0.5)
    
    return Rectangle((x - size/2, y - size/2), size, size,
             edgecolor = ec,
             facecolor = fc,
             fill=fill,
             lw=0.5)

def circle_at(x, y, value, type=1, fill=True):
    size = 1.0
    
    value = min(value, 1.0)
    value = max(value, 0.0)

    fc = get_color(value)
    ec = (0.5,0.5,0.5)
    
    # return Rectangle((x - size/2, y - size/2), size, size,
    #          edgecolor = ec,
    #          facecolor = fc,
    #          fill=fill,
    #          lw=0.5)
    # print(x,y)
    return Circle((x, y), size/2,
             alpha = 0.5,
             edgecolor = ec,
             facecolor = fc,
             fill=fill,
             lw=0.5)
             
def box4_at(x, y, value, type=1, fill=True):
    size = 1.0
    
    value = min(value, 1.0)
    value = max(value, 0.0)

    fc = get_color(value)
    ec = (0.3,0.3,0.3)
    ec = fc
    
    return Rectangle((x - size/8, y - size/2), size/4, size,
             edgecolor = ec,
             facecolor = fc,
             fill=fill,
             lw=0.1)

def bkg_box_at(x, y, value):
    size = 0.6
    return Rectangle((x - size/2, y - size/2), size, size,
             edgecolor = (.75,.75,.75),
             facecolor = (.9,.9,.9),
             fill=True,
             lw=0.5)
             
def bkg_empty_box_at(x, y, value):
    size = 0.6
    return Rectangle((x - size/2, y - size/2), size, size,
             edgecolor = (.75,.75,.75),
             facecolor = (1.0,1.0,1.0),
             fill=True,
             lw=0.5)

# Draw a Quantum Volume rectangle with specified width and depth, and grey-scale value 
def qv_box_at(x, y, qv_width, qv_depth, value, depth_base):
    #print(f"{qv_width} {qv_depth} {depth_index(qv_depth, depth_base)}")
    return Rectangle((x - 0.5, y - 0.5), depth_index(qv_depth, depth_base), qv_width,
             edgecolor = (value,value,value),
             facecolor = (value,value,value),
             fill=True,
             lw=1)

# format a number using K,M,B,T for large numbers
# (sign handling may be incorrect)
def format_number(num):
    if isinstance(num, str): num = float(num)
    num = float('{:.3g}'.format(abs(num)))
    sign = ''
    metric = {'T': 1000000000000, 'B': 1000000000, 'M': 1000000, 'K': 1000, '': 1}
    for index in metric:
        num_check = num / metric[index]
        if num_check >= 1:
            num = round(num_check)
            sign = index
            break
    numstr = f"{str(num)}"
    if '.' in numstr:
        numstr = numstr.rstrip('0').rstrip('.')
    return f"{numstr}{sign}"

##### Volumetric Plots

# Plot the background for the volumetric analysis    
def plot_volumetric_background(max_qubits=11, QV=32, depth_base=2, suptitle=None, avail_qubits=0):
    
    if suptitle == None:
        suptitle = f"Volumetric Positioning\nCircuit Dimensions and Fidelity Overlaid on Quantum Volume = {QV}"

    QV0 = QV
    qv_estimate = False
    est_str = ""
    if QV == 0:                 # QV = 0 indicates "do not draw QV background or label"
        QV = 8192
        
    elif QV < 0:                # QV < 0 indicates "add est. to label"
        QV = -QV
        qv_estimate = True
        est_str = " (est.)"
        
    max_width = 13
    if max_qubits > 11: max_width = 18
    if max_qubits > 14: max_width = 20
    if max_qubits > 16: max_width = 24
    #print(f"... {avail_qubits} {max_qubits} {max_width}")
    
    plot_width = 6.8
    plot_height = 0.5 + plot_width * (max_width / max_depth_log)
    #print(f"... {plot_width} {plot_height}")
    
    # define matplotlib figure and axis; use constrained layout to fit colorbar to right
    fig, ax = plt.subplots(figsize=(plot_width, plot_height), constrained_layout=True)

    plt.suptitle(suptitle)

    plt.xlim(0, max_depth_log)
    plt.ylim(0, max_width)

    # circuit depth axis (x axis)
    xbasis = [x for x in range(1,max_depth_log)]
    xround = [depth_base**(x-1) for x in xbasis]
    xlabels = [format_number(x) for x in xround]
    ax.set_xlabel('Circuit Depth')
    ax.set_xticks(xbasis)  
    plt.xticks(xbasis, xlabels, color='black', rotation=45, ha='right', va='top', rotation_mode="anchor")
    
    # other label options
    #plt.xticks(xbasis, xlabels, color='black', rotation=-60, ha='left')
    #plt.xticks(xbasis, xlabels, color='black', rotation=-45, ha='left', va='center', rotation_mode="anchor")

    # circuit width axis (y axis)
    ybasis = [y for y in range(1,max_width)]
    yround = [1,2,3,4,5,6,7,8,10,12,15]     # not used now
    ylabels = [str(y) for y in yround]      # not used now 
    #ax.set_ylabel('Circuit Width (Number of Qubits)')
    ax.set_ylabel('Circuit Width')
    ax.set_yticks(ybasis)

    #create simple line plot (not used right now)
    #ax.plot([0, 10],[0, 10])
    
    log2QV = math.log2(QV)
    QV_width = log2QV
    QV_depth = log2QV * QV_transpile_factor
    
    # show a quantum volume rectangle of QV = 64 e.g. (6 x 6)
    if QV0 != 0:
        ax.add_patch(qv_box_at(1, 1, QV_width, QV_depth, 0.87, depth_base))
    
    # the untranspiled version is commented out - we do not show this by default
    # also show a quantum volume rectangle un-transpiled
    # ax.add_patch(qv_box_at(1, 1, QV_width, QV_width, 0.80, depth_base))

    # show 2D array of volumetric cells based on this QV_transpiled
    # DEVNOTE: we use +1 only to make the visuals work; s/b without
    # Also, the second arg of the min( below seems incorrect, needs correction
    maxprod = (QV_width + 1) * (QV_depth + 1)
    for w in range(1, min(max_width, round(QV) + 1)):
        
        # don't show VB squares if width greater than known available qubits
        if avail_qubits != 0 and w > avail_qubits:
            continue
        
        i_success = 0
        for d in xround:
        
            # polarization factor for low circuit widths
            maxtest = maxprod / ( 1 - 1 / (2**w) )
            
            # if circuit would fail here, don't draw box
            if d > maxtest: continue
            if w * d > maxtest: continue
            
            # guess for how to capture how hardware decays with width, not entirely correct

            # # reduce maxtext by a factor of number of qubits > QV_width
            # # just an approximation to account for qubit distances
            # if w > QV_width:
            #     over = w - QV_width 
            #     maxtest = maxtest / (1 + (over/QV_width))

            # draw a box at this width and depth
            id = depth_index(d, depth_base) 
            
            # show vb rectangles; if not showing QV, make all hollow
            if QV0 == 0:
                ax.add_patch(bkg_empty_box_at(id, w, 0.5))
            else:
                ax.add_patch(bkg_box_at(id, w, 0.5))
            
            # save index of last successful depth
            i_success += 1
        
        # plot empty rectangle after others       
        d = xround[i_success]
        id = depth_index(d, depth_base) 
        ax.add_patch(bkg_empty_box_at(id, w, 0.5))
        
    
    # Add annotation showing quantum volume
    if QV0 != 0:
        t = ax.text(max_depth_log - 2.0, 1.5, f"QV{est_str}={QV}", size=12,
                horizontalalignment='right', verticalalignment='center', color=(0.2,0.2,0.2),
                bbox=dict(boxstyle="square,pad=0.3", fc=(.9,.9,.9), ec="grey", lw=1))
                
    # add colorbar to right of plot
    plt.colorbar(cm.ScalarMappable(cmap=cmap), shrink=0.6, label="Avg Result Fidelity", panchor=(0.0, 0.7))
            
    return ax

x_annos = []
y_annos = []
x_anno_offs = []
y_anno_offs = []
anno_labels = []
    
def plot_volumetric_background_aq(max_qubits=11, AQ=22, depth_base=2, suptitle=None, avail_qubits=0):
    
    if suptitle == None:
        suptitle = f"Volumetric Positioning\nCircuit Dimensions and Fidelity Overlaid on Algorithmic Qubits = {AQ}"

    AQ0 = AQ
    aq_estimate = False
    est_str = ""

    if AQ == 0:
        AQ=20
        
    if AQ < 0:   
        AQ0 = 0             # AQ < 0 indicates "add est. to label"
        AQ = -AQ
        aq_estimate = True
        est_str = " (est.)"
        
    max_width = 13
    if max_qubits > 11: max_width = 18
    if max_qubits > 14: max_width = 20
    if max_qubits > 16: max_width = 24
    #print(f"... {avail_qubits} {max_qubits} {max_width}")
    
    seed = 6.8
    #plot_width = 0.5 + seed * (max_width / max_depth_log)
    plot_width = seed
    plot_height = 0.5 + seed * (max_width / max_depth_log)
    #print(f"... {plot_width} {plot_height}")
    
    # define matplotlib figure and axis; use constrained layout to fit colorbar to right
    fig, ax = plt.subplots(figsize=(plot_width, plot_height), constrained_layout=True)

    plt.suptitle(suptitle)

    plt.xlim(0, max_depth_log)
    plt.ylim(0, max_width)

    # circuit depth axis (x axis)
    xbasis = [x for x in range(1,max_depth_log)]
    xround = [depth_base**(x-1) for x in xbasis]
    xlabels = [format_number(x) for x in xround]
    ax.set_xlabel('Number of 2Q gates')
    ax.set_xticks(xbasis)  
    plt.xticks(xbasis, xlabels, color='black', rotation=45, ha='right', va='top', rotation_mode="anchor")
    
    # other label options
    #plt.xticks(xbasis, xlabels, color='black', rotation=-60, ha='left')
    #plt.xticks(xbasis, xlabels, color='black', rotation=-45, ha='left', va='center', rotation_mode="anchor")

    # circuit width axis (y axis)
    ybasis = [y for y in range(1,max_width)]
    yround = [1,2,3,4,5,6,7,8,10,12,15]     # not used now
    ylabels = [str(y) for y in yround]      # not used now 
    #ax.set_ylabel('Circuit Width (Number of Qubits)')
    ax.set_ylabel('Circuit Width')
    ax.set_yticks(ybasis)

    #create simple line plot (not used right now)
    #ax.plot([0, 10],[0, 10])
    
    #log2AQsq = math.log2(AQ*AQ)
    AQ_width = AQ
    AQ_depth = AQ*AQ
    
    # show a quantum volume rectangle of AQ = 6 e.g. (6 x 36)
    if AQ0 != 0:
        ax.add_patch(qv_box_at(1, 1, AQ_width, AQ_depth, 0.87, depth_base))
    
    # the untranspiled version is commented out - we do not show this by default
    # also show a quantum volume rectangle un-transpiled
    # ax.add_patch(qv_box_at(1, 1, QV_width, QV_width, 0.80, depth_base))

    # show 2D array of volumetric cells based on this QV_transpiled
    # DEVNOTE: we use +1 only to make the visuals work; s/b without
    # Also, the second arg of the min( below seems incorrect, needs correction
    maxprod = AQ_depth
    for w in range(1, max_width):
        
        # don't show VB squares if width greater than known available qubits
        if avail_qubits != 0 and w > avail_qubits:
            continue
        
        i_success = 0
        for d in xround:
                    
            # if circuit would fail here, don't draw box
            if d > maxprod: continue
            if (w-1) > maxprod: continue
            
            # guess for how to capture how hardware decays with width, not entirely correct

            # # reduce maxtext by a factor of number of qubits > QV_width
            # # just an approximation to account for qubit distances
            # if w > QV_width:
            #     over = w - QV_width 
            #     maxtest = maxtest / (1 + (over/QV_width))

            # draw a box at this width and depth
            id = depth_index(d, depth_base) 
            
            # show vb rectangles; if not showing QV, make all hollow
            if AQ0 == 0:
                ax.add_patch(bkg_empty_box_at(id, w, 0.5))
            else:
                ax.add_patch(bkg_box_at(id, w, 0.5))
            
            # save index of last successful depth
            i_success += 1
        
        # plot empty rectangle after others       
        d = xround[i_success]
        id = depth_index(d, depth_base) 
        ax.add_patch(bkg_empty_box_at(id, w, 0.5))
        
    
    # Add annotation showing quantum volume
    if AQ0 != 0:
        t = ax.text(max_depth_log - 2.0, 1.5, f"AQ{est_str}={AQ}", size=12,
                horizontalalignment='right', verticalalignment='center', color=(0.2,0.2,0.2),
                bbox=dict(boxstyle="square,pad=0.3", fc=(.9,.9,.9), ec="grey", lw=1))
                
    # add colorbar to right of plot
    plt.colorbar(cm.ScalarMappable(cmap=cmap), shrink=0.6, label="Avg Result Fidelity", panchor=(0.0, 0.7))
            
    return ax

x_annos = []
y_annos = []
x_anno_offs = []
y_anno_offs = []
anno_labels = []

# init arrays to hold annotation points for label spreading
def vplot_anno_init ():

    global x_annos, y_annos, x_anno_offs, y_anno_offs, anno_labels
    
    x_annos = []
    y_annos = []
    x_anno_offs = []
    y_anno_offs = []
    anno_labels = []
    

# Plot one group of data for volumetric presentation    
def plot_volumetric_data(ax, w_data, d_data, f_data, depth_base=2, label='Depth',
        labelpos=(0.2, 0.7), labelrot=0, type=1, fill=True, w_max=18, do_label=False):

    # since data may come back out of order, save point at max y for annotation
    i_anno = 0
    x_anno = 0 
    y_anno = 0
    
    # plot data rectangles
    for i in range(len(d_data)):
        x = depth_index(d_data[i], depth_base)
        y = float(w_data[i])
        f = f_data[i]
        ax.add_patch(box_at(x, y, f, type=type, fill=fill))

        if y >= y_anno:
            x_anno = x
            y_anno = y
            i_anno = i
            
    x_annos.append(x_anno)
    y_annos.append(y_anno)
    
    anno_dist = math.sqrt( (y_anno - 1)**2 + (x_anno - 1)**2 )
    
    # adjust radius of annotation circle based on maximum width of apps
    anno_max = 10
    if w_max > 10:
        anno_max = 14
    if w_max > 14:
        anno_max = 18
        
    scale = anno_max / anno_dist

    # offset of text from end of arrow
    if scale > 1:
        x_anno_off = scale * x_anno - x_anno - 0.5
        y_anno_off = scale * y_anno - y_anno
    else:
        x_anno_off = 0.7
        y_anno_off = 0.5
        
    x_anno_off += x_anno
    y_anno_off += y_anno
    
    # print(f"... {xx} {yy} {anno_dist}")
    x_anno_offs.append(x_anno_off)
    y_anno_offs.append(y_anno_off)
    
    anno_labels.append(label)
    
    if do_label:
        ax.annotate(label, xy=(x_anno+labelpos[0], y_anno+labelpos[1]), rotation=labelrot,
            horizontalalignment='left', verticalalignment='bottom', color=(0.2,0.2,0.2))


def plot_volumetric_data_aq(ax, w_data, d_data, f_data, depth_base=2, label='Depth',
        labelpos=(0.2, 0.7), labelrot=0, type=1, fill=True, w_max=18, do_label=False):

    # since data may come back out of order, save point at max y for annotation
    i_anno = 0
    x_anno = 0 
    y_anno = 0
    
    # plot data rectangles
    for i in range(len(d_data)):
        x = depth_index(d_data[i], depth_base)
        y = float(w_data[i])
        f = f_data[i]
        ax.add_patch(circle_at(x, y, f, type=type, fill=fill))

        if y >= y_anno:
            x_anno = x
            y_anno = y
            i_anno = i
            
    x_annos.append(x_anno)
    y_annos.append(y_anno)
    
    anno_dist = math.sqrt( (y_anno - 1)**2 + (x_anno - 1)**2 )
    
    # adjust radius of annotation circle based on maximum width of apps
    anno_max = 10
    if w_max > 10:
        anno_max = 14
    if w_max > 14:
        anno_max = 18
        
    scale = anno_max / anno_dist

    # offset of text from end of arrow
    if scale > 1:
        x_anno_off = scale * x_anno - x_anno - 0.5
        y_anno_off = scale * y_anno - y_anno
    else:
        x_anno_off = 0.7
        y_anno_off = 0.5
        
    x_anno_off += x_anno
    y_anno_off += y_anno
    
    # print(f"... {xx} {yy} {anno_dist}")
    x_anno_offs.append(x_anno_off)
    y_anno_offs.append(y_anno_off)
    
    anno_labels.append(label)
    
    if do_label:
        ax.annotate(label, xy=(x_anno+labelpos[0], y_anno+labelpos[1]), rotation=labelrot,
            horizontalalignment='left', verticalalignment='bottom', color=(0.2,0.2,0.2))



# Arrange the stored annotations optimally and add to plot 
def anno_volumetric_data(ax, depth_base=2, label='Depth',
        labelpos=(0.2, 0.7), labelrot=0, type=1, fill=True):
    
    # sort all arrays by the x point of the text (anno_offs)
    global x_anno_offs, y_anno_offs, anno_labels, x_annos, y_annos
    all_annos = sorted(zip(x_anno_offs, y_anno_offs, anno_labels, x_annos, y_annos))
    x_anno_offs = [a for a,b,c,d,e in all_annos]
    y_anno_offs = [b for a,b,c,d,e in all_annos]
    anno_labels = [c for a,b,c,d,e in all_annos]
    x_annos = [d for a,b,c,d,e in all_annos]
    y_annos = [e for a,b,c,d,e in all_annos]
    
    #print(f"{x_anno_offs}")
    #print(f"{y_anno_offs}")
    #print(f"{anno_labels}")
    
    for i in range(len(anno_labels)):
        x_anno = x_annos[i]
        y_anno = y_annos[i]
        x_anno_off = x_anno_offs[i]
        y_anno_off = y_anno_offs[i]
        label = anno_labels[i]
        
        if i > 0:
            x_delta = abs(x_anno_off - x_anno_offs[i - 1])
            y_delta = abs(y_anno_off - y_anno_offs[i - 1])
            
            if y_delta < 0.7 and x_delta < 2:
                y_anno_off = y_anno_offs[i] = y_anno_offs[i - 1] - 0.6
                #x_anno_off = x_anno_offs[i] = x_anno_offs[i - 1] + 0.1
                    
        ax.annotate(label,
            xy=(x_anno+0.0, y_anno+0.1),
            arrowprops=dict(facecolor='black', shrink=0.0,
                width=0.5, headwidth=4, headlength=5, edgecolor=(0.8,0.8,0.8)),
            xytext=(x_anno_off + labelpos[0], y_anno_off + labelpos[1]),
            rotation=labelrot,
            horizontalalignment='left', verticalalignment='baseline',
            color=(0.2,0.2,0.2),
            clip_on=True)
 
 
####################################
# TEST METHODS 
        
# Test metrics module, with simple test data
def test_metrics ():
    init_metrics()
    
    store_metric('group1', 'circuit1', 'create_time', 123)
    store_metric('group1', 'circuit2', 'create_time', 234)
    store_metric('group2', 'circuit1', 'create_time', 156)
    store_metric('group2', 'circuit2', 'create_time', 278)
    
    store_metric('group1', 'circuit1', 'exec_time', 223)
    store_metric('group1', 'circuit2', 'exec_time', 334)
    store_metric('group2', 'circuit1', 'exec_time', 256)
    store_metric('group2', 'circuit2', 'exec_time', 378)
    
    store_metric('group1', 'circuit1', 'fidelity', 1.0)
    store_metric('group1', 'circuit2', 'fidelity', 0.8)
    store_metric('group2', 'circuit1', 'fidelity', 0.9)
    store_metric('group2', 'circuit2', 'fidelity', 0.7)
      
    aggregate_metrics()
    
    report_metrics()
    #report_metrics_for_group("badgroup")
    
    plot_metrics()

#test_metrics()
