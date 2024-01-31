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
import traceback
import matplotlib.cm as cm
import copy

# Raw and aggregate circuit metrics
circuit_metrics = {  }

circuit_metrics_detail = {  }    # for iterative algorithms
circuit_metrics_detail_2 = {  }  # used to break down to 3rd dimension
circuit_metrics_final_iter = {  } # used to store final results for the last circuit in iterative algorithms.

group_metrics = { "groups": [],
    "avg_depths": [], "avg_xis": [], "avg_tr_depths": [], "avg_tr_xis": [], "avg_tr_n2qs": [],
    "avg_create_times": [], "avg_elapsed_times": [], "avg_exec_times": [],
    "avg_fidelities": [], "avg_hf_fidelities": [],
    "std_create_times": [], "std_elapsed_times": [], "std_exec_times": [],
    "std_fidelities": [], "std_hf_fidelities": [],    
    "avg_exec_creating_times": [], "avg_exec_validating_times": [], "avg_exec_running_times": [],
    "job_ids": []
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

# Suffix to append to filename of DATA- files
data_suffix = ""

# Option to save plot images (all of them)
save_plot_images = True

# Option to show plot images. Useful if it is desired to not show plots while running scripts
show_plot_images = True

# Option to show elapsed times in the metrics plots
show_elapsed_times = True

# When ratio of max time to min time exceeds this use a logscale
logscale_for_times_threshold = 50

# Toss out elapsed times for any run if the initial value is this factor of the second value 
# (applies only to area plots - remove once queue time is removed earlier)
omit_initial_elapsed_time_factor = 0

# if tossing large elapsed times, assume elapsed is this multiple of exec time
initial_elapsed_time_multiplier = 1.1

# remove creating time from elapsed time when displaying (default)
# this seems to remove queue time in some cases (IBM machines only)
remove_creating_time_from_elapsed = True

# For depth plots, show algorithmic and normalized plots on separate axes
use_two_depth_axes = False

# Option to generate volumetric positioning charts
do_volumetric_plots = True

# Option to include all app charts with vplots at end
do_app_charts_with_all_metrics = False

# Number of ticks on volumetric depth axis
max_depth_log = 22

# Quantum Volume to display on volumetric background (default = 0)
QV = 0

# Algorithmic Qubits (defaults)
AQ = 12
aq_cutoff = 0.368   # below this circuits not considered successful

aq_mode = 0         # 0 - use default plot behavior, 1 - use AQ modified plots

# average transpile factor between base QV depth and our depth based on results from QV notebook
QV_transpile_factor = 12.7     

# Base for volumetric plot logarithmic axes
#depth_base = 1.66  # this stretches depth axis out, but other values have issues:
#1) need to round to avoid duplicates, and 2) trailing zeros are getting removed 
depth_base = 2

# suppress plotting for low fidelity at this level
suppress_low_fidelity_level = 0.015

# Get the current time formatted
def get_timestr():
    #timestr = strftime("%Y-%m-%d %H:%M:%S UTC", gmtime())
    timestr = strftime("%b %d, %Y %H:%M:%S UTC", gmtime())
    return timestr


######################################################################

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


##################################################
# DATA ANALYSIS - METRICS COLLECTION AND REPORTING
      
# Initialize the metrics module, creating an empty table of metrics
def init_metrics ():
    global start_time
    
    # create empty dictionary for circuit metrics
    circuit_metrics.clear()
    circuit_metrics_detail.clear()
    circuit_metrics_detail_2.clear()
    circuit_metrics_final_iter.clear()
    
    # create empty arrays for group metrics
    group_metrics["groups"] = []

    group_metrics["avg_depths"] = []
    group_metrics["avg_xis"] = []
    group_metrics["avg_tr_depths"] = []
    group_metrics["avg_tr_xis"] = []
    group_metrics["avg_tr_n2qs"] = []
    
    group_metrics["avg_create_times"] = []
    group_metrics["avg_elapsed_times"] = []
    group_metrics["avg_exec_times"] = []
    group_metrics["avg_fidelities"] = []
    group_metrics["avg_hf_fidelities"] = []
    
    group_metrics["std_create_times"] = []
    group_metrics["std_elapsed_times"] = []
    group_metrics["std_exec_times"] = []
    group_metrics["std_fidelities"] = []
    group_metrics["std_hf_fidelities"] = []
    
    group_metrics["avg_exec_creating_times"] = []
    group_metrics["avg_exec_validating_times"] = []
    group_metrics["avg_exec_running_times"] = []
    
    group_metrics["job_ids"] = []
    
    # store the start of execution for the current app
    start_time = time.time()
    print(f'... execution starting at {get_timestr()}')

# End metrics collection for an application
def end_metrics():
    global end_time

    end_time = time.time()
    total_run_time = round(end_time - start_time, 3)
    print(f'... execution complete at {get_timestr()} in {total_run_time} secs')
    print("")

##################################################
# METRICS STORE AND GET FUNCTIONS

# Store a single or multiple metric(s) associated with a group and circuit in the group
def store_metric (group, circuit, metric, value):
    group = str(group)
    circuit = str(circuit)
    
    # ensure that a table for this group and circuit exists
    if group not in circuit_metrics:
        circuit_metrics[group] = { }
    if circuit not in circuit_metrics[group]:
        circuit_metrics[group][circuit] = { }
        
    # if the value is a dict, store each metric provided
    if type(value) is dict:
        for key in value:
            # If you want to store multiple metrics in one go,
            # then simply provide these in the form of a dictionary under the value input
            # In this case, the metric input will be ignored
            store_metric(group, circuit, key, value[key]) 
    else:
        circuit_metrics[group][circuit][metric] = value
    #print(f'{group} {circuit} {metric} -> {value}') 
    
# method to pop the all metrics associated with a group and circuit in the group
def pop_metric (group, circuit):
    group = str(group)
    circuit = str(circuit)
    
    # ensure that a table for this group and circuit exists
    if group in circuit_metrics:
        if circuit in circuit_metrics[group]:
            pop_metric_dict = circuit_metrics[group].pop(circuit)
    
            return pop_metric_dict
    
# Store "final iteration" metric(s) associated with a group and circuit in the group
def store_props_final_iter(group, circuit, metric, value):
    group = str(group)
    circuit = str(circuit)
    
    # ensure that a table for this group and circuit exists
    if group not in circuit_metrics_final_iter:
        circuit_metrics_final_iter[group] = {}
    if circuit not in circuit_metrics_final_iter[group]:
        circuit_metrics_final_iter[group][circuit] = { }
        
    # store value or values to the final iteration tables
    if type(value) is dict:
        for key in value:
            store_props_final_iter(group, circuit, key, value[key])
    else:
        circuit_metrics_final_iter[group][circuit][metric] = value

# Return the value for a single or multiple metric(s) in the given a group and circuit
def get_metric (group, circuit, metric):
    group = str(group)
    circuit = str(circuit)

    # if the metric is a dict, return an array of metric values
    if type(metric) is dict:
        values = []
        for key in value:
            values.append(get_metric(group, circuit, key)) 
        return values

    # otherwise return single value
    if metric in circuit_metrics[group][circuit]:
        return circuit_metrics[group][circuit][metric]
    else:
        return 0    # DEVNOTE: might want to raise exception?


##################################################
# METRICS AGGREGATION FUNCTIONS
   
# Aggregate metrics for a specific group, creating average across circuits in group
def aggregate_metrics_for_group (group):
    group = str(group)
    
    # generate totals, then divide by number of circuits to calculate averages    
    if group in circuit_metrics:

        # job ids handled specially, maintain array in the aggregate
        group_job_ids = []

        # loop over circuits in group to generate totals
        for circuit in circuit_metrics[group]:
            for metric in circuit_metrics[group][circuit]:
                value = circuit_metrics[group][circuit][metric]
                #print(f'{group} {circuit} {metric} -> {value}')
                if metric == "job_id": group_job_ids.append(value)
 
        # store averages in arrays keyed by group and structured for reporting and plotting
        group_metrics["groups"].append(group)
        
        # store an array of job ids to permit access later to stored job data
        group_metrics["job_ids"].append(group_job_ids)
        
        # aggregate depth metrics
        # skip these if there is not a real circuit for this group
        # DEVNOTE: this is a klunky way to flag plot behavior; improve later
        avg, std = get_circuit_stats_for_metric(group, "depth", 0)
        if avg > 0:
            group_metrics["avg_depths"].append(avg)
        avg, std = get_circuit_stats_for_metric(group, "tr_depth", 0)
        if avg > 0:
            group_metrics["avg_tr_depths"].append(avg)
        
        # aggregate depth derivatives
        avg, std = get_circuit_stats_for_metric(group, "xi", 3)
        group_metrics["avg_xis"].append(avg)
        avg, std = get_circuit_stats_for_metric(group, "tr_xi", 3)
        group_metrics["avg_tr_xis"].append(avg)
        avg, std = get_circuit_stats_for_metric(group, "tr_n2q", 3)
        group_metrics["avg_tr_n2qs"].append(avg)
        
        # aggregate time metrics
        avg, std = get_circuit_stats_for_metric(group, "create_time", 3)
        group_metrics["avg_create_times"].append(avg)
        group_metrics["std_create_times"].append(std)
        avg, std = get_circuit_stats_for_metric(group, "elapsed_time", 3)
        group_metrics["avg_elapsed_times"].append(avg)
        group_metrics["std_elapsed_times"].append(std)
        avg, std = get_circuit_stats_for_metric(group, "exec_time", 3)
        group_metrics["avg_exec_times"].append(avg)
        group_metrics["std_exec_times"].append(std)

        # aggregate fidelity metrics
        avg, std = get_circuit_stats_for_metric(group, "fidelity", 3)
        group_metrics["avg_fidelities"].append(avg)
        group_metrics["std_fidelities"].append(std)
        avg, std = get_circuit_stats_for_metric(group, "hf_fidelity", 3)
        group_metrics["avg_hf_fidelities"].append(avg)
        group_metrics["std_hf_fidelities"].append(std)
        
        # aggregate specal time metrics (not used everywhere)
        # skip if not collected at all
        avg, std = get_circuit_stats_for_metric(group, "exec_creating_time", 3)
        if avg > 0:
            group_metrics["avg_exec_creating_times"].append(avg)
        avg, std = get_circuit_stats_for_metric(group, "exec_validating_time", 3)
        if avg > 0:
            group_metrics["avg_exec_validating_times"].append(avg)
        avg, std = get_circuit_stats_for_metric(group, "exec_running_time", 3)
        if avg > 0:
            group_metrics["avg_exec_running_times"].append(avg)
 
        
# Compute average and stddev for a metric in a given circuit group
# DEVNOTE: this creates new array every time; could be more efficient if multiple metrics done at once
def get_circuit_stats_for_metric(group, metric, precision):
    metric_array = []
    for circuit in circuit_metrics[group]:
        if metric in circuit_metrics[group][circuit]:
            metric_array.append(circuit_metrics[group][circuit][metric])
        else:
            metric_array.append(None)
    metric_array = [x for x in metric_array if x is not None]
    if len(metric_array) == 0:
        return 0, 0
    avg = round(np.average(metric_array), precision)
    std = round(np.std(metric_array)/np.sqrt(len(metric_array)), precision)
    return avg, std
    
            
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
                    print(f"Average Circuit Algorithmic Depth, \u03BE (xi) for the {group} qubit group = {int(avg_depth)}, {avg_xi}")
            
            avg_tr_xi = 0
            if len(group_metrics["avg_tr_xis"]) > 0:
                avg_tr_xi = group_metrics["avg_tr_xis"][group_index]
                
            avg_tr_n2q = 0
            if len(group_metrics["avg_tr_n2qs"]) > 0:
                avg_tr_n2q = group_metrics["avg_tr_n2qs"][group_index]
            
            if len(group_metrics["avg_tr_depths"]) > 0:
                avg_tr_depth = group_metrics["avg_tr_depths"][group_index]
                if avg_tr_depth > 0:
                    print(f"Average Normalized Transpiled Depth, \u03BE (xi), 2q gates for the {group} qubit group = {int(avg_tr_depth)}, {avg_tr_xi}, {avg_tr_n2q}")
                    
            avg_create_time = group_metrics["avg_create_times"][group_index]
            avg_elapsed_time = group_metrics["avg_elapsed_times"][group_index]
            avg_exec_time = group_metrics["avg_exec_times"][group_index]
            print(f"Average Creation, Elapsed, Execution Time for the {group} qubit group = {avg_create_time}, {avg_elapsed_time}, {avg_exec_time} secs")
            
            # report these detailed times, but only if they have been collected (i.e., len of array > 0)
            # not all backedns generate these data elements
            if len(group_metrics["avg_exec_creating_times"]) > 0:
                if len(group_metrics["avg_exec_creating_times"]) > group_index:
                    avg_exec_creating_time = group_metrics["avg_exec_creating_times"][group_index]
                else:
                    avg_exec_creating_time = 0

                if len(group_metrics["avg_exec_validating_times"]) > 0:
                    if len(group_metrics["avg_exec_validating_times"]) > group_index:
                        avg_exec_validating_time = group_metrics["avg_exec_validating_times"][group_index]
                    else:
                        avg_exec_validating_time = 0
                        
                if len(group_metrics["avg_exec_running_times"]) > 0:
                    if len(group_metrics["avg_exec_running_times"]) > group_index:
                        avg_exec_running_time = group_metrics["avg_exec_running_times"][group_index]
                    else:
                        avg_exec_running_time = 0
                            
                print(f"Average Transpiling, Validating, Running Times for group {group} = {avg_exec_creating_time}, {avg_exec_validating_time}, {avg_exec_running_time} secs")
            
            avg_fidelity = group_metrics["avg_fidelities"][group_index]
            avg_hf_fidelity = group_metrics["avg_hf_fidelities"][group_index]
            print(f"Average Hellinger, Normalized Fidelity for the {group} qubit group = {avg_hf_fidelity}, {avg_fidelity}")
            
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
def finalize_group(group, report=True):
    group = str(group)
    
    #print(f"... finalize group={group}")

    # loop over circuits in group to generate totals
    group_done = True
    for circuit in circuit_metrics[group]:
        #print(f"  ... metrics = {group} {circuit} {circuit_metrics[group][circuit]}")
        
        if "elapsed_time" not in circuit_metrics[group][circuit]:
            group_done = False
            break
    
    #print(f"  ... group_done = {group} {group_done}")
    if group_done and report:
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


######################################################
# DATA ANALYSIS - LEVEL 2 METRICS - ITERATIVE CIRCUITS

# Aggregate and report on metrics for the given groups, if all circuits in group are complete (2 levels)
def finalize_group_2_level(group):

    #print(f"... finalize group={group} 2-level")

    # loop over circuits in group to generate totals
    group_done = True
    for circuit in circuit_metrics[group]:
        #print(f"  ... metrics = {group} {circuit} {circuit_metrics[group][circuit]}")
        
        if "elapsed_time" not in circuit_metrics[group][circuit]:
            group_done = False
            break
    
    #print(f"  ... group_done = {group} {group_done}")
    if group_done:
    
        # before aggregating, perform aggregtion at 3rd level to create a single entry 
        # for each circuit_id and a separate table of detail metrics
        process_circuit_metrics_2_level(group)
        
        aggregate_metrics_for_group(group)
        print("************")
        report_metrics_for_group(group)
        
    # sort the group metrics (sometimes they come back out of order)
    sort_group_metrics()
    
    
# Process the circuit metrics to aggregate to third level by splitting circuit_id 
# This is used when there is a third level of iteration.  The same circuit is executed multiple times
# and the metrics collected are indexed by as idx1 * 1000 and idx2
# Create a circuit_metrics_detail containing all these metrics, aggregate them to circuit_metrics
# Create a circuit_metrics_detail_2 that has the detail metrics aggregated to support plotting

def process_circuit_metrics_2_level(num_qubits):
    global circuit_metrics_detail
    global circuit_metrics_detail_2
    
    group = str(num_qubits)
    
    # print out what was received
    #jsonDataStr = json.dumps(circuit_metrics[group], indent=2).replace('\n', '\n  ')
    #print("  ==> circuit_metrics: %s" % jsonDataStr) 
    
    #print(f"... process_circuit_metrics_2_level({num_qubits})") 
    circuit_metrics_detail[group] = circuit_metrics[group]
    circuit_metrics[group] = { }
    
    circuit_metrics_detail_2[group] = { }
    
    avg_fidelity = 0
    total_elapsed_time = 0
    total_exec_time = 0
    
    # loop over all the collected metrics to split the index into idx1 and idx2
    count = 0
    for circuit_id in circuit_metrics_detail[group]:
        #if count == 0: print(f"...   circuit_id={circuit_id}")
                
        id = int(circuit_id)
        idx1 = id; idx2 = -1
        if id >= 1000:
            idx1 = int(id / 1000)
            idx2 = id % 1000
            
        #print(f"...   idx1, idx2={idx1} {idx2}") 
        
        # if we have metrics for this (outer) circuit_id, then we need to accumulate these metrics
        # as they are encountered; otherwise there will just be one added the first time.
        # The result is that the circuit_metrics_detail dict is created, but indexed by one index (idx1 * 1000 + idx2)
        if idx1 in circuit_metrics[group]:
            last_circuit_id = str(int(circuit_id) - 1)
            ''' note: do not accumulate here, it is done in the plotting code
            circuit_metrics_detail[group][circuit_id]["elapsed_time"] += circuit_metrics_detail[group][last_circuit_id]["elapsed_time"]
            circuit_metrics_detail[group][circuit_id]["exec_time"] += circuit_metrics_detail[group][last_circuit_id]["exec_time"]
            '''
       
        # if there are no circuit_metrics created yet for this idx1, start a detail_2 table 
        else:
            circuit_metrics_detail_2[group][idx1] = { }
          
        # copy each of the detail metrics to the detail_2 dict indexed by idx1 and idx2 as they are encountered
        circuit_metrics_detail_2[group][idx1][idx2] = circuit_metrics_detail[group][circuit_id]
        
        # copy each detail entry to circuit_metrics_new so last one ends up there,
        # storing the last entry as the primary entry for this circuit id
        circuit_metrics[group][idx1] = circuit_metrics_detail[group][circuit_id]  
        
        # at the end we have one entry in the circuit_metrics table for the group and primary circuit_id
        # circuit_metrics_detail_2 has all the detail metrics indexed by group, idx1 and idx2 (where idx1=circuit_id)
        # circuit_metrics_detail is not used, just an intermediate
        
        count += 1 


# The method below is used in one form of iteration benchmark (TDB whether to retain JN)

iterations_metrics = {}

# Separate out metrics for final vs. intermediate circuits

def process_iteration_metrics(group_id):
    global circuit_metrics
    global iterations_metrics
    g_id = str(group_id)
    iterations_metrics[g_id] = {}
    
    for iteration, data in circuit_metrics[g_id].items():            
        for key, value in data.items():
            if iteration == '1':
                iterations_metrics[g_id][key] = []
            
            iterations_metrics[g_id][key].append(value)
      
    del circuit_metrics[g_id]
    return iterations_metrics
 
 
# convenience functions to print all circuit metrics (for debugging)

def dump_json(msg, data):
    jsonDataStr = json.dumps(data, indent=2).replace('\n', '\n  ')    
    print(f"{msg}: {jsonDataStr}") 
    
def print_all_circuit_metrics():

    dump_json("  ==> all circuit_metrics", circuit_metrics)
    
    print("  ==> all detail 2 circuit_metrics:")
    for group in circuit_metrics_detail_2:
        for circuit_id in circuit_metrics_detail_2[group]:
            print(f"    group {group} circuit {circuit_id}")
            for it in circuit_metrics_detail_2[group][circuit_id]:
                mets = circuit_metrics_detail_2[group][circuit_id][it]
                elt = round(mets["elapsed_time"], 3)
                ext = round(mets["exec_time"], 3)
                fid = round(mets["fidelity"], 3) if "fidelity" in mets else -1
                opt_ext = round(mets["opt_exec_time"], 3) if "opt_exec_time" in mets else -1
                print(f"      iteration {it} = {elt} {ext} {fid} {opt_ext}")


############################################
# DATA ANALYSIS - FIDELITY CALCULATIONS

## Uniform distribution function commonly used

def uniform_dist(num_state_qubits):
    dist = {}
    for i in range(2**num_state_qubits):
        key = bin(i)[2:].zfill(num_state_qubits)
        dist[key] = 1/(2**num_state_qubits)
    return dist                

### Analysis methods to be expanded and eventually compiled into a separate analysis.py file
import math
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

    if q_sum == 0:
        print("ERROR: polarization_fidelity(), expected distribution is invalid, all counts equal to 0")
        return 0

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
    
    # in some situations (error mitigation) this can go negative, use abs value
    if total < 0:
        print("WARNING: using absolute value in fidelity calculation")
        total = abs(total)
        
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
                  
    returns both polarization fidelity and the hellinger fidelity

    Polarization from: `https://arxiv.org/abs/2008.11294v1`
    """
    
    # get length of random key in correct_dist to find how many qubits measured
    num_measured_qubits = len(list(correct_dist.keys())[0])
    
    # ensure that all keys in counts are zero padded to this length
    counts = {k.zfill(num_measured_qubits): v for k, v in counts.items()}
    
    # calculate hellinger fidelity between measured expectation values and correct distribution
    hf_fidelity = hellinger_fidelity_with_expected(counts, correct_dist)
    
    # to limit cpu and memory utilization, skip noise correction if more than 16 measured qubits
    if num_measured_qubits > 16:
        return { 'fidelity':hf_fidelity, 'hf_fidelity':hf_fidelity }

    # if not provided, generate thermal dist based on number of qubits
    if thermal_dist == None:
        thermal_dist = uniform_dist(num_measured_qubits)

    # set our fidelity rescaling value as the hellinger fidelity for a depolarized state
    floor_fidelity = hellinger_fidelity_with_expected(thermal_dist, correct_dist)

    # rescale fidelity result so uniform superposition (random guessing) returns fidelity
    # rescaled to 0 to provide a better measure of success of the algorithm (polarization)
    new_floor_fidelity = 0
    fidelity = rescale_fidelity(hf_fidelity, floor_fidelity, new_floor_fidelity)

    return { 'fidelity':fidelity, 'hf_fidelity':hf_fidelity }
    

###############################################
# METRICS UTILITY FUNCTIONS - FOR VISUALIZATION

# get the min and max width over all apps in shared_data
# suppress low fidelity cells if flag set
def get_min_max(shared_data, suppress_low_fidelity=False):
    w_max = 0
    w_min = 0
    for app in shared_data:
        group_metrics = shared_data[app]["group_metrics"]
        w_data = group_metrics["groups"]
        f_data = group_metrics["avg_fidelities"]
        
        low_fidelity_count = True
        for i in range(len(w_data)):
            y = float(w_data[i])
            
            # need this to handle rotated groups
            if i >= len(f_data):
                break
            
            # don't include in max width, the cells that reject for low fidelity
            f = f_data[i]
            if suppress_low_fidelity and f < suppress_low_fidelity_level:
                if low_fidelity_count: break
                else: low_fidelity_count = True
                    
            w_max = max(w_max, y)
            w_min = min(w_min, y) 
            
    return w_min, w_max


#determine width for AQ
def get_aq_width(shared_data, w_min, w_max, fidelity_metric):
    AQ=w_max
    for app in shared_data:
        group_metrics = shared_data[app]["group_metrics"]
        w_data = group_metrics["groups"]
         
        if "avg_tr_n2qs" not in group_metrics:
            continue
        if fidelity_metric not in group_metrics:
            continue
        
        n2q_data = group_metrics["avg_tr_n2qs"]            
        fidelity_data=group_metrics[fidelity_metric]
        
        while True:
            n2q_cutoff=AQ*AQ
            fail_w=[i for i in range(len(n2q_data)) if (float(n2q_data[i]) <= n2q_cutoff and float(w_data[i]) <=AQ and float(fidelity_data[i])<aq_cutoff)] 
            if len(fail_w)==0:
                break        
            AQ-=1
    
    if AQ<w_min:
        AQ=0
        
    return AQ

# Get the backend_id for current set of circuits
def get_backend_id(backend_id=None):
    if backend_id is None:
        subtitle = circuit_metrics["subtitle"]
        backend_id = subtitle[9:]
    return backend_id
    
# Get the label to be used in plots for the device, with the data_suffix concatenated
def get_backend_label(backend_id=None): 
    return get_backend_id(backend_id=backend_id) + data_suffix

# Get the title string showing the device name and current date_of_file
# DEVNOTE: we might want to change to the date contained in the data file (to show when data collected) 
def get_backend_title(backend_id=None):  
    return f"\nDevice={get_backend_label(backend_id=backend_id)}  {get_timestr()}"
 
# Extract short app name from the title passed in by user
def get_appname_from_title(suptitle):
    appname = suptitle[len('Benchmark Results - '):len(suptitle)]
    appname = appname[:appname.index(' - ')]
    
    # for creating plot image filenames replace spaces
    appname = appname.replace(' ', '-') 
    
    return appname

    
############################################
# ANALYSIS AND VISUALIZATION - METRICS PLOTS

import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))
maxcut_style = os.path.join(dir_path,'maxcut.mplstyle')
# plt.style.use(style_file)
    
# Plot bar charts for each metric over all groups
def plot_metrics (suptitle="Circuit Width (Number of Qubits)", transform_qubit_group = False, new_qubit_group = None, filters=None, suffix="", options=None):
    
    # get backend id for this set of circuits
    backend_id = get_backend_id()
    
    # Extract shorter app name from the title passed in by user   
    appname = get_appname_from_title(suptitle)
    
    # save the metrics for current application to the DATA file, one file per device
    if save_metrics:

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
        print("     ****** NO RESULTS ****** ")
        return
    
    # sort the group metrics (in case they weren't sorted when collected)
    sort_group_metrics()
    
    # flags for charts to show
    do_creates = True
    do_executes = True
    do_fidelities = True
    do_hf_fidelities = False
    do_depths = True
    do_2qs = False
    do_vbplot = True     
    
    # check if we have depth metrics to show
    do_depths = len(group_metrics["avg_depths"]) > 0

    # in AQ mode, show different metrics
    if aq_mode > 0:
        do_fidelities = True        # make this True so we can compare
        do_depths = False       
        do_hf_fidelities = True
        do_2qs = True 
        
    # if filters set, adjust these flags
    if filters != None:
        if "create" not in filters: do_creates = False
        if "execute" not in filters: do_executes = False
        if "fidelity" not in filters: do_fidelities = False
        if "hf_fidelity" not in filters: do_hf_fidelities = False
        if "depth" not in filters: do_depths = False
        if "2q" not in filters: do_2qs = False
        if "vbplot" not in filters: do_vbplot = False
        
        # this is a way to turn these on, if aq_mode not used
        if "hf_fidelity" in filters: do_hf_fidelities = True
        if "2q" in filters: do_2qs = True
    
    # generate one-column figure with multiple bar charts, with shared X axis
    cols = 1
    fig_w = 6.0
    
    numplots = 0
    if do_creates: numplots += 1
    if do_executes: numplots += 1
    if do_fidelities: numplots += 1
    if do_depths: numplots += 1
    if do_2qs: numplots += 1
    
    rows = numplots
    
    # DEVNOTE: this calculation is based on visual assessment of results and could be refined
    # compute height needed to draw same height plots, no matter how many there are
    fig_h = 3.5 + 2.0 * (rows - 1) + 0.25 * (rows - 1)
    #print(fig_h)
    
    # create the figure into which plots will be placed
    fig, axs = plt.subplots(rows, cols, sharex=True, figsize=(fig_w, fig_h))
    
    # append key circuit metrics info to the title
    fulltitle = suptitle + get_backend_title()
    if options != None:
        options_str = ''
        for key, value in options.items():
            if len(options_str) > 0: options_str += ', '
            options_str += f"{key}={value}"
        fulltitle += f"\n{options_str}"

    # and add the title to the plot
    plt.suptitle(fulltitle)
    
    axi = 0
    xaxis_set = False
    
    if rows == 1:
        ax = axs
        axs = [ax]
        
    groups = group_metrics["groups"]
    # print(f"... groups - {groups}")
    
    xlabels = None
    
    # check if elements of groups are unique
    # if not, there are multiple execution groups within each group
    # and we need to reduce to unique set of data for plotting
    if len(set(groups)) != len(groups):
        # print("*** groups are NOT unique")
        
        xlabels = groups
        # print(f"... labels = {xlabels}")
        
        groups = [i for i in range(len(groups))]
        # print(f"... new groups = {groups}")
        
        '''   DEVNOTE --- WIP. Attempt to reduce duplicate widths by averaging. 
        g = []
        ahf = []
        af = []
        shf = []
        sf = []
        
        lastgroup = 0
        ii = 0
        jj = 0
        for group in groups:
            print(group)
          
            if group != lastgroup:
                g.append([])
                g[jj] = group
                
                ahf.append([])
                af.append([])
                shf.append([])
                sf.append([])
                
                jj += 1

            else:
                print("same")
                
            ahf[jj] = group
            af[jj] = group
            
            lastgroup = group
         
        print("... new:")
        print(g)
        print(ahf)
        print(af)
        '''
 
    
    if do_creates:
    
        # set ticks specially if we had non-unique group names
        if xlabels is not None:
            axs[axi].set_xticks(groups)
            axs[axi].set_xticklabels(xlabels)
            
        if max(group_metrics["avg_create_times"]) < 0.01:
            axs[axi].set_ylim([0, 0.01])
        axs[axi].grid(True, axis = 'y', color='silver', zorder = 0)
        axs[axi].bar(groups, group_metrics["avg_create_times"], zorder = 3)
        axs[axi].set_ylabel('Avg Creation Time (sec)')
        
        # error bars
        zeros = [0] * len(group_metrics["avg_create_times"])
        std_create_times = group_metrics["std_create_times"] if "std_create_times" in group_metrics else zeros
        
        axs[axi].errorbar(groups, group_metrics["avg_create_times"], yerr=std_create_times,
                ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5, ls='',
                marker = "D", markersize = 3, mfc = 'c', mec = 'k', mew = 0.5,
                label = 'Error', alpha = 0.75, zorder = 3)
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
        plt.setp(axs[axi].get_xticklabels(), visible=False)
        axi += 1
    
    if do_executes:
    
        # set ticks specially if we had non-unique group names
        if xlabels is not None:
            axs[axi].set_xticks(groups)
            axs[axi].set_xticklabels(xlabels)
    
        avg_exec_times = group_metrics["avg_exec_times"]
        avg_elapsed_times = group_metrics["avg_elapsed_times"]
        avg_exec_creating_times = group_metrics["avg_exec_creating_times"]
        
        # Attempt to remove queue time from elapsed, heuristically
        avg_elapsed_times = modify_elapsed_times(avg_elapsed_times,
                avg_exec_creating_times, avg_exec_times)
 
        # ensure existence of std arrays (for backwards compatibility)
        zeros = [0] * len(avg_exec_times)
        std_exec_times = group_metrics["std_exec_times"] if "std_exec_times" in group_metrics else zeros
        std_elapsed_times = group_metrics["std_elapsed_times"] if "std_elapsed_times" in group_metrics else zeros
        
        axs[axi].grid(True, axis = 'y', color='silver', zorder = 0)
        
        if show_elapsed_times:    # a global setting
            axs[axi].bar(groups, avg_elapsed_times, 0.75, color='skyblue', alpha = 0.8, zorder = 3)
            
            if max(avg_elapsed_times) < 0.1 and max(avg_exec_times) < 0.1:
                axs[axi].set_ylim([0, 0.1])
        else:
            if max(avg_exec_times) < 0.1:
                axs[axi].set_ylim([0, 0.1])
            
        axs[axi].bar(groups, avg_exec_times, 0.55 if show_elapsed_times is True else 0.7, zorder = 3)
        axs[axi].set_ylabel('Avg Execution Time (sec)')
        
        # error bars
        if show_elapsed_times:
            if std_elapsed_times is not None:
                axs[axi].errorbar(groups, avg_elapsed_times, yerr=std_elapsed_times,
                    ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5, ls='',
                    marker = "D", markersize = 3, mfc = 'c', mec = 'k', mew = 0.5,
                    label = 'Error', alpha = 0.75, zorder = 3)
        
        if std_exec_times is not None:
            axs[axi].errorbar(groups, avg_exec_times, yerr=std_exec_times,
                ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5, ls='',
                marker = "D", markersize = 3, mfc = 'c', mec = 'k', mew = 0.5,
                label = 'Error', alpha = 0.75, zorder = 3)

        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
        
        if show_elapsed_times:
            axs[axi].legend(['Elapsed', 'Quantum'], loc='upper left')
        #else:
            #axs[axi].legend(['Quantum'], loc='upper left') 
        
        ###################################
        # optional log axis processing
        
        use_logscale_for_times = False
        
        # determine min and max of both data sets, with a lower limit of 0.1
        y_max_0 = max(avg_exec_times)
        y_max_0 = max(0.10, y_max_0)
        
        # for min, assume 0.001 is the minimum, in case it is 0
        y_min_0 = get_nonzero_min(avg_exec_times)

        if show_elapsed_times:
            y_max_0 = max(y_max_0, max(avg_elapsed_times))
            y_min_0 = min(y_min_0, get_nonzero_min(avg_elapsed_times))
        
        # make just a little larger for autoscaling
        y_max_0 *= 1.1                
        y_min_0 = y_min_0 / 1.1
        
        # for min, assume 0.001 is the minimum, in case it is 0
        if y_min_0 <= 0:
            y_min_0 = 0.001
        
        # print(f"{y_min_0} {y_max_0}")
        
        # force use of logscale if total range ratio above the threshold
        if logscale_for_times_threshold > 0 and y_max_0 / y_min_0 > logscale_for_times_threshold:
            use_logscale_for_times = True
            
        # set up log scale if specified
        if use_logscale_for_times:
            axs[axi].set_yscale('log') 
            
            #if y_max_0 > 0.01:
            y_max_0 *= 1.6
            y_min_0 /= 1.6
            
            if y_max_0 > 0.001 and (y_max_0 / y_min_0) < logscale_for_times_threshold:
                y_min_0 = y_max_0 / logscale_for_times_threshold
        
        # always start at 0 if not log scale
        else:
            y_min_0 = 0
        
        # set full range of the y-axis
        # print(f"{y_min_0} {y_max_0}")
        axs[axi].set_ylim([y_min_0, y_max_0])
        
        # none of these methods of sharing the x axis gives proper effect; makes extra white space
        #axs[axi].sharex(axs[2])
        #plt.setp(axs[axi].get_xticklabels(), visible=False)
        #axs[axi].set_xticklabels([])
        axi += 1
    
    if do_fidelities:
    
        #print(f"... do fidelities for group {group_metrics}")
        
        axs[axi].set_ylim([0, 1.1])
        axs[axi].grid(True, axis = 'y', color='silver', zorder = 0)
        
        axs[axi].set_ylabel('Avg Result Fidelity')
        
        #groups = group_metrics["groups"]
        #print(f"... groups - {groups}")
        
        # fidelity data
        avg_hf_fidelities = group_metrics["avg_hf_fidelities"]
        avg_fidelities = group_metrics["avg_fidelities"]
        
        #print(avg_hf_fidelities)
        #print(avg_fidelities)

        # standard error data
        zeros = [0] * len(group_metrics["avg_fidelities"])
        std_hf_fidelities = group_metrics["std_hf_fidelities"] if "std_hf_fidelities" in group_metrics else zeros
        std_fidelities = group_metrics["std_fidelities"] if "std_fidelities" in group_metrics else zeros
        
        #print(std_hf_fidelities)
        #print(std_fidelities)
           
        # set ticks specially if we had non-unique group names
        if xlabels is not None:
            axs[axi].set_xticks(groups)
            axs[axi].set_xticklabels(xlabels)
            
        # data bars
        axs[axi].bar(groups, avg_hf_fidelities, color='skyblue', alpha = 0.8, zorder = 3)
        axs[axi].bar(groups, avg_fidelities, 0.55, zorder = 3) 
        
        # error bars
        axs[axi].errorbar(groups, avg_fidelities, yerr=std_fidelities,
                ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5, ls='',
                marker = "D", markersize = 3, mfc = 'c', mec = 'k', mew = 0.5,
                label = 'Error', alpha = 0.75, zorder = 3)
        
        axs[axi].errorbar(groups, avg_hf_fidelities, yerr=std_hf_fidelities,
                ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5, ls='',
                marker = "D", markersize = 3, mfc = 'c', mec = 'k', mew = 0.5,
                label = 'Error', alpha = 0.75, zorder = 3)
                
        #[axi].set_xticklabels(xlabels)
        
        # share the x axis if it isn't already
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        axs[axi].legend(['Hellinger', 'Normalized'], loc='upper right')
        axi += 1
        
    if do_depths:
        
        # set ticks specially if we had non-unique group names
        if xlabels is not None:
            axs[axi].set_xticks(groups)
            axs[axi].set_xticklabels(xlabels)
        
        # using one axis for circuit depth
        if not use_two_depth_axes:
        
            if max(group_metrics["avg_tr_depths"]) < 20:
                axs[axi].set_ylim([0, 20])  
            axs[axi].grid(True, axis = 'y', color='silver', zorder = 0)
            
            axs[axi].bar(groups, group_metrics["avg_depths"], 0.8, zorder = 3)
            axs[axi].bar(groups, group_metrics["avg_tr_depths"], 0.5, color='C9', zorder = 3) 
            axs[axi].set_ylabel('Circuit Depth')      
        
        # using two axes for circuit depth
        else:
        
            ax2 = axs[axi].twinx()
            
            if max(group_metrics["avg_depths"]) < 20:
                axs[axi].set_ylim([0, 2 * 20]) 
            else:
                axs[axi].set_ylim([0, 2 * max(group_metrics["avg_depths"])])
                
            if max(group_metrics["avg_tr_depths"]) < 20:
                axs[axi].set_ylim([0, 20])
            
            # plot algo and normalized depth on same axis, but norm is invisible
            axs[axi].grid(True, axis = 'y', color='silver', zorder = 0)
            axs[axi].set_ylabel('Algorithmic Circuit Depth')
            
            axs[axi].bar(groups, group_metrics["avg_depths"], 0.8, zorder = 3)
            
            # use width = 0 to make it invisible
            yy0 = [0.0 for y in group_metrics["avg_tr_depths"]]
            axs[axi].bar(groups, yy0, 0.0, color='C9', zorder = 3) 
            #axs[axi].bar(groups, group_metrics["avg_tr_depths"], 0.0, color='C9', zorder = 3)
            
            # plot normalized on second axis
            if max(group_metrics["avg_tr_depths"]) < 20:
                ax2.set_ylim([0, 20])
                
            ax2.grid(True, axis = 'y', color='silver', ls='dashed', zorder = 0)
            ax2.set_ylabel('Normalized Circuit Depth')
            
            ax2.bar(groups, group_metrics["avg_tr_depths"], 0.45, color='C9', zorder = 3)
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        axs[axi].legend(['Algorithmic Depth', 'Normalized Depth'], loc='upper left')
        axi += 1
    
    if do_2qs:
    
        # set ticks specially if we had non-unique group names
        if xlabels is not None:
            axs[axi].set_xticks(groups)
            axs[axi].set_xticklabels(xlabels)
            
        if max(group_metrics["avg_tr_n2qs"]) < 20:
            axs[axi].set_ylim([0, 20])  
        axs[axi].grid(True, axis = 'y', color='silver', zorder = 0)
        axs[axi].bar(groups, group_metrics["avg_tr_n2qs"], 0.5, color='C9', zorder = 3) 
        axs[axi].set_ylabel('2Q Gates')
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        axs[axi].legend(['Normalized 2Q Gates'], loc='upper left')
        axi += 1
        
    # shared x axis label
    axs[rows - 1].set_xlabel('Circuit Width (Number of Qubits)')
     
    fig.tight_layout() 
    
    # save plot image to file
    if save_plot_images:
        save_plot_image(plt, f"{appname}-metrics" + suffix, backend_id) 
            
    # show the plot for user to see
    if show_plot_images:
        plt.show()
    
    ###################### Volumetric Plot
        
    suptitle = f"Volumetric Positioning - {appname}"
    
    # append key circuit metrics info to the title
    fulltitle = suptitle + get_backend_title()
    if options != None:
        options_str = ''
        for key, value in options.items():
            if len(options_str) > 0: options_str += ', '
            options_str += f"{key}={value}"
        fulltitle += f"\n{options_str}"
        
    # note: if using filters, both "depth or 2qs" and "vbplot" must be set for this to draw
    # with some packages, like Cirq and Braket, we do not calculate depth metrics or 2qs
    
    # generate separate figure for volumetric positioning chart of depth metrics
    if {do_depths or do_2qs} and do_volumetric_plots and do_vbplot:
        
        w_data = group_metrics["groups"]
        if aq_mode > 0:
            d_tr_data = group_metrics["avg_tr_n2qs"]
        else:
            d_tr_data = group_metrics["avg_tr_depths"]
        f_data = group_metrics["avg_fidelities"]
        
        try:            
            vplot_anno_init()
            
            max_qubits = max([int(group) for group in w_data])
            
            if aq_mode > 0:
                ax = plot_volumetric_background_aq(max_qubits=max_qubits, AQ=0,
                    depth_base=depth_base, suptitle=fulltitle, colorbar_label="Avg Result Fidelity")
            else:
                ax = plot_volumetric_background(max_qubits=max_qubits, QV=QV,
                    depth_base=depth_base, suptitle=fulltitle, colorbar_label="Avg Result Fidelity")
            
            # determine width for circuit
            w_max = 0
            for i in range(len(w_data)):
                y = float(w_data[i])
                w_max = max(w_max, y)

            # If using mid-circuit transformation, convert width data to singular circuit width value
            if transform_qubit_group:
                w_data = new_qubit_group
                group_metrics["groups"] = w_data

            if aq_mode > 0:
                plot_volumetric_data_aq(ax, w_data, d_tr_data, f_data, depth_base, fill=True,
                        label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)
            else:
                plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base, fill=True,
                        label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)

            anno_volumetric_data(ax, depth_base,
                label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False)
        
        except Exception as e:
            print('ERROR: plot_metrics(), failure when creating volumetric positioning chart')
            print(f"... exception = {e}")
            if verbose:
                print(traceback.format_exc())
        
        # save plot image to file
        if save_plot_images:
            save_plot_image(plt, f"{appname}-vplot", backend_id) 
        
        #display plot
        if show_plot_images:
            plt.show()

    # generate separate figure for volumetric positioning chart of depth metrics
    if aq_mode > 0 and {do_depths or do_2qs} and do_volumetric_plots and do_vbplot:
        
        w_data = group_metrics["groups"]
        d_tr_data = group_metrics["avg_tr_n2qs"]
        f_data = group_metrics["avg_hf_fidelities"]
        
        try:            
            vplot_anno_init()
            
            max_qubits = max([int(group) for group in w_data])
            
            ax = plot_volumetric_background_aq(max_qubits=max_qubits, AQ=0,
                depth_base=depth_base, suptitle=fulltitle, colorbar_label="Avg Hellinger Fidelity")
            
            # determine width for circuit
            w_max = 0
            for i in range(len(w_data)):
                y = float(w_data[i])
                w_max = max(w_max, y)

            # If using mid-circuit transformation, convert width data to singular circuit width value
            if transform_qubit_group:
                w_data = new_qubit_group
                group_metrics["groups"] = w_data

            plot_volumetric_data_aq(ax, w_data, d_tr_data, f_data, depth_base, fill=True,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)  
            
            anno_volumetric_data(ax, depth_base,
                label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False)
        
        except Exception as e:
            print('ERROR: plot_metrics(), failure when creating volumetric positioning chart')
            print(f"... exception = {e}")
            if verbose:
                print(traceback.format_exc())
        
        # save plot image to file
        if save_plot_images:
            save_plot_image(plt, f"{appname}-vplot-hf", backend_id) 
        
        #display plot
        if show_plot_images:
            plt.show()

# Return the minimum value in an array, but if all elements 0, return 0.001
def get_nonzero_min(array):
    f_array = list(filter(lambda x: x > 0, array)) 
    if len(f_array) < 1: f_array = [0.001]
    return min(f_array)

# Return a modifed copy of the elapsed time, removing queue time if possible using heuristics
def modify_elapsed_times(avg_elapsed_times, avg_exec_creating_times, avg_exec_times):

    # Make a copy of the elapsed times array since we may modify it
    avg_elapsed_times = [et for et in avg_elapsed_times]
    
    # DEVNOTE: on some machines (IBM, specifically), the creating time includes the queue time.
    # We can remove queue time from elapsed time by subtracting the creating time.
    # The flaw in this is that it also removes the compilation time, which is small
    # for small circuits, but could be larger for large circuits.
    # Thus, we've added the variable to enable/disable this.
    if remove_creating_time_from_elapsed and len(avg_exec_creating_times) >= len(avg_elapsed_times):
        for i in range(len(avg_elapsed_times)):
            avg_elapsed_times[i] = round(avg_elapsed_times[i] - avg_exec_creating_times[i], 3) 

    # DEVNOTE: A brutally simplistic way to toss out initially long elapsed times
    # that are most likely due to either queueing or system initialization
    if show_elapsed_times and omit_initial_elapsed_time_factor > 0:
        for i in range(len(avg_elapsed_times)):
            if avg_elapsed_times[i] > omit_initial_elapsed_time_factor * avg_exec_times[i]:
                avg_elapsed_times[i] = avg_exec_times[i] * initial_elapsed_time_multiplier
    
    return avg_elapsed_times
    
#################################################

# DEVNOTE: this function is not used, as the overlaid rectanges are not useful

# Plot metrics over all groups (2)
def plot_metrics_all_overlaid (shared_data, backend_id, suptitle=None, imagename="_ALL-vplot-1"):    
    
    global circuit_metrics
    global group_metrics
    
    subtitle = circuit_metrics["subtitle"]
    
    print("Overlaid Results From All Applications")
    
    # generate separate figure for volumetric positioning chart of depth metrics
    
    try:    
        # determine largest width for all apps
        w_min, w_max = get_min_max(shared_data)

        # allow one more in width to accommodate the merge values below
        max_qubits = int(w_max) + 1     
        #print(f"... {w_max} {max_qubits}")
        
        if aq_mode > 0:
            ax = plot_volumetric_background_aq(max_qubits=max_qubits, AQ=AQ,
                depth_base=depth_base, suptitle=suptitle, colorbar_label="Avg Hellinger Fidelity")
        else:
            ax = plot_volumetric_background(max_qubits=max_qubits, QV=QV,
                depth_base=depth_base, suptitle=suptitle, colorbar_label="Avg Result Fidelity")
            
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
            
            if "avg_tr_n2qs" not in group_metrics: continue
            n2q_tr_data = group_metrics["avg_tr_n2qs"]
    
            if aq_mode > 0:
                if "avg_hf_fidelities" not in group_metrics: continue
                f_data = group_metrics["avg_hf_fidelities"]
                plot_volumetric_data_aq(ax, w_data, n2q_tr_data, f_data, depth_base, fill=True,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)
            else:
                f_data = group_metrics["avg_fidelities"]
                plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base, fill=True,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)  

        # do annotation separately, spreading labels for readability
        anno_volumetric_data(ax, depth_base,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False)
    
    except Exception as e:
        print('ERROR: plot_metrics_all_overlaid(), failure when creating volumetric positioning chart')
        print(f"... exception = {e}")
        if verbose:
            print(traceback.format_exc())
    
    # save plot image file
    if save_plot_images:
        if aq_mode > 0: imagename += '-aq'
        save_plot_image(plt, imagename, backend_id) 
    
    #display plot
    if show_plot_images:
        plt.show()


#################################################

# Plot metrics over all groups (level 2), merging data from all apps into smaller cells if not is_individual
def plot_metrics_all_merged (shared_data, backend_id, suptitle=None,
            imagename="_ALL-vplot-2", avail_qubits=0,
            is_individual=False, score_metric=None,
            max_depth=0, suppress_low_fidelity=False):                   
    
    global circuit_metrics
    global group_metrics
    
    # determine the metric to use for scoring, i.e. the color of plot items
    if score_metric == None:
        if aq_mode > 0:
            score_metric = "avg_hf_fidelities"
        else:
            score_metric = "avg_fidelities"
    
    # if aq_mode, force is_individual to be True (cannot blend the circles's colors)
    if aq_mode > 0:
        is_individual = True
    
    # determine the label for the colorbar
    if score_metric == "avg_hf_fidelities": 
        colorbar_label="Avg Hellinger Fidelity"
    elif score_metric == "avg_fidelities":
        colorbar_label="Avg Result Fidelity"
    else:
        colorbar_label="Unknown Measure"
  
    # generate separate figure for volumetric positioning chart of depth metrics

    try:       
        # determine largest width for all apps
        w_min, w_max = get_min_max(shared_data, suppress_low_fidelity=suppress_low_fidelity)
        
        #determine width for AQ
        AQ = get_aq_width(shared_data, w_min, w_max, score_metric)

        # allow one more in width to accommodate the merge values below
        max_qubits = int(w_max) + 1  
        
        # draw the appropriate background, given the AQ mode
        if aq_mode > 0:
            ax = plot_volumetric_background_aq(max_qubits=max_qubits, AQ=AQ, depth_base=depth_base,
                suptitle=suptitle, avail_qubits=avail_qubits, colorbar_label=colorbar_label)
        else:
            ax = plot_volumetric_background(max_qubits=max_qubits, QV=QV, depth_base=depth_base,
                suptitle=suptitle, avail_qubits=avail_qubits, colorbar_label=colorbar_label)
        
        # create 2D array to hold merged value arrays with gradations, one array for each qubit size
        # plot rectangles representing these result gradations
        if not is_individual:
            plot_merged_result_rectangles(shared_data, ax, max_qubits, w_max, score_metric=score_metric,
                    max_depth=max_depth, suppress_low_fidelity=suppress_low_fidelity)
        
        # Now overlay depth metrics for each app with unfilled rects, to outline each circuit
        # if is_individual, do filled rects as there is no background fill
        
        vplot_anno_init()
        
        # Note: the following loop is required, as it creates the array of annotation points
        # In this merged version of plottig, we suppress the border as it is already drawn
        appname = None
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
                   
            if "avg_tr_n2qs" not in group_metrics: continue
            n2q_tr_data = group_metrics["avg_tr_n2qs"]
    
            filled = is_individual
            if aq_mode > 0:
                if score_metric not in group_metrics: continue
                f_data = group_metrics[score_metric]
                plot_volumetric_data_aq(ax, w_data, n2q_tr_data, f_data, depth_base, fill=filled,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max,
                   max_depth=max_depth, suppress_low_fidelity=suppress_low_fidelity)
            else:
                f_data = group_metrics[score_metric]
                plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base, fill=filled,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max,
                   max_depth=max_depth, suppress_low_fidelity=suppress_low_fidelity,
                   do_border=False)
        
        if appname == None:
            print(f"ERROR: cannot find data file for: {get_backend_label()}")
            
        # do annotation separately, spreading labels for readability
        anno_volumetric_data(ax, depth_base,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False)
    
    except Exception as e:
        print('ERROR: plot_metrics_all_merged(), failure when creating volumetric positioning chart')
        print(f"... exception = {e}")
        if verbose:
            print(traceback.format_exc())
    
    # save plot image file
    if save_plot_images:
        if aq_mode > 0: imagename += '-aq'
        save_plot_image(plt, imagename, backend_id)

    #display plot
    if show_plot_images:
        plt.show()


# Plot filled but borderless rectangles based on merged gradations of result metrics
def plot_merged_result_rectangles(shared_data, ax, max_qubits, w_max, num_grads=4, score_metric=None,
            max_depth=0, suppress_low_fidelity=False):

    depth_values_merged = []
    for w in range(max_qubits):
        depth_values_merged.append([ None ] * (num_grads * max_depth_log))
    
    # keep an array of the borders squares' centers 
    borders = []
    
    # run through depth metrics for all apps, splitting cells into gradations
    for app in shared_data:
        
        # Extract shorter app name from the title passed in by user
        appname = app[len('Benchmark Results - '):len(app)]
        appname = appname[:appname.index(' - ')]
        
        group_metrics = shared_data[app]["group_metrics"]
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
        
        if score_metric not in group_metrics: continue
        f_data = group_metrics[score_metric]
        
        if aq_mode > 0:
            if "avg_tr_n2qs" not in group_metrics: continue
            n2q_tr_data = group_metrics["avg_tr_n2qs"]
            d_tr_data = n2q_tr_data
            
        # instead of plotting data here, split into gradations in code below
        #plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base,
               #label=appname, labelpos=(0.4, 0.6), labelrot=50, type=1)  

        # aggregate value metrics for each depth cell over all apps
        low_fidelity_count = True
        for i in range(len(d_data)):
            x = depth_index(d_tr_data[i], depth_base)
            y = float(w_data[i])
            f = f_data[i]
            
            if max_depth > 0 and d_tr_data[i] > max_depth:
                print(f"... excessive depth, skipped; w={y} d={d_tr_data[i]}")
                break
                    
            # reject cells with low fidelity
            if suppress_low_fidelity and f < suppress_low_fidelity_level:
                if low_fidelity_count: break
                else: low_fidelity_count = True
                    
            # accumulate largest width for all apps
            w_max = max(w_max, y)
            
            xp = x * 4
            
            #store center of border rectangle
            borders.append((int(xp), y))
            
            if x > max_depth_log - 1:
                print(f"... data out of chart range, skipped; w={y} d={d_tr_data[i]}")
                break
                
            for grad in range(num_grads):
                e = depth_values_merged[int(w_data[i])][int(xp + grad)]
                if e == None: 
                    e = { "value": 0.0, "count": 0 }
                e["count"] += 1
                e["value"] += f
                depth_values_merged[int(w_data[i])][int(xp + grad)] = e
    
    #for depth_values in depth_values_merged:
        #print(f"-- {depth_values}")
            
    # compute and plot the average fidelity at each width / depth gradation with narrow filled rects 
    for wi in range(len(depth_values_merged)):
        w = depth_values_merged[wi]
        #print(f"... w = {w}")
        
        low_fidelity_count = True
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
                
                # reject cells with low fidelity
                if suppress_low_fidelity and f < suppress_low_fidelity_level:
                    if low_fidelity_count: break
                    else: low_fidelity_count = True
                ax.add_patch(box4_at(x, y, f, type=1, fill=True))
        
    # draw borders at w,d location of each cell, offset to account for the merge process above
    for (x,y) in borders: 
        x = x/4 + 0.125
        ax.add_patch(box_at(x, y, f, type=1, fill=False))
        
    #print("**** merged...")
    #print(depth_values_merged)
    

#################################################

### plot metrics across all apps for a backend_id

def plot_all_app_metrics(backend_id, do_all_plots=False,
        include_apps=None, exclude_apps=None, suffix="", avail_qubits=0,
        is_individual=False, score_metric=None,
        filters=None, options=None,
        max_depth=0, suppress_low_fidelity=False):

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

    # show vplots if enabled
    if do_volumetric_plots:
    
        # this is an overlay plot, no longer used, not very useful; better to merge
        '''
        suptitle = f"Volumetric Positioning - All Applications (Combined)\nDevice={backend_id}  {get_timestr()}"
        plot_metrics_all_overlaid(shared_data, backend_id, suptitle=suptitle, imagename="_ALL-vplot-2")
        
        '''

        # draw the volumetric plot and append the circuit metrics subtitle to the title
        suptitle = "Volumetric Positioning - All Applications (Merged)"
        fulltitle = suptitle + get_backend_title()
        
        plot_metrics_all_merged(shared_data, backend_id, suptitle=fulltitle, 
                imagename="_ALL-vplot-2"+suffix, avail_qubits=avail_qubits,
                is_individual=is_individual, score_metric=score_metric,
                max_depth=max_depth, suppress_low_fidelity=suppress_low_fidelity)
        
    # show all app metrics charts if enabled
    if do_app_charts_with_all_metrics or do_all_plots:
        for app in shared_data:
            #print("")
            #print(app)
            group_metrics = shared_data[app]["group_metrics"]
            plot_metrics(app, filters=filters, options=options)


### Plot Metrics for a specific application

def plot_metrics_for_app(backend_id, appname, apiname="Qiskit", filters=None, options=None, suffix=""):
    global circuit_metrics
    global group_metrics

    # load saved data from file
    api = "qiskit"
    shared_data = load_app_metrics(api, backend_id)
    
    # since the bar plots use the subtitle field, set it here
    circuit_metrics["subtitle"] = f"device = {backend_id}"
        
    app = "Benchmark Results - " + appname + " - " + apiname
    
    if app not in shared_data:
        print(f"ERROR: cannot find app: {appname}")
        return
    
    group_metrics = shared_data[app]["group_metrics"]
    plot_metrics(app, filters=filters, suffix=suffix, options=options)

# save plot as image
def save_plot_image(plt, imagename, backend_id):

    # don't leave slashes in the filename
    backend_id = backend_id.replace("/", "_")
     
    # not used currently
    date_of_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists('__images'): os.makedirs('__images')
    if not os.path.exists(f'__images/{backend_id}{data_suffix}'): os.makedirs(f'__images/{backend_id}{data_suffix}')
    
    pngfilename = f"{backend_id}{data_suffix}/{imagename}"
    pngfilepath = os.path.join(os.getcwd(),"__images", pngfilename + ".jpg")
    
    plt.savefig(pngfilepath)
    
    #print(f"... saving (plot) image file:{pngfilename}.jpg")   
    
    pdffilepath = os.path.join(os.getcwd(),"__images", pngfilename + ".pdf")
    
    plt.savefig(pdffilepath)
    
    
#################################################
# ANALYSIS AND VISUALIZATION - AREA METRICS PLOTS

# map known X metrics to labels    
known_x_labels = {
    'cumulative_create_time' : 'Cumulative Circuit Creation Time (s)',
    'cumulative_elapsed_time' : 'Cumulative Elapsed Quantum Execution Time (s)',
    'cumulative_exec_time' : 'Cumulative Quantum Execution Time (s)',
    'cumulative_opt_exec_time' : 'Cumulative Classical Optimization Time (s)',
    'cumulative_depth' : 'Cumulative Circuit Depth'
}

x_label_save_str = {
    'create_time' : 'createTime',
    'elapsed_time' : 'elapsedTime',
    'exec_time' : 'execTime',
    'opt_exec_time' : 'optTime',
    'depth' : 'depth'
}


# map known Y metrics to labels
known_y_labels = {
    #'num_qubits' : 'Circuit Width'     # this is only used for max-cut area plots 
    #'num_qubits' : 'Problem Size (Number of Variables)'      # use Problem Size instead
    'num_qubits' : 'Problem Size (# of Variables)'      # use Problem Size instead
}
# map known Score metrics to labels    
known_score_labels = {
    'approx_ratio' : 'Approximation Ratio',
    'cvar_ratio' : 'CVaR Ratio',
    'gibbs_ratio' : 'Gibbs Objective Function',
    'bestcut_ratio' : 'Best Measurement Ratio',
    'fidelity' : 'Result Fidelity',
    'max_fidelity' : 'Max. Result Fidelity',
    'hf_fidelity' : 'Hellinger Fidelity'
}

# string that will go into the name of the figure when saved
score_label_save_str = {
    'approx_ratio' : 'apprRatio',
    'cvar_ratio' : 'CVaR',
    'bestcut_ratio' : 'bestCut',
    'gibbs_ratio' : 'gibbs',
    'fidelity' : 'fidelity',
    'hf_fidelity' : 'hf'
}

 
# Plot all the given "Score Metrics" against the given "X Metrics" and "Y Metrics" 
def plot_all_area_metrics(suptitle='',
            score_metric='fidelity', x_metric='cumulative_exec_time', y_metric='num_qubits',
            fixed_metrics={}, num_x_bins=100,
            y_size=None, x_size=None, x_min=None, x_max=None, offset_flag=False,
            options=None, suffix='', which_metric='approx_ratio'):

    # if no metrics to plot, just return
    if score_metric is None or x_metric is None or y_metric is None:
        return
        
    if type(score_metric) == str:
        score_metric = [score_metric]
    if type(x_metric) == str:
        x_metric = [x_metric]
    if type(y_metric) == str:
        y_metric = [y_metric]
    
    # loop over all the given X and Score metrics, generating a plot for each combination
    for s_m in score_metric:
        for x_m in x_metric:
            for y_m in y_metric:
                #print("plotting area metrics for " + s_m + " " + x_m + " " + y_m)
                plot_area_metrics(suptitle, s_m, x_m, y_m, fixed_metrics, num_x_bins, y_size, x_size, x_min, x_max, offset_flag=offset_flag, options=options, suffix=suffix, which_metric=which_metric)

def get_best_restart_ind(group, which_metric = 'approx_ratio'):
    """
    From all the restarts, obtain the restart index for which the final iteration has the highest value of the specified metric

    Args:
        group (str): circuit width
        which_metric (str, optional): Defaults to 'approx_ratio'. Other valid options are 'gibbs_ratio', 'cvar_ratio', 'bestcut_ratio'
    """
    restart_indices = list(circuit_metrics_detail_2[group].keys())
    fin_AR_restarts = []
    for restart_ind in restart_indices:
        iter_inds = list(circuit_metrics_detail_2[group][restart_ind].keys())
        fin_AR = circuit_metrics_detail_2[group][restart_ind][max(iter_inds)][which_metric]
        fin_AR_restarts.append(fin_AR)
    best_index = fin_AR_restarts.index(max(fin_AR_restarts))
    
    return restart_indices[best_index]

# Plot the given "Score Metric" against the given "X Metric" and "Y Metric"
def plot_area_metrics(suptitle='',
            score_metric='fidelity', x_metric='cumulative_exec_time', y_metric='num_qubits', fixed_metrics={}, num_x_bins=100,
            y_size=None, x_size=None, x_min=None, x_max=None, offset_flag=False,
            options=None, suffix='', which_metric='approx_ratio'):
    """
    Plots a score metric as an area plot, on axes defined by x_metric and y_metric
    
    fixed_metrics: (dict) A dictionary mapping metric keywords to the values they are to be held at;
        for example: 
        
        fixed_metrics = {'rounds': 2}
            
            when the y-axis is num_qubits or 
        
        fixed_metrics = {'num_qubits': 4}
        
            when the y-axis is rounds.    
    """
    # get backend id for this set of circuits
    backend_id = get_backend_id()
    
    # Extract shorter app name from the title passed in by user   
    appname = get_appname_from_title(suptitle)
    
    # map known metrics to labels    
    x_label = known_x_labels[x_metric]
    y_label = known_y_labels[y_metric]
    score_label = known_score_labels[score_metric]
    #print("plotting area metrics for " + score_label + " " + x_label + " " + y_label)
    
    # process cumulative and maximum options
    xs, x, y, scores = [], [], [], []
    cumulative_flag, maximum_flag = False, False
    if len(x_metric) > 11 and x_metric[:11] == 'cumulative_':
        cumulative_flag = True
        x_metric = x_metric[11:]
    if score_metric[:4] == 'max_':
        maximum_flag = True
        score_metric = score_metric[4:]  
    
    #print(f"  ==> all detail 2 circuit_metrics:")
    for group in circuit_metrics_detail_2:
        
        num_qubits = int(group)
        
        if 'num_qubits' in fixed_metrics:
            if num_qubits != fixed_metrics['num_qubits']:
                continue
        
        x_size_groups, x_groups, y_groups, score_groups = [], [], [], []
        
        # Get the best AR index
        restart_index = get_best_restart_ind(group, which_metric = which_metric)
        
        # Each problem instance at size num_qubits; need to collate across iterations
        for circuit_id in [restart_index]:#circuit_metrics_detail_2[group]:
            # circuit_id here denotes the restart index
                
            x_last, score_last = 0, 0
            x_sizes, x_points, y_points, score_points = [], [], [], []            
            
            metrics_array = circuit_metrics_detail_2[group][circuit_id]
      
            for it in metrics_array:
                mets = metrics_array[it]
                        
                if x_metric not in mets: break
                if score_metric not in mets: break
                
                x_value = mets[x_metric]
                
                # DEVNOTE: A brutally simplistic way to toss out initially long elapsed times
                # that are most likely due to either queueing or system initialization
                if x_metric == 'elapsed_time' and it == 0 and omit_initial_elapsed_time_factor > 0:
                    if (it + 1) in metrics_array:
                        mets2 = metrics_array[it + 1]
                        x_value2 = mets2[x_metric]
                        if x_value > (omit_initial_elapsed_time_factor * x_value2):
                            x_value = x_value2
                            
                # get each metric and accumulate if indicated
                x_raw = x_now = x_value
                if cumulative_flag:
                    x_now += x_last
                x_last = x_now
                
                if y_metric == 'num_qubits':
                    y_now = num_qubits
                else:
                    y_now = mets[y_metric]
                
                # Count only iterations at valid fixed_metric values
                for fixed_m in fixed_metrics:
                    if mets[fixed_m] != fixed_metrics[fixed_m]:
                        continue
                    # Support intervals e.g. {'depth': (15, 65)}
                    elif len(fixed_metrics[fixed_m]) == 2:
                        if mets[fixed_m]<fixed_metrics[fixed_m][0] or mets[fixed_m]>fixed_metrics[fixed_m][1]:
                            continue
                
                if maximum_flag:
                    score_now = max(score_last, mets[score_metric])
                else:
                    score_now = mets[score_metric]
                score_last = score_now
      
                # need to shift x_now by the 'size', since x_now inb the cumulative 
                #x_points.append((x_now - x_raw) if cumulative_flag else x_now)
                #x_points.append((x_now - x_raw))
                x_points.append(x_now - x_raw/2)
                y_points.append(y_now)
                x_sizes.append(x_raw)
                score_points.append(score_now)
            
            x_size_groups.append(x_sizes)
            x_groups.append(x_points)
            y_groups.append(y_points)
            score_groups.append(score_points)
        
        ''' don't do binning for now
        #print(f"  ... x_ = {num_x_bins} {len(x_groups)} {x_groups}")
        #x_sizes_, x_, y_, scores_ = x_bin_averaging(x_size_groups, x_groups, y_groups, score_groups, num_x_bins=num_x_bins)
        '''
        # instead use the last of the groups
        i_last = len(x_groups) - 1
        x_sizes_ = x_size_groups[i_last]
        x_ = x_groups[i_last]
        y_ = y_groups[i_last]
        scores_ = score_groups[i_last]
        
        #print(f"  ... x_ = {len(x_)} {x_}") 
        #print(f"  ... x_sizes_ = {len(x_sizes_)} {x_sizes_}")
        
        xs = xs + x_sizes_
        x = x + x_
        y = y + y_
        scores = scores + scores_

    # append the circuit metrics subtitle to the title
    fulltitle = suptitle + get_backend_title()
    if options != None:
        options_str = ''
        for key, value in options.items():
            if len(options_str) > 0: options_str += ', '
            options_str += f"{key}={value}"
        fulltitle += f"\n{options_str}"
    
    # if the y axis data is sparse or non-linear, linearize the axis data and sort all arrays
    if needs_linearize(y, gap=2):
    
        # sort the data by y axis values, as it may be loaded out of order from file storage
        z = sorted(zip(y,x,xs,scores))
        y = [_y for _y,_x,_xs,_s in z]
        x = [_x for _y,_x,_xs,_s in z]
        xs = [_xs for _y,_x,_xs,_s in z]
        scores = [_s for _y,_x,_xs,_s in z]
        
        # convert irregular y-axis data to linear if any non-linear gaps in the data
        yy, ylabels = linearize_axis(y, gap=2, outer=2, fill=True)
        
    else:
        yy = y
        ylabels = None
       
    # the x axis min/max values will be min(x)/max(x) or values supplied by caller
    if x_min == None:
        #x_min = min(x)
        x_min = 0       # if not supplied, always use 0 for area plots, as leftmost time is 0
        
    if x_max == None:
        x_max = max(x)
        x_max += max(xs)/2  # if not supplied, rightmost time must include the width
    else:
        x_max = x_max - 1  # subtract one to account for the auto-label algorithm in background function
   
    with plt.style.context(maxcut_style):
        # plot the metrics background with its title
        ax = plot_metrics_background(fulltitle, y_label, x_label, score_label,
                    y_max=max(yy), x_max=x_max, y_min=min(yy), x_min=x_min, ylabels=ylabels)
            
        if y_size == None:
            y_size = 1.0
    
        #print(f"... num: {num_x_bins} {len(x)} {x_size} {x}")
    
        # add a grid on the x axis (with the maxcut style of alpha=0.5, this color is best for pdf)
        ax.grid(True, axis = 'x', zorder = 0, color='silver')
    
        # plot all bars, with width specified as an array that matches array size of the x,y values
        plot_volumetric_data(ax, yy, x, scores, depth_base=-1,
                    label='Depth', labelpos=(0.2, 0.7), labelrot=0,
                    type=1, fill=True, w_max=18, do_label=False,
                    x_size=xs, y_size=y_size, zorder=3, offset_flag=offset_flag)                      
        
        plt.tight_layout()
        
        if save_plot_images:
            save_plot_image(plt, os.path.join(f"{appname}-area-"
                                              + score_label_save_str[score_metric] + '-'
                                              + x_label_save_str[x_metric]
                                              + (('-' + suffix) if len(suffix) > 0 else '')),
                                              backend_id)

# Check if axis data needs to be linearized
# Returns true if data sparse or non-linear; sparse means with any gap > gap size
def needs_linearize(values, gap=2):
    #print(f"{values = }")

    # if only one element, no need to linearize
    if len(values) < 2:
        return False
    
    # simple logic for now: return if any gap > 2
    for i in range(len(values)):    
        if i > 0:
            delta = values[i] - values[i - 1]
            if delta > gap:
                return True
                
    # no need to linearize if all small gaps
    return False

# convert irregular axis data to linear, with minimum gap size
# only show labels for the actual data points
# (the labels assume that the return data will be plotted with 2 points before and after)
# DEVNOTE: the use of this function is limited to the special case of the maxcut plots for now,
# given the limited range of problem definitions
def linearize_axis(values, gap=2, outer=2, fill=True):  
    #print(f"{values = }")
    #print(f"{outer = }")
    
    # if only one element, no need to linearize
    if len(values) < 2:
        return values, None
    
    # use this flag to test if there are gaps > gap
    gaps_exist = False
    
    # add labels at beginning
    basis = [None] * outer

    # loop over values and generate new values that are separated by the gap value
    newvalues = []
    for i in range(len(values)):
        newvalues.append(values[i])
        
        # first point is unchanged
        if i == 0:
            basis.append(values[i]) 
        
        # subsequent points are unchanged if same, but modified by gap if different
        if i > 0:
            delta = values[i] - values[i - 1]
            if delta == 0:
                #print("delta 0")
                newvalues[i] = newvalues[i - 1]
            elif delta > gap:
                #print("delta 1+")
                gaps_exist = True
                newvalues[i] = newvalues[i - 1] + gap
                if fill and gap > 1: basis.append(None)         # put space between items if gap > 1
                basis.append(values[i])
    
    # add labels at end    
    basis += [None] * outer

    #print(f"{newvalues = }")
    #print(f"{basis = }")
      
    # format new labels as strings, showing only the actual values (non-zero)
    ylabels = [format_number(yb) if yb != None else '' for yb in basis]
    #print(f"{ylabels = }")
    
    if gaps_exist:
        return newvalues, ylabels
    else:
        return values, None
    
  
# Helper function to bin for averaging metrics, for instances occurring at equal num_qubits
# DEVNOTE: this binning approach creates unevenly spaced bins, cannot use the delta between then for size
def x_bin_averaging(x_size_groups, x_groups, y_groups, score_groups, num_x_bins):

    # find min and max across all the groups
    bin_xs, bin_x, bin_y, bin_s = {}, {}, {}, {}
    x_min, x_max = x_groups[0][0], x_groups[0][0]
    for group in x_groups:
        min_, max_ = min(group), max(group)
        if min_ < x_min:
            x_min = min_
        if max_ > x_max:
            x_max = max_
    step = (x_max - x_min)/num_x_bins
    
    # loop over each group
    for group in range(len(x_groups)):      
        
        # for each item in the group, accumulate into bins
        # place into a new bin, if if has larger x value than last one
        k = 0
        for i in range(len(x_groups[group])):
            while x_groups[group][i] >= x_min + k*step:
                k += 1
            if k not in bin_x:
                bin_xs[k] = []
                bin_x[k] = []
                bin_y[k] = []
                bin_s[k] = []
                    
            bin_xs[k] = bin_xs[k] + [x_size_groups[group][i]]
            bin_x[k] = bin_x[k] + [x_groups[group][i]]
            bin_y[k] = bin_y[k] + [y_groups[group][i]]
            bin_s[k] = bin_s[k] + [score_groups[group][i]]
    
    # for each bin, compute average from all the elements in the bin
    new_xs, new_x, new_y, new_s = [], [], [], []    
    for k in bin_x:
        new_xs.append(sum(bin_xs[k])/len(bin_xs[k]))
        new_x.append(sum(bin_x[k])/len(bin_x[k]))
        new_y.append(sum(bin_y[k])/len(bin_y[k]))
        new_s.append(sum(bin_s[k])/len(bin_s[k]))
    
    return new_xs, new_x, new_y, new_s
    

def plot_ECDF(suptitle="",
              options=None, suffix=None):
    """
    Plot the ECDF (Empirical Cumulative Distribution Function)
    for each circuit width and degree

    Parameters
    ----------
    suptitle : 
    options : 
    suffix :
    """
    # get backend id for this set of circuits
    backend_id = get_backend_id()
    
    # Extract shorter app name from the title passed in by user   
    appname = get_appname_from_title(suptitle)

    with plt.style.context(maxcut_style):
        fig, axs = plt.subplots(1, 1)#, figsize=(6.4,4.8))#, constrained_layout=True, figsize=(6,4))#, sharex=True
    
        # Create more appropriate title
        suptitle = "Cumulative Distribution (ECDF) - " + appname
        
        # append key circuit metrics info to the title
        fulltitle = suptitle + get_backend_title()
        if options != None:
            options_str = ''
            for key, value in options.items():
                if len(options_str) > 0: options_str += ', '
                options_str += f"{key}={value}"
            fulltitle += f"\n{options_str}"
        
        # and add the title to the plot
        plt.title(fulltitle)
    
        for group in circuit_metrics_final_iter:
            best_restart_ind = str(get_best_restart_ind(group))
            for restart_ind in [best_restart_ind]:#circuit_metrics_final_iter[group]:
                
                cumul_counts = circuit_metrics_final_iter[group][restart_ind]['cumul_counts']
                unique_sizes = circuit_metrics_final_iter[group][restart_ind]['unique_sizes']
                optimal_value = circuit_metrics_final_iter[group][restart_ind]['optimal_value']
                axs.plot(np.array(unique_sizes) / optimal_value, np.array(cumul_counts) / cumul_counts[-1], marker='o',
                         #ls = '-', label = f"Width={group}")#" degree={deg}") # lw=1,
                         ls = '-', label = f"Problem Size={group}")#" degree={deg}") # lw=1,

        axs.set_ylabel('Fraction of Total Counts')
        axs.set_xlabel(r'$\frac{\mathrm{Cut\ Size}}{\mathrm{Max\ Cut\ Size}}$')
        axs.grid()

        axs.legend(loc='upper left')#loc='center left', bbox_to_anchor=(1, 0.5))

        fig.tight_layout()

        # save plot image to file
        if save_plot_images:
            save_plot_image(plt, f"{appname}-ECDF-" + suffix, backend_id)
            
        # show the plot for user to see
        if show_plot_images:
            plt.show()

def plot_cutsize_distribution(suptitle="Circuit Width (Number of Qubits)",
                              list_of_widths = [],
                              options=None, suffix=None):
    """
    For each circuit size and degree, plot the measured distribution of cutsizes
    corresponding to the last optimizer iteration, as well as uniform random sampling
    """
    

    if not list_of_widths:
        # If list_of_widths is emply, set it to contain all widths
        list_of_widths = list(circuit_metrics_final_iter.keys())
    # Convert list_of_widths elements to string
    list_of_widths = [str(width) for width in list_of_widths]
    
    group_metrics_optgaps = get_distribution_and_stats()
    # 'quantile_optgaps'
    
    for width in list_of_widths:
        plot_cutsize_distribution_single_width(width, suptitle, options, group_metrics_optgaps, suffix)
        
def plot_cutsize_distribution_single_width(width, suptitle, options, group_metrics_optgaps, suffix):
    
    # get backend id
    backend_id = get_backend_id()

    # Extract shorter app name from the title passed in by user
    appname = get_appname_from_title(suptitle)
    with plt.style.context(maxcut_style):
        fig, axs = plt.subplots(1, 1)

        suptitle = "Empirical Distribution of Cut Sizes - " + appname
        fulltitle = get_full_title(
            #suptitle, options) + "\nwidth={}".format(width)
            suptitle, options) + "\nProblem Size = {}".format(width)
        plt.title(fulltitle)

        indx = group_metrics_optgaps['groups'].index(int(width))  # get index corresponding to width
        # Plot distribution of cut sizes for circuit
        dist = group_metrics_optgaps['cutsize_ratio_dist']
        axs.plot(dist['ratios'][indx], dist['frequencies'][indx], marker='o',
                 ls='-', c='k', ms=2, mec='k', mew=0.4, lw=1,
                 label="Circuit Sampling")  # " degree={deg}") # lw=1,

        # Also plot the distribution obtained from uniform random sampling
        dist = group_metrics_optgaps['random_cutsize_ratio_dist']
        axs.plot(dist['ratios'][indx], dist['frequencies'][indx],
             marker='o', ms=1, mec = 'k',mew=0.2, lw=10,alpha=0.5,
             ls = '-', label = "Uniform Random Sampling", c = "pink")  # " degree={deg}") # lw=1,

        # Plot vertical lines corresponding to the various metrics
        plotted_metric_values = []
        for metric in ['approx_ratio', 'cvar_ratio', 'bestcut_ratio', 'gibbs_ratio']:
            curdict = group_metrics_optgaps[metric]
            curmetricval = curdict['ratiovals'][indx]
            lw=1; ls='solid'
            if curmetricval in plotted_metric_values:
                # for lines that will coincide, assign different styles to distinguish them
                lw=1.5; ls='dashed'
            plotted_metric_values.append(curmetricval)
            axs.axvline(x=curmetricval, color=curdict['color'], label=curdict['label'], lw=lw, ls=ls)
            

        axs.set_ylabel('Fraction of Total Counts')
        axs.set_xlabel(r'$\frac{\mathrm{Cut\ Size}}{\mathrm{Max\ Cut\ Size}}$')
        axs.grid()
        axs.set_xlim(left=-0.02, right=1.02)
        axs.legend(loc='upper left')

        fig.tight_layout()

        # save plot image to file
        if save_plot_images:
            save_plot_image(plt, f"{appname}-cutsize_dist-" + suffix + "width-{}".format(width), backend_id)
            
        # show the plot for user to see
        if show_plot_images:
            plt.show()
    

def get_full_title(suptitle = '', options = dict()):
    """
    Return title for figure
    """
    # create title for this set of circuits
    fulltitle = suptitle + get_backend_title()
    if options != None:
        options_str = ''
        for key, value in options.items():
            if len(options_str) > 0: options_str += ', '
            options_str += f"{key}={value}"
        fulltitle += f"\n{options_str}"
    return fulltitle

# Plot angles
def plot_angles_polar(suptitle = '', options=None, suffix = ''):
    """
    Create a polar angle plot, showing the beta and gamma angles
    Parameters
    ----------
    options : dictionary

    Returns
    -------
    None.

    """
    widths = group_metrics['groups']
    num_widths = len(widths)
    maxRadius = 10
    minRadius = 2
    radii = np.linspace(minRadius, maxRadius,num_widths)
    angles_arr = []
    for ind, width_str in enumerate(widths):
        deg = list(circuit_metrics[width_str].keys())[0]
        angles = circuit_metrics_final_iter[width_str][str(deg)]['converged_thetas_list']
        angles_arr.append(angles)
    rounds = len(angles_arr[0]) // 2
    
    fulltitle = get_full_title(suptitle=suptitle, options=options)
    cmap_beta = cm.get_cmap('autumn')
    cmap_gamma = cm.get_cmap('winter')
    colors = np.linspace(0.05,0.95, rounds)
    colors_beta = [cmap_beta(i) for i in colors]
    colors_gamma = [cmap_gamma(i) for i in colors]
    with plt.style.context(maxcut_style):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        plt.title(fulltitle)
        for i in range(rounds):
            # plot betas
            # Note: Betas go from 0 to pi, while gammas go from 0 to 2pi
            # Hence, plot 2*beta and 1*gamma to cover the entire circle
            betas = [2 * angles_arr[rind][i] for rind in range(num_widths)]
            ax.plot(betas, radii, marker='o', ms=7, ls = 'None', mec = 'k', mew=0.5,alpha=0.7, c=colors_beta[i], label=r'$2\beta_{}$'.format(i+1))
        for i in range(rounds):
            # plot gammas
            gammas = [angles_arr[rind][i+rounds] for rind in range(num_widths)]
            ax.plot(gammas, radii, marker='s', ms=7, ls = 'None', mec = 'k', mew=0.5, alpha=0.7, c=colors_gamma[i], label=r'$\gamma_{}$'.format(i+1))

        ax.set_rmax(maxRadius+1)
        ax.set_rticks(radii)
        ax.set_yticklabels(labels=widths)
        ax.set_xticks(np.pi/2 * np.arange(4))
        ax.set_xticklabels(labels=[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'], fontsize=15)
        ax.set_rlabel_position(0)
        ax.grid(True)
        fig.tight_layout()
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        
        # save plot image to file
        if save_plot_images:
            backend_id = get_backend_id()
            appname = get_appname_from_title(suptitle)
            save_plot_image(plt, f"{appname}-angles-" + suffix, backend_id) 
                
        # show the plot for user to see
        if show_plot_images:
            plt.show()
        

def get_distribution_and_stats():
    """Returns a dictionary with values, colors and labels for various metrics.

    Returns:
        dictionary
    """
    group_metrics_optgaps = {'approx_ratio' : {'color' : 'r', 'label': 'Approx. Ratio', 'gapvals' : [], 'ratiovals':[]},
                            'cvar_ratio' : {'color' : 'g', 'label': 'CVaR Ratio', 'gapvals' : [],'ratiovals':[]},
                            'bestcut_ratio' : {'color' : 'm', 'label': 'Best Measurement Ratio', 'gapvals' : [],'ratiovals':[]},
                            'gibbs_ratio' : {'color' : 'y', 'label' : 'Gibbs Objective Function', 'gapvals' : [],'ratiovals':[]},
                            'quantile_optgaps' : {'gapvals' : [],'ratiovals':[]},
                            'violin' : {'gapvals' : []},# gapvals is a list of [xlist, ylist], 
                            'cutsize_ratio_dist' : {'ratios':[],'frequencies':[]},
                            'random_cutsize_ratio_dist' : {'ratios':[],'frequencies':[]},
                            'groups' : []} #widths

    # circuit_metrics_detail_2.keys() may not be in an ascending order. Sort the groups (i.e. widths)
    groups = list(circuit_metrics_detail_2.keys())
    groups = sorted(groups, key=lambda x: int(x))
    group_metrics_optgaps["groups"] = [int(g) for g in groups]
    
    for group in groups:
        best_restart_ind = get_best_restart_ind(group)
        for circuit_id in [best_restart_ind]:#circuit_metrics_detail_2[group]:
            # save the metric from the last iteration
            last_ind = max(circuit_metrics_detail_2[group][circuit_id].keys())
            mets = circuit_metrics_detail_2[group][circuit_id][last_ind]
            
            # Store the ratio values for objective functions
            group_metrics_optgaps['approx_ratio']['ratiovals'].append(mets["approx_ratio"])
            group_metrics_optgaps['cvar_ratio']['ratiovals'].append(mets["cvar_ratio"])
            group_metrics_optgaps['bestcut_ratio']['ratiovals'].append(mets["bestcut_ratio"])
            group_metrics_optgaps['gibbs_ratio']['ratiovals'].append(mets["gibbs_ratio"])
            
            # Compute optimality gaps for the objective function types
            group_metrics_optgaps['approx_ratio']['gapvals'].append(abs(1.0 - mets["approx_ratio"]) * 100)
            group_metrics_optgaps['cvar_ratio']['gapvals'].append(abs(1.0 - mets["cvar_ratio"]) * 100)
            group_metrics_optgaps['bestcut_ratio']['gapvals'].append(abs(1.0 - mets["bestcut_ratio"]) * 100)
            group_metrics_optgaps['gibbs_ratio']['gapvals'].append(abs(1.0 - mets["gibbs_ratio"]) * 100)
            
            # Also store the optimality gaps at the three quantiles values
            # Here, optgaps are defined as weight(cut)/weight(maxcut) * 100
            q_vals = mets["quantile_optgaps"] # in fraction form. List of floats
            q_vals = [q_vals[i] * 100 for i in range(len(q_vals))] # In percentages
            group_metrics_optgaps['quantile_optgaps']['gapvals'].append(q_vals)
            
            # Store empirical distribution of cut size values / optimal value 
            unique_sizes = circuit_metrics_final_iter[group][str(circuit_id)]['unique_sizes']
            unique_counts = circuit_metrics_final_iter[group][str(circuit_id)]['unique_counts']
            optimal_value = circuit_metrics_final_iter[group][str(circuit_id)]['optimal_value']

            full_size_list = list(range(optimal_value + 1))
            full_counts_list = [unique_counts[unique_sizes.index(s)] if s in unique_sizes else 0 for s in full_size_list]
            group_metrics_optgaps['cutsize_ratio_dist']['ratios'].append(np.array(full_size_list) / optimal_value)
            group_metrics_optgaps['cutsize_ratio_dist']['frequencies'].append(np.array(full_counts_list) / sum(full_counts_list))
            # Also store locations for the half-violin plots to be plotted in the detailed opt-gap plots
            # gap values for the violin plot will be 1 - unique_sizes / optimal size
            violin_yvals = 100 * (1 - np.array(full_size_list) / optimal_value)
            # Normalize the violin plot so that the max width will be 1 unit along horizontal axis
            violin_xvals =  np.array(full_counts_list) / max(full_counts_list)
            group_metrics_optgaps['violin']['gapvals'].append([violin_xvals, violin_yvals])
            
            # Store empirican distribution of cut size values / optimal value for random sampling
            unique_sizes_unif = circuit_metrics_final_iter[group][str(circuit_id)]['unique_sizes_unif']
            unique_counts_unif = circuit_metrics_final_iter[group][str(circuit_id)]['unique_counts_unif']
            full_size_list = list(range(optimal_value + 1))
            full_counts_list_unif = [unique_counts_unif[unique_sizes_unif.index(s)] if s in unique_sizes_unif else 0 for s in full_size_list]
            group_metrics_optgaps['random_cutsize_ratio_dist']['ratios'].append(np.array(full_size_list) / optimal_value)
             
            # hack to avoid crash if all zeros
            sum_full_counts_list_unif = sum(full_counts_list_unif)
            if sum_full_counts_list_unif <= 0: sum_full_counts_list_unif = 1
            
            group_metrics_optgaps['random_cutsize_ratio_dist']['frequencies'].append(np.array(full_counts_list_unif) / sum_full_counts_list_unif)
       
    return group_metrics_optgaps
    
# Plot detailed optgaps
def plot_metrics_optgaps (suptitle="", 
                          transform_qubit_group = False, 
                          new_qubit_group = None, filters=None, 
                          suffix="", objective_func_type = 'cvar_ratio',
                          which_metrics_to_plot = "all",
                          options=None):
    """
    Create and two plots:
        1. Bar plots showing the optimality gap in terms of the approximation ratio vs circuit widths.
            Also plot quartiles on top.
        2. Line plots showing the optimality gaps measured in terms of all available objective function types.
            Also plot quartiles, and violin plots.
    Currently only used for maxcut
    """

    # get backend id for this set of circuits
    backend_id = get_backend_id()
    
    # Extract shorter app name from the title passed in by user   
    appname = get_appname_from_title(suptitle)
        
    if len(group_metrics["groups"]) == 0:
        print(f"\n{suptitle}")
        print("     ****** NO RESULTS ****** ")
        return
    
    # sort the group metrics (in case they weren't sorted when collected)
    sort_group_metrics()
    # DEVNOTE: Add to group metrics here; this should be done during execute
    
    # Create a dictionary, with keys specifying metric type, and values specifying corresponding optgap values
    group_metrics_optgaps = get_distribution_and_stats()

    if which_metrics_to_plot == 'all' or type(which_metrics_to_plot) != list:
        which_metrics_to_plot = ['approx_ratio', 'cvar_ratio', 'bestcut_ratio', 'gibbs_ratio', 'quantile_optgaps', 'violin']
    
    # check if we have sparse or non-linear axis data and linearize if so
    xvalues = group_metrics_optgaps["groups"]
    xlabels = None
    if needs_linearize(xvalues, gap=2):
        #print("... needs linearize")
        
        # convert irregular x-axis data to linear if any non-linear gaps in the data
        xx, xlabels = linearize_axis(xvalues, gap=2, outer=0, fill=False)
        xvalues = xx
    
    # Create title for the plots
    fulltitle = get_full_title(suptitle=suptitle, options=options)

    ############################################################
    ##### Optimality gaps bar plot
    
    with plt.style.context(maxcut_style):
        fig, axs = plt.subplots(1, 1)
        plt.title(fulltitle)
        axs.set_ylabel(r'Optimality Gap ($\%$)')
        #axs.set_xlabel('Circuit Width (Number of Qubits)')
        axs.set_xlabel(known_y_labels['num_qubits'])    # indirection
                
        axs.set_xticks(xvalues)
        if xlabels != None:
            plt.xticks(xvalues, xlabels)
 
        limopts = max(group_metrics_optgaps['approx_ratio']['gapvals'])
        if limopts > 5:
            axs.set_ylim([0, max(40, limopts) * 1.1])
        else:
            axs.set_ylim([0, 5.0])
            
        axs.grid(True, axis = 'y', color='silver', zorder = 0)  # other bars use this silver color
        #axs.grid(True, axis = 'y', zorder = 0)
        axs.bar(xvalues, group_metrics_optgaps['approx_ratio']['gapvals'], 0.8, zorder = 3)
        
        # NOTE: Can move the calculation or the errors variable to before the plotting. This code is repeated in the detailed plotting as well.
        # Plot quartiles
        q_vals = group_metrics_optgaps['quantile_optgaps']['gapvals'] # list of lists; shape (number of circuit widths, 3)
        # Indices are of the form (circuit width index, quantile index)
        center_optgaps = [q_vals[i][1] for i in range(len(q_vals))]
        down_error = [q_vals[i][0] - q_vals[i][1] for i in range(len(q_vals))]
        up_error = [q_vals[i][1] - q_vals[i][2] for i in range(len(q_vals))]
        errors = [up_error, down_error]

        axs.errorbar(xvalues, center_optgaps, yerr = errors, ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5,ls='', marker = "D", markersize = 8, mfc = 'c', mec = 'k', mew = 0.5,label = 'Quartiles', alpha = 0.75, zorder = 5)

        fig.tight_layout()
        axs.legend()

        # save plot image to file
        if save_plot_images:
            save_plot_image(plt, f"{appname}-optgaps-bar" + suffix, backend_id) 
                
        # show the plot for user to see
        if show_plot_images:
            plt.show()


    ############################################################
    ##### Detailed optimality gaps plot
    
    with plt.style.context(maxcut_style):
        fig, axs = plt.subplots(1, 1)
        plt.title(fulltitle)
        axs.set_ylabel(r'Optimality Gap ($\%$)')
        #axs.set_xlabel('Circuit Width (Number of Qubits)')
        axs.set_xlabel(known_y_labels['num_qubits'])    # indirection
        
        axs.set_xticks(xvalues)
        if xlabels != None:
            plt.xticks(xvalues, xlabels)

        if 'violin' in which_metrics_to_plot:
            list_of_violins = group_metrics_optgaps['violin']['gapvals']
            # violinx_list = [x for [x,y] in list_of_violins]
            # violiny_list = [y for [x,y] in list_of_violins]
            violin_list_locs = xvalues
            for loc, vxy in zip(violin_list_locs, list_of_violins):
                vy = vxy[1]
                vx = vxy[0]
                axs.fill_betweenx(vy, loc, loc + vx, color='r', alpha=0.2,lw=0)
        
        # Plot violin plots
        plt_handles = dict()
        
        # Plot the quantile optimality gaps as errorbars
        if 'quantile_optgaps' in which_metrics_to_plot:
            q_vals = group_metrics_optgaps['quantile_optgaps']['gapvals'] # list of lists; shape (number of circuit widths, 3)
            # Indices are of the form (circuit width index, quantile index)
            center_optgaps = [q_vals[i][1] for i in range(len(q_vals))]
            down_error = [q_vals[i][0] - q_vals[i][1] for i in range(len(q_vals))]
            up_error = [q_vals[i][1] - q_vals[i][2] for i in range(len(q_vals))]
            errors = [up_error, down_error]

            plt_handles['quantile_optgaps'] = axs.errorbar(xvalues, center_optgaps, yerr = errors,ecolor = 'k', elinewidth = 1, barsabove = False, capsize=5,ls='', marker = "D", markersize = 8, mfc = 'c', mec = 'k', mew = 0.5,label = 'Quartiles', alpha = 0.75)

        for metric_str in set(which_metrics_to_plot) - set(["quantile_optgaps", "violin"]):
            # For all metrics to be plotted, except quantile optgaps and violin plots, plot a line
            # Plot a solid line for the objective function, and dashed otherwise
            ls = '-' if metric_str == objective_func_type else '--'
            plt_handles[metric_str], = axs.plot(xvalues, group_metrics_optgaps[metric_str]['gapvals'],marker='o', lw=1,ls = ls,color = group_metrics_optgaps[metric_str]['color'],label = group_metrics_optgaps[metric_str]['label'])    


        # Put up the legend, but with labels arranged in the order specified by ideal_lgnd_seq
        ideal_lgnd_seq = ['approx_ratio', 'cvar_ratio', 'gibbs_ratio', 'bestcut_ratio', 'quantile_optgaps']
        handles_list= [plt_handles[s] for s in ideal_lgnd_seq if s in plt_handles]
        axs.legend(handles=handles_list, ncol=2, loc='upper right')# loc='center left', bbox_to_anchor=(1, 0.5)) # For now, we are only plotting for degree 3, and not -3
        
        # Set y limits
        ylim_top = 0
        for o_f in ['approx_ratio', 'cvar_ratio', 'bestcut_ratio', 'gibbs_ratio']:
            ylim_top = max(ylim_top, max(group_metrics_optgaps[o_f]['gapvals']))
        ylim_top = max(ylim_top, max(map(max, group_metrics_optgaps['quantile_optgaps']['gapvals'])))
        if ylim_top > 60: 
            ylim_top = 100 + 3
            bottom = 0 - 3
        elif ylim_top > 10:
            ylim_top = 60 + 3
            bottom = 0 - 3
        else:
            ylim_top = 8 + 1
            bottom = 0 - 1
            
        axs.set_ylim(bottom=bottom, top = ylim_top)
        # axs.set_ylim(bottom=0,top=100)

        # Add grid
        plt.grid()
        fig.tight_layout() 
        
        # save plot image to file
        if save_plot_images:
            save_plot_image(plt, f"{appname}-optgaps-" + suffix, backend_id)
                
        # show the plot for user to see
        if show_plot_images:
            plt.show()


#############################################
# ANALYSIS AND VISUALIZATION - DATA UTILITIES

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

    # if saving raw circuit data, add it too
    #shared_data[app]["circuit_metrics"] = circuit_metrics
    
    # be sure we have a __data directory
    if not os.path.exists('__data'): os.makedirs('__data')
    
    # create filename based on the backend_id and optional data_suffix
    filename = f"__data/DATA-{backend_id}{data_suffix}.json"
    
    # overwrite the existing file with the merged data
    with open(filename, 'w+') as f:
        json.dump(shared_data, f, indent=2, sort_keys=True)
        f.close()
 
# Load the application metrics from the given data file
# Returns a dict containing circuit and group metrics
def load_app_metrics (api, backend_id):

    # don't leave slashes in the filename
    backend_id = backend_id.replace("/", "_")
    
    # create filename based on the backend_id and optional data_suffix
    filename = f"__data/DATA-{backend_id}{data_suffix}.json"
        
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
        #print(app)
        
        # this is very old and could potentially be removed (but would need testing)
        if "group_metrics" not in shared_data[app]:
            print(f"... upgrading version of app data {app}")
            shared_data[app] = { "circuit_metrics":None, "group_metrics":shared_data[app] }

        group_metrics = shared_data[app]["group_metrics"]
        #print(group_metrics)
        
        # need to include avg_hf_fidelities
        if "avg_hf_fidelities" not in group_metrics:
            print(f"... upgrading version of app data {app}")
            #print(f"... upgrading version of app data {app}, adding avg_hf_fidelities")
            group_metrics["avg_hf_fidelities"] = copy.copy(group_metrics["avg_fidelities"])
        
        # need to include avg_tr_n2qs
        if "avg_tr_n2qs" not in group_metrics:
            #print(f"... upgrading version of app data {app}, adding avg_tr_n2qs")
            group_metrics["avg_tr_n2qs"] = copy.copy(group_metrics["avg_tr_depths"])
            for i in range(len(group_metrics["avg_tr_n2qs"])):
                group_metrics["avg_tr_n2qs"][i] *= group_metrics["avg_tr_xis"][i]
                
        #print(group_metrics)
        
    return shared_data
            
            
##############################################
# VOLUMETRIC PLOT

import math
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize

############### Color Map functions
 
# Create a selection of colormaps from which to choose; default to custom_spectral
cmap_spectral = plt.get_cmap('Spectral')
cmap_greys = plt.get_cmap('Greys')
cmap_blues = plt.get_cmap('Blues')
cmap_custom_spectral = None

# the default colormap is the spectral map
cmap = cmap_spectral
cmap_orig = cmap_spectral

# current cmap normalization function (default None)
cmap_norm = None

default_fade_low_fidelity_level = 0.16
default_fade_rate = 0.7

# Specify a normalization function here (default None)
def set_custom_cmap_norm(vmin, vmax):

    global cmap_norm
    
    if vmin == vmax or (vmin == 0.0 and vmax == 1.0):
        print("... setting cmap norm to None")
        cmap_norm = None
    else:
        print(f"... setting cmap norm to [{vmin}, {vmax}]")
        cmap_norm = Normalize(vmin=vmin, vmax=vmax)
    
# Remake the custom spectral colormap with user settings
def set_custom_cmap_style(
            fade_low_fidelity_level=default_fade_low_fidelity_level,
            fade_rate=default_fade_rate):
            
    #print("... set custom map style")
    global cmap, cmap_custom_spectral, cmap_orig
    cmap_custom_spectral = create_custom_spectral_cmap(
                fade_low_fidelity_level=fade_low_fidelity_level, fade_rate=fade_rate)
    cmap = cmap_custom_spectral
    cmap_orig = cmap_custom_spectral
       
# Create the custom spectral colormap from the base spectral
def create_custom_spectral_cmap(
            fade_low_fidelity_level=default_fade_low_fidelity_level,
            fade_rate=default_fade_rate):

    # determine the breakpoint from the fade level
    num_colors = 100
    breakpoint = round(fade_low_fidelity_level * num_colors)
    
    # get color list for spectral map
    spectral_colors = [cmap_spectral(v/num_colors) for v in range(num_colors)]

    #print(fade_rate)
    
    # create a list of colors to replace those below the breakpoint
    # and fill with "faded" color entries (in reverse)
    low_colors = [0] * breakpoint
    #for i in reversed(range(breakpoint)):
    for i in range(breakpoint):
    
        # x is index of low colors, normalized 0 -> 1
        x = i / breakpoint
    
        # get color at this index
        bc = spectral_colors[i]
        r0 = bc[0]
        g0 = bc[1]
        b0 = bc[2]
        z0 = bc[3]
        
        r_delta = 0.92 - r0
        
        #print(f"{x} {bc} {r_delta}")
         
        # compute saturation and greyness ratio
        sat_ratio = 1 - x
        
        #grey_ratio = 1 - x
        '''  attempt at a reflective gradient   
        if i >= breakpoint/2:
            xf = 2*(x - 0.5)
            yf = pow(xf, 1/fade_rate)/2
            grey_ratio = 1 - (yf + 0.5)
        else:
            xf = 2*(0.5 - x)
            yf = pow(xf, 1/fade_rate)/2
            grey_ratio = 1 - (0.5 - yf)
        '''   
        grey_ratio = 1 - math.pow(x, 1/fade_rate)
        
        #print(f"  {xf} {yf} ")
        #print(f"  {sat_ratio} {grey_ratio}")

        r = r0 + r_delta * sat_ratio
        
        g_delta = r - g0
        b_delta = r - b0
        g = g0 + g_delta * grey_ratio
        b = b0 + b_delta * grey_ratio 
            
        #print(f"{r} {g} {b}\n")    
        low_colors[i] = (r,g,b,z0)
        
    #print(low_colors)

    # combine the faded low colors with the regular spectral cmap to make a custom version
    cmap_custom_spectral = ListedColormap(low_colors + spectral_colors[breakpoint:])

    #spectral_colors = [cmap_custom_spectral(v/10) for v in range(10)]
    #for i in range(10): print(spectral_colors[i])
    #print("")
    
    return cmap_custom_spectral

# Make the custom spectral color map the default on module init
set_custom_cmap_style()

# Return the color associated with the spcific value, using color map norm
def get_color(value):
    
    # if there is a normalize function installed, scale the data
    if cmap_norm:
        value = float(cmap_norm(value))
        
    if cmap == cmap_spectral:
        value = 0.05 + value*0.9
    elif cmap == cmap_blues:
        value = 0.00 + value*1.0
    else:
        value = 0.0 + value*0.95
        
    return cmap(value)


############### Helper functions
 
# return the base index for a circuit depth value
# take the log in the depth base, and add 1
def depth_index(d, depth_base):
    if depth_base <= 1:
        return d
    if d == 0:
        return 0
    return math.log(d, depth_base) + 1


# draw a box at x,y with various attributes   
def box_at(x, y, value, type=1, fill=True, x_size=1.0, y_size=1.0, alpha=1.0, zorder=1):
    
    value = min(value, 1.0)
    value = max(value, 0.0)

    fc = get_color(value)
    ec = (0.5,0.5,0.5)
    
    return Rectangle((x - (x_size/2), y - (y_size/2)), x_size, y_size,
             alpha=alpha,
             edgecolor = ec,
             facecolor = fc,
             fill=fill,
             lw=0.5*y_size,
             zorder=zorder)

# draw a circle at x,y with various attributes 
def circle_at(x, y, value, type=1, fill=True):
    size = 1.0
    
    value = min(value, 1.0)
    value = max(value, 0.0)

    fc = get_color(value)
    ec = (0.5,0.5,0.5)
    
    return Circle((x, y), size/2,
             alpha = 0.7,                       # DEVNOTE: changed to 0.7 from 0.5, to handle only one cell
             edgecolor = ec,
             facecolor = fc,
             fill=fill,
             lw=0.5)
             
def box4_at(x, y, value, type=1, fill=True, alpha=1.0):
    size = 1.0
    
    value = min(value, 1.0)
    value = max(value, 0.0)

    fc = get_color(value)
    ec = (0.3,0.3,0.3)
    ec = fc
    
    return Rectangle((x - size/8, y - size/2), size/4, size,
             alpha=alpha,
             edgecolor = ec,
             facecolor = fc,
             fill=fill,
             lw=0.1)

def bkg_box_at(x, y, value=0.9):
    size = 0.6
    return Rectangle((x - size/2, y - size/2), size, size,
             edgecolor = (.75,.75,.75),
             facecolor = (value,value,value),
             fill=True,
             lw=0.5)
             
def bkg_empty_box_at(x, y):
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

# format a number using K,M,B,T for large numbers, optionally rounding to 'digits' decimal places if num > 1
# (sign handling may be incorrect)
def format_number(num, digits=0):
    if isinstance(num, str): num = float(num)
    num = float('{:.3g}'.format(abs(num)))
    sign = ''
    metric = {'T': 1000000000000, 'B': 1000000000, 'M': 1000000, 'K': 1000, '': 1}
    for index in metric:
        num_check = num / metric[index]
        if num_check >= 1:
            num = round(num_check, digits)
            sign = index
            break
    numstr = f"{str(num)}"
    if '.' in numstr:
        numstr = numstr.rstrip('0').rstrip('.')
    return f"{numstr}{sign}"

##### Volumetric Plots

# Plot the background for the volumetric analysis    
def plot_volumetric_background(max_qubits=11, QV=32, depth_base=2, suptitle=None, avail_qubits=0, colorbar_label="Avg Result Fidelity"):
    
    if suptitle == None:
        suptitle = f"Volumetric Positioning\nCircuit Dimensions and Fidelity Overlaid on Quantum Volume = {QV}"

    QV0 = QV
    qv_estimate = False
    est_str = ""
    if QV == 0:                 # QV = 0 indicates "do not draw QV background or label"
        QV = 2048
        
    elif QV < 0:                # QV < 0 indicates "add est. to label"
        QV = -QV
        qv_estimate = True
        est_str = " (est.)"
        
    if avail_qubits > 0 and max_qubits > avail_qubits:
        max_qubits = avail_qubits
        
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
    else:
        ax.add_patch(qv_box_at(1, 1, QV_width, QV_depth, 0.91, depth_base))
    
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
            
            # show vb rectangles; if not showing QV, make all hollow (or less dark)
            if QV0 == 0:
                #ax.add_patch(bkg_empty_box_at(id, w))
                ax.add_patch(bkg_box_at(id, w, 0.95))
            
            else:
                ax.add_patch(bkg_box_at(id, w, 0.9))
            
            # save index of last successful depth
            i_success += 1
        
        # plot empty rectangle after others       
        d = xround[i_success]
        id = depth_index(d, depth_base) 
        ax.add_patch(bkg_empty_box_at(id, w))
        
    
    # Add annotation showing quantum volume
    if QV0 != 0:
        t = ax.text(max_depth_log - 2.0, 1.5, f"QV{est_str}={QV}", size=12,
                horizontalalignment='right', verticalalignment='center', color=(0.2,0.2,0.2),
                bbox=dict(boxstyle="square,pad=0.3", fc=(.9,.9,.9), ec="grey", lw=1))
                
    # add colorbar to right of plot
    plt.colorbar(cm.ScalarMappable(cmap=cmap), cax=None, ax=ax,
            shrink=0.6, label=colorbar_label, panchor=(0.0, 0.7))
            
    return ax


def plot_volumetric_background_aq(max_qubits=11, AQ=12, depth_base=2, suptitle=None, avail_qubits=0, colorbar_label="Avg Result Fidelity"):
    
    if suptitle == None:
        suptitle = f"Volumetric Positioning\nCircuit Dimensions and Fidelity Overlaid on Algorithmic Qubits = {AQ}"

    AQ0 = AQ
    aq_estimate = False
    est_str = ""

    if AQ == 0:
        AQ=12
        
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
    else:
        ax.add_patch(qv_box_at(1, 1, AQ_width, AQ_depth, 0.91, depth_base))
    
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
                #ax.add_patch(bkg_empty_box_at(id, w))
                ax.add_patch(bkg_box_at(id, w, 0.95))
            else:
                ax.add_patch(bkg_box_at(id, w, 0.9))
            
            # save index of last successful depth
            i_success += 1
        
        # plot empty rectangle after others       
        d = xround[i_success]
        id = depth_index(d, depth_base) 
        ax.add_patch(bkg_empty_box_at(id, w))
        
    
    # Add annotation showing quantum volume
    if AQ0 != 0:
        t = ax.text(max_depth_log - 2.0, 1.5, f"AQ{est_str}={AQ}", size=12,
                horizontalalignment='right', verticalalignment='center', color=(0.2,0.2,0.2),
                bbox=dict(boxstyle="square,pad=0.3", fc=(.9,.9,.9), ec="grey", lw=1))
                
    # add colorbar to right of plot
    plt.colorbar(cm.ScalarMappable(cmap=cmap), cax=None, ax=ax,
                shrink=0.6, label=colorbar_label, panchor=(0.0, 0.7))
            
    return ax


# Linear Background Analog of the QV Volumetric Background, to allow arbitrary metrics on each axis
def plot_metrics_background(suptitle, ylabel, x_label, score_label,
            y_max, x_max, y_min=0, x_min=0, ylabels=None):
    
    if suptitle == None:
        suptitle = f"{ylabel} vs. {x_label}, Parameter Positioning of {score_label}"
    
    # plot_width = 6.8
    # plot_height = 5.0

    # assume y max is the max of the y data 
    # we only do circuit width for now, so show 3 qubits more than the max
    max_width = y_max + 3
    min_width = y_min - 3
    
    fig, ax = plt.subplots() #constrained_layout=True, figsize=(plot_width, plot_height))

    plt.title(suptitle)
    
    # DEVNOTE: this code could be made more general, rounding the axis max to 20 divs nicely
    # round the max up to be divisible evenly (in multiples of 0.05 or 0.005) by num_xdivs 
    num_xdivs = 20
    max_base = num_xdivs * 0.05
    if x_max != None and x_max > 1.0:
        x_max = max_base * int((x_max + max_base) / max_base)
    if x_max != None and x_max > 0.1:
        max_base = num_xdivs * 0.005
        x_max = max_base * int((x_max + max_base) / max_base)
    
    #print(f"... {x_min} {x_max} {max_base}")
    if x_min < 0.1: x_min = 0
    
    # and compute the step size for the tick divisions
    step = (x_max - x_min) / num_xdivs   
    plt.xlim(x_min - step/2, x_max + step/2)
       
    #plt.ylim(y_min*0.5, y_max*1.5)
    plt.ylim(min_width, max_width)

    # circuit metrics (x axis)
    xround = [step * x for x in range(num_xdivs + 1)]
    
    # format x labels > 1 to N decimal places, depending on total range
    digits = 0
    if x_max < 24: digits = 1
    if x_max < 10: digits = 2
    xlabels = [format_number(x, digits=digits) for x in xround]
    
    ax.set_xlabel(x_label)
    ax.set_xticks(xround)  
    plt.xticks(xround, xlabels, color='black', rotation=45, ha='right', va='top', rotation_mode="anchor")
    
    # circuit metrics (y axis)
    ybasis = [y for y in range(min_width + 1, max_width)]
    #yround = [(y_max - y_min)/12 * y for y in range(0,25,2)]    # not used now, since we only do circuit width
    #ylabels = [format_number(y) for y in yround]
        
    ax.set_ylabel(ylabel)
    #ax.set_yticks(yround)
    ax.set_yticks(ybasis)   

    if ylabels != None:
        plt.yticks(ybasis, ylabels)
      
    # if score label is accuracy volume, get the cmap colors and invert them
    if score_label == 'Accuracy Volume':
        global cmap
        cmap_colors = [cmap_orig(v/1000) for v in range(1000)]
        cmap_colors.reverse()
        cmap = ListedColormap(cmap_colors)

    else:
        cmap = cmap_orig

    # add colorbar to right of plot (scale if normalize function installed)    
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=cmap_norm), cax=None, ax=ax,
            shrink=0.6, label=score_label, panchor=(0.0, 0.7))
    if score_label == 'Accuracy Volume':
        cbar.ax.invert_yaxis()
        
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
        labelpos=(0.2, 0.7), labelrot=0, type=1, fill=True, w_max=18, do_label=False, do_border=True,
        x_size=1.0, y_size=1.0, zorder=1, offset_flag=False,
        max_depth=0, suppress_low_fidelity=False):

    # since data may come back out of order, save point at max y for annotation
    i_anno = 0
    x_anno = 0 
    y_anno = 0
    
    # plot data rectangles
    low_fidelity_count = True
    
    last_y = -1
    k = 0

    # determine y-axis dimension for one pixel to use for offset of bars that start at 0
    (_, dy) = get_pixel_dims(ax)
    
    # do this loop in reverse to handle the case where earlier cells are overlapped by later cells
    for i in reversed(range(len(d_data))):
        x = depth_index(d_data[i], depth_base)
        y = float(w_data[i])
        f = f_data[i]
        
        # each time we star a new row, reset the offset counter
        # DEVNOTE: this is highly specialized for the QA area plots, where there are 8 bars
        # that represent time starting from 0 secs.  We offset by one pixel each and center the group
        if y != last_y:
            last_y = y
            k = 3              # hardcoded for 8 cells, offset by 3
        
        #print(f"{i = } {x = } {y = }")
        
        if max_depth > 0 and d_data[i] > max_depth:
            #print(f"... excessive depth (2), skipped; w={y} d={d_data[i]}")
            break
            
        # reject cells with low fidelity
        if suppress_low_fidelity and f < suppress_low_fidelity_level:
            if low_fidelity_count: break
            else: low_fidelity_count = True
        
        # the only time this is False is when doing merged gradation plots
        if do_border == True:
        
            # this case is for an array of x_sizes, i.e. each box has different width
            if isinstance(x_size, list):
                
                # draw each of the cells, with no offset
                if not offset_flag:
                    ax.add_patch(box_at(x, y, f, type=type, fill=fill, x_size=x_size[i], y_size=y_size, zorder=zorder))
                    
                # use an offset for y value, AND account for x and width to draw starting at 0
                else:
                    ax.add_patch(box_at((x/2 + x_size[i]/4), y + k*dy, f, type=type, fill=fill, x_size=x+ x_size[i]/2, y_size=y_size, zorder=zorder))
                
            # this case is for only a single cell
            else:
                ax.add_patch(box_at(x, y, f, type=type, fill=fill, x_size=x_size, y_size=y_size))

        # save the annotation point with the largest y value
        if y >= y_anno:
            x_anno = x
            y_anno = y
            i_anno = i
        
        # move the next bar down (if using offset)
        k -= 1
    
    # if no data rectangles plotted, no need for a label
    if x_anno == 0 or y_anno == 0:
        return
        
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
        labelpos=(0.2, 0.7), labelrot=0, type=1, fill=True, w_max=18, do_label=False,
        max_depth=0, suppress_low_fidelity=False):

    # since data may come back out of order, save point at max y for annotation
    i_anno = 0
    x_anno = 0 
    y_anno = 0
    
    # plot data rectangles
    low_fidelity_count = True
    for i in range(len(d_data)):
        x = depth_index(d_data[i], depth_base)
        y = float(w_data[i])
        f = f_data[i]
        
        if max_depth > 0 and d_data[i] > max_depth:
            #print(f"... excessive depth (2), skipped; w={y} d={d_data[i]}")
            break
        
        # reject cells with low fidelity
        if suppress_low_fidelity and f < suppress_low_fidelity_level:
            if low_fidelity_count: break
            else: low_fidelity_count = True
            
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
 
# Return the x and y equivalent to a single pixel for the given plot axis
def get_pixel_dims(ax):

    # transform 0 -> 1 to pixel dimensions
    pixdims = ax.transData.transform([(0,1),(1,0)])-ax.transData.transform((0,0))
    xpix = pixdims[1][0]
    ypix = pixdims[0][1]
    
    #determine x- and y-axis dimension for one pixel 
    dx = (1 / xpix)
    dy = (1 / ypix)
    
    return (dx, dy)
    
    
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
