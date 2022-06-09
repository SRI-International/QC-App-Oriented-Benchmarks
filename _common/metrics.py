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

# Raw and aggregate circuit metrics
circuit_metrics = {  }

circuit_metrics_detail = {  }    # for iterative algorithms
circuit_metrics_detail_2 = {  }  # used to break down to 3rd dimension

group_metrics = { "groups": [],
    "avg_create_times": [], "avg_elapsed_times": [], "avg_exec_times": [], "avg_fidelities": [], "avg_hf_fidelities": [],
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
QV = 32

# Algorithmic Qubits (defaults)
AQ = 22
aq_cutoff = 0.368   # below this circuits not considered successful

aq_mode = 0         # 0 - use default plot behavior, 1 - use AQ modified plots

# average transpile factor between base QV depth and our depth based on results from QV notebook
QV_transpile_factor = 12.7     

# Base for volumetric plot logarithmic axes
#depth_base = 1.66  # this stretches depth axis out, but other values have issues:
#1) need to round to avoid duplicates, and 2) trailing zeros are getting removed 
depth_base = 2

# Get the current time formatted
def get_timestr():
    #timestr = strftime("%Y-%m-%d %H:%M:%S UTC", gmtime())
    timestr = strftime("%b %d, %Y %H:%M:%S UTC", gmtime())
    return timestr

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
    
    # create empty arrays for group metrics
    group_metrics["groups"] = []
    
    group_metrics["avg_create_times"] = []
    group_metrics["avg_elapsed_times"] = []
    group_metrics["avg_exec_times"] = []
    group_metrics["avg_fidelities"] = []
    group_metrics["avg_hf_fidelities"] = []
    
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
    print(f'... execution starting at {get_timestr()}')

# End metrics collection for an application
def end_metrics():
    global end_time

    end_time = time.time()
    total_run_time = round(end_time - start_time, 3)
    print(f'... execution complete at {get_timestr()} in {total_run_time} secs')
    print("")
 
 
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
    
    # if the value is a dict, store each metric provided
    if type(value) is dict:
        for key in value:
            store_metric(group, circuit, key, value[key])


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
        group_hf_fidelity = 0
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
                if metric == "hf_fidelity": group_hf_fidelity += value
                
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
        avg_hf_fidelity = round(group_hf_fidelity / num_circuits, 3)
        
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
        group_metrics["avg_hf_fidelities"].append(avg_hf_fidelity)

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
            avg_hf_fidelity = group_metrics["avg_hf_fidelities"][group_index]
            print(f"Average Fidelity for the {group} qubit group = {avg_fidelity}")
            #if aq_mode > 0:
            #   print(f"Average Hellinger Fidelity for the {group} qubit group = {avg_hf_fidelity}")
            print(f"Average Hellinger Fidelity for the {group} qubit group = {avg_hf_fidelity}")
            
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
    
    print(f"  ==> all detail 2 circuit_metrics:")
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
    # calculate fidelity via hellinger fidelity between correct distribution and our measured expectation values
    hf_fidelity = hellinger_fidelity_with_expected(counts, correct_dist)

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
    fidelity = rescale_fidelity(hf_fidelity, floor_fidelity, new_floor_fidelity)

    return { 'fidelity':fidelity, 'hf_fidelity':hf_fidelity }
    

###############################################
# METRICS UTILITY FUNCTIONS - FOR VISUALIZATION

# get the min and max width over all apps in shared_data
def get_min_max(shared_data):
    w_max = 0
    w_min = 0
    for app in shared_data:
        group_metrics = shared_data[app]["group_metrics"]
        w_data = group_metrics["groups"]
        for i in range(len(w_data)):
            y = float(w_data[i])
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
def get_backend_id():
    subtitle = circuit_metrics["subtitle"]
    backend_id = subtitle[9:]
    return backend_id
 
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
        print(f"     ****** NO RESULTS ****** ")
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
    
    # generate one-column figure with multiple bar charts, with shared X axis
    cols = 1
    fig_w = 6.0
    
    numplots = 0
    if do_creates: numplots += 1
    if do_executes: numplots += 1
    if do_fidelities: numplots += 1
    if do_hf_fidelities: numplots += 1
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
    fulltitle = suptitle + f"\nDevice={backend_id}  {get_timestr()}"
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
    
    if do_hf_fidelities:
        axs[axi].set_ylim([0, 1.0])
        axs[axi].bar(group_metrics["groups"], group_metrics["avg_hf_fidelities"]) 
        axs[axi].set_ylabel('Avg Hellinger Fidelity')
        
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
    
    if do_2qs:
        if max(group_metrics["avg_tr_n2qs"]) < 20:
            axs[axi].set_ylim([0, 20])  
        axs[axi].bar(group_metrics["groups"], group_metrics["avg_tr_n2qs"], 0.5, color='C9') 
        axs[axi].set_ylabel('2Q Gates')
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        axs[axi].legend(['Transpiled 2Q Gates'], loc='upper left')
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
        
    suptitle = f"Volumetric Positioning - {appname}"
    
    # append key circuit metrics info to the title
    fulltitle = suptitle + f"\nDevice={backend_id}  {get_timestr()}"
    if options != None:
        options_str = ''
        for key, value in options.items():
            if len(options_str) > 0: options_str += ', '
            options_str += f"{key}={value}"
        fulltitle += f"\n{options_str}"
    
    global cmap   
    
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

            cmap = cmap_spectral

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
            print(f'ERROR: plot_metrics(), failure when creating volumetric positioning chart')
            print(f"... exception = {e}")
            if verbose:
                print(traceback.format_exc())
        
        # save plot image to file
        if save_plot_images:
            save_plot_image(plt, f"{appname}-vplot", backend_id) 
        
        #display plot
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

            cmap = cmap_spectral

            # If using mid-circuit transformation, convert width data to singular circuit width value
            if transform_qubit_group:
                w_data = new_qubit_group
                group_metrics["groups"] = w_data

            plot_volumetric_data_aq(ax, w_data, d_tr_data, f_data, depth_base, fill=True,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)  
            
            anno_volumetric_data(ax, depth_base,
                label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False)
        
        except Exception as e:
            print(f'ERROR: plot_metrics(), failure when creating volumetric positioning chart')
            print(f"... exception = {e}")
            if verbose:
                print(traceback.format_exc())
        
        # save plot image to file
        if save_plot_images:
            save_plot_image(plt, f"{appname}-vplot-hf", backend_id) 
        
        #display plot
        plt.show()       


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
        print(f'ERROR: plot_metrics_all_overlaid(), failure when creating volumetric positioning chart')
        print(f"... exception = {e}")
        if verbose:
            print(traceback.format_exc())
    
    # save plot image file
    if save_plot_images:
        if aq_mode > 0: imagename += '-aq'
        save_plot_image(plt, imagename, backend_id) 
    
    #display plot
    plt.show()    


#################################################

# Plot metrics over all groups (level 2), merging data from all apps into smaller cells if not is_individual
def plot_metrics_all_merged (shared_data, backend_id, suptitle=None, imagename="_ALL-vplot-2", avail_qubits=0, is_individual=True, score_metric=None):                   
    
    global circuit_metrics
    global group_metrics
    
    # determine the metric to use for scoring, i.e. the color of plot items
    if score_metric == None:
        if aq_mode > 0:
            score_metric = "avg_hf_fidelities"
        else:
            score_metric = "avg_fidelities"
    
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
        w_min, w_max = get_min_max(shared_data)
        
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
            plot_merged_result_rectangles(shared_data, ax, max_qubits, w_max, score_metric=score_metric)
        
        # Now overlay depth metrics for each app with unfilled rects, to outline each circuit
        # if is_individual, do filled rects as there is no background fill
        
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
                   
            if "avg_tr_n2qs" not in group_metrics: continue
            n2q_tr_data = group_metrics["avg_tr_n2qs"]
    
            filled = is_individual
            
            if aq_mode > 0:
                if score_metric not in group_metrics: continue
                f_data = group_metrics[score_metric]
                plot_volumetric_data_aq(ax, w_data, n2q_tr_data, f_data, depth_base, fill=filled,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)
            else:
                f_data = group_metrics[score_metric]
                plot_volumetric_data(ax, w_data, d_tr_data, f_data, depth_base, fill=filled,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, w_max=w_max)
        
        # do annotation separately, spreading labels for readability
        anno_volumetric_data(ax, depth_base,
                   label=appname, labelpos=(0.4, 0.6), labelrot=15, type=1, fill=False)
    
    except Exception as e:
        print(f'ERROR: plot_metrics_all_merged(), failure when creating volumetric positioning chart')
        print(f"... exception = {e}")
        if verbose:
            print(traceback.format_exc())
    
    # save plot image file
    if save_plot_images:
        if aq_mode > 0: imagename += '-aq'
        save_plot_image(plt, imagename, backend_id)

    #display plot
    plt.show()


# Plot filled but borderless rectangles based on merged gradations of result metrics
def plot_merged_result_rectangles(shared_data, ax, max_qubits, w_max, num_grads=4, score_metric=None):

    depth_values_merged = []
    for w in range(max_qubits):
        depth_values_merged.append([ None ] * (num_grads * max_depth_log))
    
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
    

#################################################

### plot metrics across all apps for a backend_id

def plot_all_app_metrics(backend_id, do_all_plots=False,
        include_apps=None, exclude_apps=None, suffix="", avail_qubits=0, is_individual=True, score_metric=None):

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
        cmap = cmap_spectral
        suptitle = f"Volumetric Positioning - All Applications (Combined)\nDevice={backend_id}  {get_timestr()}"
        plot_metrics_all_overlaid(shared_data, backend_id, suptitle=suptitle, imagename="_ALL-vplot-2")
        
        '''

        # draw the volumetric plot and append the circuit metrics subtitle to the title
        suptitle = f"Volumetric Positioning - All Applications (Merged)"
        fulltitle = suptitle + f"\nDevice={backend_id}  {get_timestr()}"
        
        # use a spectral colormap
        cmap = cmap_spectral
        plot_metrics_all_merged(shared_data, backend_id, suptitle=fulltitle, imagename="_ALL-vplot-2"+suffix, avail_qubits=avail_qubits, is_individual=is_individual, score_metric=score_metric)
        
        # also draw with a blues colormap (not now actually)
        '''
        cmap = cmap_blues
        plot_metrics_all_merged(shared_data, backend_id, suptitle=fulltitle, imagename="_ALL-vplot-2b"+suffix, avail_qubits=avail_qubits)  
        '''
        
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
        
    app = "Benchmark Results - " + appname + " - " + apiname
    
    group_metrics = shared_data[app]["group_metrics"]
    plot_metrics(app, filters=filters, suffix=suffix)

# save plot as image
def save_plot_image(plt, imagename, backend_id):

    # don't leave slashes in the filename
    backend_id = backend_id.replace("/", "_")
     
    # not used currently
    date_of_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists('__images'): os.makedirs('__images')
    if not os.path.exists(f'__images/{backend_id}'): os.makedirs(f'__images/{backend_id}')
    
    pngfilename = f"{backend_id}/{imagename}"
    pngfilepath = os.path.join(os.getcwd(),"__images", pngfilename + ".jpg")
    
    plt.savefig(pngfilepath)
    
    #print(f"... saving (plot) image file:{pngfilename}.jpg")   
    
    pdffilepath = os.path.join(os.getcwd(),"__images", pngfilename + ".pdf")
    
    plt.savefig(pdffilepath)
    
    
#################################################
# ANALYSIS AND VISUALIZATION - AREA METRICS PLOTS

# map known X metrics to labels    
known_x_labels = {
    'cumulative_create_time' : 'Cumulative Circuit Creation Time',
    'cumulative_exec_time' : 'Cumulative Quantum Execution Time',
    'cumulative_opt_exec_time' : 'Cumulative Classical Optimizer Time',
    'cumulative_depth' : 'Cumulative Circuit Depth'
}
# map known Y metrics to labels    
known_y_labels = {
    'num_qubits' : 'Circuit Width'
}
# map known Score metrics to labels    
known_score_labels = {
    'approx_ratio' : 'Avg Approximation Ratio',
    'max_approx_ratio' : 'Max Approximation Ratio',
    'fidelity' : 'Avg Result Fidelity',
    'max_fidelity' : 'Max Result Fidelity'
}

 
# Plot all the given "Score Metrics" against the given "X Metrics" and "Y Metrics" 
def plot_all_area_metrics(suptitle=None, score_metric='fidelity', x_metric='exec_time', y_metric='num_qubits', average_over_x_axis=True, fixed_metrics={}, num_x_bins=100, y_size=None, x_size=None, options=None):

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
                plot_area_metrics(suptitle, s_m, x_m, y_m, average_over_x_axis, fixed_metrics, num_x_bins, y_size, x_size, options=options)
       
# Plot the given "Score Metric" against the given "X Metric" and "Y Metric"         
def plot_area_metrics(suptitle=None, score_metric='fidelity', x_metric='cumulative_exec_time', y_metric='num_qubits', average_over_x_axis=True, fixed_metrics={}, num_x_bins=100, y_size=None, x_size=None, options=None):
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
        
        # Each problem instance at size num_qubits; need to collate across iterations
        i = 0
        for circuit_id in circuit_metrics_detail_2[group]:
                
            x_last, score_last = 0, 0
            x_sizes, x_points, y_points, score_points = [], [], [], []            
            
            for it in circuit_metrics_detail_2[group][circuit_id]:
                mets = circuit_metrics_detail_2[group][circuit_id][it]

                # get each metric and accumulate if indicated
                x_raw = x_now = mets[x_metric]
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
    fulltitle = suptitle + f"\nDevice={backend_id}  {get_timestr()}"
    if options != None:
        options_str = ''
        for key, value in options.items():
            if len(options_str) > 0: options_str += ', '
            options_str += f"{key}={value}"
        fulltitle += f"\n{options_str}"
    
    # plot the metrics background with its title
    ax = plot_metrics_background(fulltitle, y_label, x_label, score_label,
                y_max=max(y), x_max=max(x), y_min=min(y), x_min=min(x))
                                 
    # no longer used, instead we pass the array of sizes
    #if x_size == None:
        #x_size=(max(x)-min(x))/num_x_bins
        
    if y_size == None:
        y_size = 1.0
    
    #print(f"... num: {num_x_bins} {len(x)} {x_size} {x}")
    
    # plot all the bars, with width specified as an array that matches the array size of the x,y values
    plot_volumetric_data(ax, y, x, scores, depth_base=-1, label='Depth', labelpos=(0.2, 0.7), 
                        labelrot=0, type=1, fill=True, w_max=18, do_label=False,
                        x_size=xs, y_size=y_size)                         
        

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
    

# Plot bar charts for each metric over all groups
def plot_metrics_optgaps (suptitle="Circuit Width (Number of Qubits)", transform_qubit_group = False, new_qubit_group = None, filters=None, suffix="", options=None):
    
    # get backend id for this set of circuits
    backend_id = get_backend_id()
    
    # Extract shorter app name from the title passed in by user   
    appname = get_appname_from_title(suptitle)
        
    if len(group_metrics["groups"]) == 0:
        print(f"\n{suptitle}")
        print(f"     ****** NO RESULTS ****** ")
        return
    
    # sort the group metrics (in case they weren't sorted when collected)
    sort_group_metrics()
    
    # flags for charts to show
    do_depths = True
    
    # check if we have depth metrics to show
    do_depths = len(group_metrics["avg_depths"]) > 0
    
    # DEVNOTE: Add to group metrics here; this should be done during execute
    group_metrics_2 = {}
    group_metrics_2['approx_ratio'] = []
    group_metrics_2['optimality_gap'] = []
    for group in circuit_metrics_detail_2:
        num_qubits = int(group)
        
        # Each problem instance at size num_qubits; need to collate across iterations
        i = 0
        for circuit_id in circuit_metrics_detail_2[group]:
            
        
            for it in circuit_metrics_detail_2[group][circuit_id]:
                mets = circuit_metrics_detail_2[group][circuit_id][it]
                
            # save the metric from the last iteration
            group_metrics_2['approx_ratio'].append(mets['approx_ratio'])
            group_metrics_2['optimality_gap'].append(1.0 - mets['approx_ratio'])
                
            # and just break after the first circuit, since we aare not averaging
            break
            
    #print(f"... group_metrics_2['approx_ratio'] = {group_metrics_2['approx_ratio']}")
    #print(f"... group_metrics_2['optimality_gap'] = {group_metrics_2['optimality_gap']}")       
    
    # generate one-column figure with multiple bar charts, with shared X axis
    cols = 1
    fig_w = 6.0
    
    numplots = 1
  
    rows = numplots
    
    # DEVNOTE: this calculation is based on visual assessment of results and could be refined
    # compute height needed to draw same height plots, no matter how many there are
    fig_h = 3.5 + 2.0 * (rows - 1) + 0.25 * (rows - 1)
    #print(fig_h)
    
    # create the figure into which plots will be placed
    fig, axs = plt.subplots(rows, cols, sharex=True, figsize=(fig_w, fig_h))
    
    # Create more appropriate title
    suptitle = "Optimality Gaps - " + appname
    
    # append key circuit metrics info to the title
    fulltitle = suptitle + f"\nDevice={backend_id}  {get_timestr()}"
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
        
    if do_depths:
        if max(group_metrics["avg_tr_depths"]) < 20:
            axs[axi].set_ylim([0, 20])  
        axs[axi].bar(group_metrics["groups"], group_metrics_2["optimality_gap"], 0.8)
        #axs[axi].bar(group_metrics["groups"], group_metrics["avg_tr_depths"], 0.5, color='C9') 
        #axs[axi].set_ylabel(known_score_labels['approx_ratio'])
        axs[axi].set_ylabel('Optimality Gap (%)')
        
        if rows > 0 and not xaxis_set:
            axs[axi].sharex(axs[rows-1])
            xaxis_set = True
            
        axs[axi].legend(['Degree 3', 'Degree -3'], loc='upper left')
        axi += 1
    
    # shared x axis label
    axs[rows - 1].set_xlabel('Circuit Width (Number of Qubits)')
     
    fig.tight_layout() 
    
    # save plot image to file
    if save_plot_images:
        save_plot_image(plt, f"{appname}-optgaps" + suffix, backend_id) 
            
    # show the plot for user to see
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
    if depth_base <= 1:
        return d
    if d == 0:
        return 0
    return math.log(d, depth_base) + 1


# draw a box at x,y with various attributes   
def box_at(x, y, value, type=1, fill=True, x_size=1.0, y_size=1.0):
    
    value = min(value, 1.0)
    value = max(value, 0.0)

    fc = get_color(value)
    ec = (0.5,0.5,0.5)
    
    return Rectangle((x - (x_size/2), y - (y_size/2)), x_size, y_size,
             edgecolor = ec,
             facecolor = fc,
             fill=fill,
             lw=0.5*y_size)

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
    plt.colorbar(cm.ScalarMappable(cmap=cmap), shrink=0.6, label=colorbar_label, panchor=(0.0, 0.7))
            
    return ax


def plot_volumetric_background_aq(max_qubits=11, AQ=22, depth_base=2, suptitle=None, avail_qubits=0, colorbar_label="Avg Result Fidelity"):
    
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
    plt.colorbar(cm.ScalarMappable(cmap=cmap), shrink=0.6, label=colorbar_label, panchor=(0.0, 0.7))
            
    return ax


# Linear Background Analog of the QV Volumetric Background, to allow arbitrary metrics on each axis
def plot_metrics_background(suptitle, ylabel, x_label, score_label, y_max, x_max, y_min=0, x_min=0):
    
    if suptitle == None:
        suptitle = f"{ylabel} vs. {x_label}, Parameter Positioning of {score_label}"
    
    plot_width = 6.8
    plot_height = 5.0
    #print(f"... {plot_width} {plot_height}")
    
    # assume y max is the max of the y data 
    # we only do circuit width for now, so show 3 qubits more than the max
    max_width = y_max + 3
    
    # define matplotlib figure and axis; use constrained layout to fit colorbar to right
    fig, ax = plt.subplots(figsize=(plot_width, plot_height), constrained_layout=True)

    plt.suptitle(suptitle)
    
    # round the max up to be divisible evenly (in multiples of 0.1) by num_xdivs 
    num_xdivs = 20
    max_base = num_xdivs * 0.1
    x_max = max_base * int((x_max + max_base) / max_base)
    
    #print(f"... {x_min} {x_max} {max_base} {x_max}")
    if x_min < 0.1: x_min = 0
    
    step = (x_max - x_min) / num_xdivs
    
    plt.xlim(x_min - step/2, x_max + step/2)
       
    #plt.ylim(y_min*0.5, y_max*1.5)
    plt.ylim(0, max_width)

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
    
    # other label options
    #plt.xticks(xbasis, xlabels, color='black', rotation=-60, ha='left')
    #plt.xticks(xbasis, xlabels, color='black', rotation=-45, ha='left', va='center', rotation_mode="anchor")

    # circuit metrics (y axis)
    ybasis = [y for y in range(1, max_width)]
    #yround = [(y_max - y_min)/12 * y for y in range(0,25,2)]    # not used now, since we only do circuit width
    #ylabels = [format_number(y) for y in yround]
        
    ax.set_ylabel(ylabel)
    #ax.set_yticks(yround)
    ax.set_yticks(ybasis)    
    
    # add colorbar to right of plot
    plt.colorbar(cm.ScalarMappable(cmap=cmap), shrink=0.6, label=score_label, panchor=(0.0, 0.7))
    
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
        labelpos=(0.2, 0.7), labelrot=0, type=1, fill=True, w_max=18, do_label=False,
        x_size=1.0, y_size=1.0):

    # since data may come back out of order, save point at max y for annotation
    i_anno = 0
    x_anno = 0 
    y_anno = 0
    
    # plot data rectangles
    for i in range(len(d_data)):
        x = depth_index(d_data[i], depth_base)
        y = float(w_data[i])
        f = f_data[i]
        
        if isinstance(x_size, list):
            ax.add_patch(box_at(x, y, f, type=type, fill=fill, x_size=x_size[i], y_size=y_size))
        else:
            ax.add_patch(box_at(x, y, f, type=type, fill=fill, x_size=x_size, y_size=y_size))

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
