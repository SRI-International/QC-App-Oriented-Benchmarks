import maxcut_benchmark
import metrics
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

maxcut_benchmark.verbose = False
maxcut_style = os.path.join('..','..','_common','maxcut.mplstyle')
#%% Functions for analyzing the effects of initial conditions
def plot_effects_of_initial_conditions():
    num_qubits = list(metrics.circuit_metrics_detail_2.keys())[0]
    restart_keys = list(metrics.circuit_metrics_detail_2[num_qubits].keys())
    
    starting_angles_list = [0] * len(restart_keys)
    final_angles_list = [0] * len(restart_keys)
    final_AR_list = [0] * len(restart_keys)
    for ind, restart_key in enumerate(restart_keys):
        thisRestart = metrics.circuit_metrics_detail_2[num_qubits][restart_key]
        minimizer_keys = list(thisRestart.keys())
        starting_angles_list[ind] = thisRestart[0]['thetas_array']
        final_angles_list[ind] = thisRestart[max(minimizer_keys)]['thetas_array']
        final_AR_list[ind] = thisRestart[max(minimizer_keys)]['approx_ratio']
    
    worst_ind = final_AR_list.index(min(final_AR_list))
    best_ind = final_AR_list.index(max(final_AR_list))

    worst_restart_key = restart_keys[worst_ind]
    best_restart_key = restart_keys[best_ind]

    worst_dict = metrics.circuit_metrics_final_iter[str(num_qubits)][str(worst_restart_key)]
    best_dict = metrics.circuit_metrics_final_iter[str(num_qubits)][str(best_restart_key)]
    
    plot_worst_best_init_conditions(worst_dict, best_dict)
    plot_AR_histogram(final_AR_list)
    
def plot_AR_histogram(final_AR_list):
    
    with plt.style.context(maxcut_style):
        fig, axs = plt.subplots(1, 1)

        # Create more appropriate title
        suptitle = "Histogram of Approximation Ratios"
        options = get_options_effect_init()
        fulltitle = get_title(suptitle, options)
            
        # and add the title to the plot
        plt.suptitle(fulltitle)
        
        ###########################################################
        n, bins, patches = axs.hist(x=final_AR_list, density=False, bins=25, color = 'k', rwidth=0.85, alpha = 0.8)
        ###########################################################
        
        axs.set_ylabel('Counts')
        axs.set_xlabel(r'Approximation Ratio')
        # axs.grid()
        axs.grid(axis='y')
        axs.set_xlim(xmax = 1)
        axs.set_yticks(list(range(0, 5 * (1 + int(n.max()) // 5),5)))

        # axs.legend()
        fig.tight_layout()
        
        # figName = os.path.join(figLoc, 'histogram_of_ARs')
        # plt.savefig(figName + '.pdf')
        # plt.savefig(figName + '.png')    

def get_title(suptitle, options):
    # append key circuit metrics info to the title
    maxcut_inputs = maxcut_benchmark.maxcut_inputs
    backend_id = maxcut_inputs.get('backend_id') 
    fulltitle = suptitle + f"\nDevice={backend_id}  {metrics.get_timestr()}"
    if options != None:
        options_str = ''
        for key, value in options.items():
            if len(options_str) > 0: options_str += ', '
            options_str += f"{key}={value}"
        fulltitle += f"\n{options_str}"
    return fulltitle

    
def get_options_effect_init():
    maxcut_inputs = metrics.maxcut_inputs
    num_shots = maxcut_inputs.get('num_shots')
    width = maxcut_inputs.get('max_qubits')
    degree = maxcut_inputs.get('degree')
    restarts = maxcut_inputs.get('max_circuits')
    options = dict(shots=num_shots, width=width, degree=degree, restarts=restarts)
    return options


def plot_worst_best_init_conditions(worst_dict, best_dict):
    
    
    # Get 2 colors for QAOA, and one for uniform sampling
    cmap = cm.get_cmap('RdYlGn')
    clr_random = 'pink'
    # colors = [cmap(0.1), cmap(0.4), cmap(0.8)]
    colors = np.linspace(0.05,0.95,2, endpoint=True)
    colors = [cmap(i) for i in colors]
    
    #### Good vs Bad Initial Conditions
    with plt.style.context(maxcut_style):
        fig, axs = plt.subplots(1, 1)

        # Create more appropriate title
        suptitle = "Empirical Distribution of cut sizes"
        options = get_options_effect_init()
        fulltitle = get_title(suptitle, options)            
        # and add the title to the plot
        plt.suptitle(fulltitle)
        
        ###########################################################
        # Plot the distribution obtained from uniform random sampling
        unique_counts_unif = worst_dict['unique_counts_unif']
        unique_sizes_unif = worst_dict['unique_sizes_unif']
        optimal_value = worst_dict['optimal_value']
        color = clr_random
        uniform_approx_ratio = - maxcut_benchmark.compute_sample_mean(worst_dict['unique_counts_unif'], worst_dict['unique_sizes_unif']) / optimal_value
        axs.plot(np.array(unique_sizes_unif) / optimal_value, np.array(unique_counts_unif) / sum(unique_counts_unif),
                marker='o', ms=1, mec = 'k',mew=0.2, lw=10,alpha=0.5,
                ls = '-', label = "Random Sampling", c = color)#" degree={deg}") # lw=1, , c = colors[1]
        axs.axvline(x = uniform_approx_ratio, color = color, alpha = 0.5)
        ###########################################################

        ###########################################################
        # Plot best distribution
        unique_counts = best_dict['unique_counts']
        unique_sizes = best_dict['unique_sizes']
        optimal_value = best_dict['optimal_value']
        fin_appr_ratio = - maxcut_benchmark.compute_sample_mean(best_dict['unique_counts'], best_dict['unique_sizes']) / optimal_value
        color = colors[1]
        axs.plot(np.array(unique_sizes) / optimal_value, np.array(unique_counts) / sum(unique_counts), marker='o',
                ls = '-', ms=4, mec = 'k', mew=0.2, lw=2,
                label = "Initial Condition 1", c = color)#" degree={deg}") # lw=1, , 
        axs.axvline(x = fin_appr_ratio, color = color, alpha = 0.5) # label = 'Approx Ratio (Best Run)'
        # thet = best['converged_thetas_list']
        # thet = ['{:.2f}'.format(val) for val in thet]
        # thet=','.join(str(x) for x in thet)
        # axs.text(fin_appr_ratio, 1, "{:.2f}".format(fin_appr_ratio) ,fontsize = 10,c=color,rotation=75)
        ###########################################################

        ###########################################################
        # Plot the worst distribution
        unique_counts = worst_dict['unique_counts']
        unique_sizes = worst_dict['unique_sizes']
        optimal_value = worst_dict['optimal_value']
        fin_appr_ratio=- maxcut_benchmark.compute_sample_mean(worst_dict['unique_counts'], worst_dict['unique_sizes']) / optimal_value
        color = colors[0]
        axs.plot(np.array(unique_sizes) / optimal_value, np.array(unique_counts) / sum(unique_counts), marker='o',
                ls = '-', ms=4, mec = 'k', mew=0.2, lw=2,
                label = "Initial Condition 2", c = color)#" degree={deg}") # lw=1, , c = colors[1]
        axs.axvline(x = fin_appr_ratio, color = color, alpha = 0.5)#, label = 'Approx Ratio (Worst Run)')
        ###########################################################
                
        
                
        
        axs.set_ylabel('Fraction of Total Counts')
        axs.set_xlabel(r'$\frac{\mathrm{Cut\ Size}}{\mathrm{Max\ Cut\ Size}}$')
        axs.grid()

        # axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs.legend()
        fig.tight_layout()
        
        # figName = os.path.join(figLoc, 'good_vs_bad')
        # plt.savefig(figName + '.pdf')
        # plt.savefig(figName + '.png')
        
    
