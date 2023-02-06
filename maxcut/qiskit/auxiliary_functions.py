import maxcut_benchmark
import metrics
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import json 

maxcut_benchmark.verbose = False
maxcut_style = os.path.join('..', '..', '_common', 'maxcut.mplstyle')
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
        final_angles_list[ind] = thisRestart[max(
            minimizer_keys)]['thetas_array']
        final_AR_list[ind] = thisRestart[max(minimizer_keys)]['approx_ratio']

    worst_ind = final_AR_list.index(min(final_AR_list))
    best_ind = final_AR_list.index(max(final_AR_list))

    worst_restart_key = restart_keys[worst_ind]
    best_restart_key = restart_keys[best_ind]

    worst_dict = metrics.circuit_metrics_final_iter[str(
        num_qubits)][str(worst_restart_key)]
    best_dict = metrics.circuit_metrics_final_iter[str(
        num_qubits)][str(best_restart_key)]

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
        n, bins, patches = axs.hist(
            x=final_AR_list, density=False, bins=25, color='k', rwidth=0.85, alpha=0.8)
        ###########################################################

        axs.set_ylabel('Counts')
        axs.set_xlabel(r'Approximation Ratio')
        # axs.grid()
        axs.grid(axis='y')
        axs.set_xlim(xmax=1)
        axs.set_yticks(list(range(0, 5 * (1 + int(n.max()) // 5), 5)))

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
            if len(options_str) > 0:
                options_str += ', '
            options_str += f"{key}={value}"
        fulltitle += f"\n{options_str}"
    return fulltitle


def get_options_effect_init():
    maxcut_inputs = maxcut_benchmark.maxcut_inputs
    num_shots = maxcut_inputs.get('num_shots')
    width = maxcut_inputs.get('max_qubits')
    degree = maxcut_inputs.get('degree')
    restarts = maxcut_inputs.get('max_circuits')
    options = dict(shots=num_shots, width=width,
                   degree=degree, restarts=restarts)
    return options


def plot_worst_best_init_conditions(worst_dict, best_dict):

    # Get 2 colors for QAOA, and one for uniform sampling
    cmap = cm.get_cmap('RdYlGn')
    clr_random = 'pink'
    # colors = [cmap(0.1), cmap(0.4), cmap(0.8)]
    colors = np.linspace(0.05, 0.95, 2, endpoint=True)
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
        uniform_approx_ratio = - maxcut_benchmark.compute_sample_mean(
            worst_dict['unique_counts_unif'], worst_dict['unique_sizes_unif']) / optimal_value
        axs.plot(np.array(unique_sizes_unif) / optimal_value, np.array(unique_counts_unif) / sum(unique_counts_unif),
                 marker='o', ms=1, mec='k', mew=0.2, lw=10, alpha=0.5,
                 ls='-', label="Random Sampling", c=color)  # " degree={deg}") # lw=1, , c = colors[1]
        axs.axvline(x=uniform_approx_ratio, color=color, alpha=0.5)
        ###########################################################

        ###########################################################
        # Plot best distribution
        unique_counts = best_dict['unique_counts']
        unique_sizes = best_dict['unique_sizes']
        optimal_value = best_dict['optimal_value']
        fin_appr_ratio = - maxcut_benchmark.compute_sample_mean(
            best_dict['unique_counts'], best_dict['unique_sizes']) / optimal_value
        color = colors[1]
        axs.plot(np.array(unique_sizes) / optimal_value, np.array(unique_counts) / sum(unique_counts), marker='o',
                 ls='-', ms=4, mec='k', mew=0.2, lw=2,
                 label="Initial Condition 1", c=color)  # " degree={deg}") # lw=1, ,
        # label = 'Approx Ratio (Best Run)'
        axs.axvline(x=fin_appr_ratio, color=color, alpha=0.5)
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
        fin_appr_ratio = - maxcut_benchmark.compute_sample_mean(
            worst_dict['unique_counts'], worst_dict['unique_sizes']) / optimal_value
        color = colors[0]
        axs.plot(np.array(unique_sizes) / optimal_value, np.array(unique_counts) / sum(unique_counts), marker='o',
                 ls='-', ms=4, mec='k', mew=0.2, lw=2,
                 label="Initial Condition 2", c=color)  # " degree={deg}") # lw=1, , c = colors[1]
        # , label = 'Approx Ratio (Worst Run)')
        axs.axvline(x=fin_appr_ratio, color=color, alpha=0.5)
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

    #%% Radar plot


def radar_plot(min_qubits=4, max_qubits=6, num_shots=1000, restarts=10, objective_func_type='approx_ratio',
               rounds=1, degree=3, backend_id='qasm_simulator', provider_backend=None,
               hub="ibm-q", group="open", project="main", exec_options=None,
               ):

    # Implement the runs
    maxcut_benchmark.run(
        min_qubits=min_qubits, max_qubits=max_qubits, max_circuits=restarts, num_shots=num_shots,
        method=2, rounds=rounds, degree=degree, alpha=0.1, N=10, parameterized=False,
        num_x_bins=15, max_iter=30,
        backend_id=backend_id, provider_backend=provider_backend,
        hub=hub, group=group, project=project, exec_options=exec_options,
        objective_func_type=objective_func_type, do_fidelities=False,
        save_res_to_file=False, save_final_counts=False, plot_results=False,
        detailed_save_names=False
    )  # thetas_array = thetas_array

    # Get the inputs as a dictionary
    gen_prop = maxcut_benchmark.maxcut_inputs
    gen_prop['widths'] = list(metrics.group_metrics['groups'])
    # Data for plotting
    radar_plot_data = rearrange_radar_data_for_plotting(gen_prop)
    plot_from_data(gen_prop, radar_plot_data)
    
    
def load_fixed_angles():
    fixed_angle_file = os.path.join('..', '_common', 'angles_regular_graphs.json')

    with open(fixed_angle_file, 'r') as json_file:
        # 'thetas_array', 'approx_ratio_list', 'num_qubits_list'
        fixed_angle_data = json.load(json_file)
        
    return fixed_angle_data

def get_radar_full_title(gen_prop):
    objective_func_type = gen_prop.get('objective_func_type')
    num_shots = gen_prop.get('num_shots')
    rounds = gen_prop.get('rounds')
    method = gen_prop.get('method')
    suptitle = f"Benchmark Results - MaxCut ({method}) - Qiskit"
    obj_str = metrics.known_score_labels[objective_func_type]
    options = {'rounds': rounds, 'Objective Function': obj_str, 'num_shots' : num_shots}
    fulltitle = get_title(suptitle=suptitle, options=options)
    return fulltitle

def plot_from_data(gen_prop, radar_plot_data):
    
    fulltitle = get_radar_full_title(gen_prop)
    rounds = gen_prop.get('rounds')
    degree = gen_prop.get('degree')
    widths = gen_prop.get('widths')
    
    # load fixed angle data
    fixed_angle_data = load_fixed_angles()[str(degree)][str(rounds)]
    ar_fixed = fixed_angle_data['AR']
    thetas_fixed = fixed_angle_data['beta'] + fixed_angle_data['gamma']        

    arranged_thetas_arr = radar_plot_data.get('arranged_thetas_arr')
    arranged_AR_arr = radar_plot_data.get('arranged_AR_arr')
    radii_arr = radar_plot_data.get('radii_arr')

    num_widths = len(widths)
    radii = radii_arr[:, 0]
    maxRadius = max(radii)
    
    for ind in range(2 * rounds):
        # For each angle, plot and save
        with plt.style.context(maxcut_style):
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

            # Get figure name and title for figure
            if ind < rounds:
                # betas go from 0 to pi
                addnlabel = r'$2\beta_{}$'.format(ind + 1)
                fname = 'beta-{}'.format(ind+1)
            else:
                # gammas go from 0 to 2 pi
                addnlabel = r'$\gamma_{}$'.format(ind - rounds + 1)
                fname = 'gamma-{}'.format(ind - rounds + 1)

            plt.suptitle(fulltitle + '\n' + addnlabel)

            # Get the angles
            if ind < rounds:
                # then we are plotting 2 * betas, so multiply by 2
                angles = 2 * arranged_thetas_arr[:, :, ind].flatten()
            else:
                # we are plotting gammas, so do not multiple
                angles = arranged_thetas_arr[:, :, ind].flatten()

            # Plot the dots on the radar plot
            ars = arranged_AR_arr.flatten()  # approximation ratios
            radii_plt = radii_arr.flatten()
            restart_dots = ax.scatter(angles, radii_plt, c=ars, s=100, cmap='YlGn',
                                      marker='o', alpha=0.8, label=r'Converged Angles'.format(0+1), linewidths=0)

            # Plot the fixed angles
            if ind < rounds:
                # 2 beta, so multiply by 2
                fixed_ang = [thetas_fixed[ind] * 2] * num_widths
            else:
                # 1 * gamma, do not multiply
                fixed_ang = [thetas_fixed[ind]] * num_widths
            ax.scatter(fixed_ang, radii, c=[ar_fixed] * num_widths, s=100,
                       cmap='YlGn', marker='s', alpha=0.8, edgecolor='k', label='Fixed Angels')

            ax.set_rmax(maxRadius+1)
            ax.set_rticks(radii, labels=[str(w) for w in widths], fontsize=5)
            ax.set_xticks(np.pi/2 * np.arange(4), labels=[
                r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'], fontsize=15)
            ax.set_rlabel_position(0)
            ax.grid(True)

            # ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
            cbar = plt.colorbar(restart_dots, location='left',
                                shrink=0.8, orientation='vertical', aspect=15)

            cbar.set_label("Energy Approximation Ratio")
            fig.tight_layout()

    
            ## For now, not saving the plots.
            # fold = os.path.join(
            #     '__radarPlots',  'rounds-{}_shots-{}'.format(rounds, num_shots), 'final')
            # if not os.path.exists(fold):
            #     os.makedirs(fold)
            # filename = os.path.join(fold, fname)
            # plt.savefig(filename + '.jpg')
            # plt.savefig(filename + '.pdf')


def rearrange_radar_data_for_plotting(gen_prop):
    """
    Return a dictionary with radii, and angles of the final and initial points of the minimizer, along with the initial and final approximation ratios

    Returns:
        dictionary
    """

    # First, retrieve radar data
    radar_data = get_radar_data_from_metrics()
    widths = gen_prop.get('widths')
    num_widths = len(list(radar_data.keys()))
    maxRadius = 10  # of polar plot
    minRadius = 2  # of polar plot
    radii = np.linspace(minRadius, maxRadius, num_widths)

    niter = gen_prop.get("max_circuits")
    rounds = gen_prop.get("rounds")

    radii_arr = np.zeros((num_widths, niter))
    arranged_AR_arr = np.zeros((num_widths, niter))
    arranged_thetas_arr = np.zeros((num_widths, niter, 2 * rounds))

    arranged_initthetas_arr = np.zeros((num_widths, niter, 2 * rounds))
    arranged_init_AR_arr = np.zeros((num_widths, niter))

    for width_ind, width in enumerate(widths):
        radii_arr[width_ind, :] = radii[width_ind]
        for iter_ind in range(niter):
            # Get color of dot
            ar = radar_data[width]['ARs'][iter_ind]
            # Get final angle location
            thetas = radar_data[width]['final_angles'][iter_ind]

            # Put these values in a list
            arranged_AR_arr[width_ind, iter_ind] = ar
            arranged_thetas_arr[width_ind, iter_ind, :] = thetas

            # Put initial values in list
            arranged_init_AR_arr[width_ind,
                                 iter_ind] = radar_data[width]['init_AR'][iter_ind]
            arranged_initthetas_arr[width_ind, iter_ind,
                                    :] = radar_data[width]['init_angles'][iter_ind]

    return dict(arranged_AR_arr=arranged_AR_arr,
                arranged_thetas_arr=arranged_thetas_arr,
                arranged_init_AR_arr=arranged_init_AR_arr,
                arranged_initthetas_arr=arranged_initthetas_arr,
                radii_arr=radii_arr
                )


def get_radar_data_from_metrics():
    """
    Extract only the initial angles, final angles and approximation ratios from data stored in metrics

    Returns:
        dictionary: with initial and final angles, and initial and final approximation ratios
    """
    restarts = maxcut_benchmark.maxcut_inputs.get("max_circuits")
    detail2 = metrics.circuit_metrics_detail_2

    widths = list(detail2.keys())
    radar_data = {width: dict(ARs=[], init_angles=[], final_angles=[], init_AR=[])
                  for width in widths
                  }

    for width in widths:
        iter_inds = list(range(1, restarts + 1))
        for iter_ind in iter_inds:
            cur_mets = detail2[width][iter_ind]
            # Initial angles and ARs
            starting_iter_index = min(list(cur_mets.keys()))
            cur_init_thetas = cur_mets[starting_iter_index]['thetas_array']
            cur_init_AR = cur_mets[starting_iter_index]['approx_ratio']
            radar_data[width]['init_angles'].append(cur_init_thetas)
            radar_data[width]['init_AR'].append(cur_init_AR)

            # Final angles and ARs
            fin_iter_index = max(list(cur_mets.keys()))
            cur_final_thetas = cur_mets[fin_iter_index]['thetas_array']
            cur_AR = cur_mets[fin_iter_index]['approx_ratio']
            radar_data[width]['ARs'].append(cur_AR)
            radar_data[width]['final_angles'].append(cur_final_thetas)

    return radar_data
