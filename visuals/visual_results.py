from .video_generation.visualization import Visualization
import pickle, os, time, pandas as pd, numpy as np
from matplotlib import pyplot as plt
import torch
import copy
import math

meas = [('ent', "Self-Information (nats)"), ('mut-det', "Determinant Mutual-Information (nats)"), ('mut-dia', "Diagonal Mutual-Information (nats)"), \
        ('prio-det', "Determinant Prior Entropy (nats)"), ('prio-dia', "Diagonal Prior Entropy (nats)"), ('marg-det', "Determinant Marginal Entropy (nats)"), \
        ('post-det', "Determinant Posterior Entropy (nats)"), ('post-dia', "Diagonal Posterior Entropy (nats)"), ('marg-dia', "Diagonal Marginal Entropy (nats)"), \
        ('var', "Variance (units^2)"), ('me', "Mean Error (unit)"), ('mae', "Mean Absolute Error (unit)"), \
        ('rsme', "Root Mean Square Error (unit)"), ('exe_tim', "Execution Time (seconds)"), ('train_tim', "Training Time (seconds)"), ('sen_tim', "Total Episode Time (seconds)")]

#####################################################

def pickle_write(path, contents):
    mkdir([path])
    with open(path + "pickle.pkl", "wb") as f:
        pickle.dump(contents, f, pickle.HIGHEST_PROTOCOL)

#####################################################

def setting_write(path, all_vars_listed, all_params):
    mkdir([path])
    with open(path + "settings.txt", "w") as f:
        for param_key in all_vars_listed:
            for variable_key in all_vars_listed[param_key]:
                f.write(f"\n{param_key} - {variable_key} - {all_params[param_key][variable_key]}")
                if param_key == "net_params" and variable_key == "net_type":
                    f.write(f"\n{param_key} - {variable_key} - {all_params[param_key]['nonlinearity']}")

#####################################################

def list_to_map(lst, dims):
    newlist = [[0.0]*dims[1]]*dims[0]
    for j in range(dims[1]):
        for i in range(dims[0]):
            newlist[i][j] = lst[j*dims[0] + i]
    return newlist

#####################################################

def data_from_exp(agents_experience, all_params):
    paths_with_means = []
    heatmaps = []

    for agent in agents_experience:

        path = []
        heatmap = []
        for exp in agent:
            pos = exp.state.last()
            x = int(pos[0]/all_params['sim_params']['scale_dim'][0])
            y = int(pos[1]/all_params['sim_params']['scale_dim'][1])
            angle = 0 if len(pos) < 3 else pos[2]*45
            
            path.append((x, y, angle))
            heatmap.append([list_to_map(exp.world_understanding['temp'], all_params['sim_params']['world_dim']), \
                            list_to_map(exp.world_understanding['ent'],  all_params['sim_params']['world_dim']), \
                            list_to_map(exp.world_understanding['var'],  all_params['sim_params']['world_dim'])])

        paths_with_means.append(path)
        heatmaps.append(heatmap)

    return paths_with_means, heatmaps

#####################################################

def generate_single_line_plot(path, x, y, degree, filename, averages, mea_print, scatter, standard_dev=None):
    mkdir([path])

    if standard_dev is not None:
        plt.errorbar(x, y, yerr = standard_dev, ecolor='r')
    elif scatter:
        plt.plot(x, y, '.')
    else:
        plt.plot(x, y)

    degree = math.floor(degree)
    if degree > 0:
        coeffs = np.polyfit(x, y, degree)
        approx = np.polyval(coeffs, x)
        plt.plot(x, approx)

    plt.xlabel('Episode Number')
    ylab = f"Agent Average {mea_print}" if averages else f"Total {mea_print}"
    plt.ylabel(ylab)
    
    plt.savefig(path + filename)
    plt.cla()
    plt.clf()
    plt.close()

#####################################################

def generate_line_plots(path, all_measurements, all_params, scatter, degree=0):
    global meas
    mkdir([path])

    episode_cnt = all_params['tra_params']['episodes']
    agent_num = all_params['sim_params']['agent_num']

    x = [i for i in range(episode_cnt)]
    
    total_path = path + "totals/"
    average_path = path + "averages/"
    average_path_errors = path + "averages_with_bars/"

    for mea in meas:
        mea_name = mea[0]
        mea_print = mea[1]
        if mea_name in all_measurements:
            
            y_tot = all_measurements[mea_name][:episode_cnt]
            y_ave = [tot/agent_num for tot in y_tot]

            generate_single_line_plot(average_path_errors, x, y_ave, degree, f"Agent Average {mea_print} Per Episode With Error Bars.png", averages=True,  mea_print=mea_print, scatter=scatter, standard_dev=all_measurements['standard_deviation'][mea_name][:episode_cnt])
            generate_single_line_plot(average_path,        x, y_ave, degree, f"Agent Average {mea_print} Per Episode.png", averages=True,  mea_print=mea_print, scatter=scatter)
            generate_single_line_plot(total_path,          x, y_tot, degree, f"Total {mea_print} Per Episode.png",         averages=False, mea_print=mea_print, scatter=scatter)

#####################################################

def generate_video(video_file_path, pickle_content, perform_training, gen_training, gen_testing, agent_perspective_count):

    relevant_training_episodes = pickle_content["relevant_training_episodes"]
    relevant_testing_results = pickle_content["relevant_testing_results"]
    all_params = pickle_content["all_params"]

    agent_perspective_count = min(agent_perspective_count, all_params['sim_params']['agent_num'])

    if gen_training:
        training_file_path = video_file_path + f"training/"
        mkdir([training_file_path])
        
        for key in relevant_training_episodes:
            training_key_file_path = training_file_path + key + "/"
            mkdir([training_key_file_path])
            
            path, heat = data_from_exp(relevant_training_episodes[key]['exp'], all_params)
            v = Visualization(moves=path, groundTruth=all_params['sim_params']['data_sim'], heatmaps=[[]], \
                                exportDirectory=training_key_file_path, suppressOutput=1, suppressVideo=0, \
                                datadisplayed=("Temperature", "Entropy", "Variance"), dimensions=all_params['sim_params']['world_dim'], scale=7)
            v.visualize()


    if gen_testing:
        testing_file_path = video_file_path + f"testing/"
        mkdir([testing_file_path])
        
        for start_method in relevant_testing_results:
            testing_start_file_path = testing_file_path + start_method + "/"
            mkdir([testing_start_file_path])
            
            for exploration in relevant_testing_results[start_method]:
                testing_start_exploration_file_path = testing_start_file_path + exploration + "/"
                if relevant_testing_results[start_method][exploration]['exp'] is not None:
                    mkdir([testing_start_exploration_file_path])
                    
                    path, heat = data_from_exp(relevant_testing_results[start_method][exploration]['exp'], all_params)
                    v = Visualization(moves=path, groundTruth=all_params['sim_params']['data_sim'], heatmaps=[[]], \
                                    exportDirectory=testing_start_exploration_file_path, suppressOutput=1, suppressVideo=0, \
                                    datadisplayed=("Temperature", "Entropy", "Variance"), dimensions=all_params['sim_params']['world_dim'], scale=7)
                    v.visualize()

        if (not gen_training) and perform_training:
            rew_mech = all_params['tra_params']['reward_mechanism']

            hi_path, hi_heat = data_from_exp(relevant_training_episodes['hi_' + rew_mech]['exp'], all_params)
            lo_path, lo_heat = data_from_exp(relevant_training_episodes['lo_' + rew_mech]['exp'], all_params)

            training_reward_directory = testing_file_path + f"training_{rew_mech}/"
            hi_train_directory = training_reward_directory + f"hi/"
            lo_train_directory = training_reward_directory + f"lo/"

            mkdir([training_reward_directory, hi_train_directory, lo_train_directory])

            hi_v = Visualization(moves=hi_path, groundTruth=all_params['sim_params']['data_sim'], heatmaps=[[]], \
                                    exportDirectory=hi_train_directory, suppressOutput=1, suppressVideo=0, \
                                    datadisplayed=("Temperature", "Entropy", "Variance"), dimensions=all_params['sim_params']['world_dim'], scale=7)
            hi_v.visualize()

            lo_v = Visualization(moves=lo_path, groundTruth=all_params['sim_params']['data_sim'], heatmaps=[[]], \
                                    exportDirectory=lo_train_directory, suppressOutput=1, suppressVideo=0, \
                                    datadisplayed=("Temperature", "Entropy", "Variance"), dimensions=all_params['sim_params']['world_dim'], scale=7)
            lo_v.visualize()

#####################################################
'''
donish
'''
def save_one_network(path, network, agent_num, episode):
    mkdir([path])
    torch.save({'net_state': network.net.state_dict(),
                'opt_state': network.optimizer.state_dict(),
                'agent_id': agent_num,
                'episode': episode
                }, path + f"agent_{agent_num}.pt")

'''
'''
def save_multiple_networks(path, networks, episode):
    for i, network in enumerate(networks):
        save_one_network(path, network, i, episode)
'''
doneish
'''
def save_networks(net_file_path, relevant_training_episodes, relevant_testing_results, delete_networks, post_tests, perform_training, episode):
    training_networks_file_path = net_file_path + f"training/"
    testing_network_file_path = net_file_path + f"testing/"

    mkdir([net_file_path, training_networks_file_path, testing_network_file_path])

    if perform_training:
        for key in relevant_training_episodes:
            current_directory = training_networks_file_path + f"{key}/"
            mkdir([current_directory])
            save_multiple_networks(current_directory, relevant_training_episodes[key]['net'], episode)

    
    if post_tests:
        for key in relevant_testing_results:
            current_directory = testing_network_file_path
            mkdir([current_directory])
            for key1 in relevant_testing_results[key]:
                save_multiple_networks(current_directory, relevant_testing_results[key][key1]['net'], episode)
                break #only needed the one key since it the network isn't trained after this point, will fix with a less confusing solution later
            break #ditto

    if delete_networks:
        if perform_training:
            for key in relevant_training_episodes:
                relevant_training_episodes[key]['net'] = None
        
        if post_tests:
            for key in relevant_testing_results:
                for key1 in relevant_testing_results[key]:
                    relevant_testing_results[key][key1]['net'] = None

#####################################################

'''
doneish
'''
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

#####################################################

'''
doneish
'''
def generate_single_bar_plot(bar_plot_file_path, labels, values, decimal_rounding_place, averages, mea_print, filename):
    mkdir([bar_plot_file_path])
    
    results = [round(i, decimal_rounding_place) for i in values]

    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    rect = ax.bar(x=x, height=results)

    ylab = f"Agent Average {mea_print}" if averages else f"Total {mea_print}"
    ax.set_ylabel(ylab)
        
    ax.set_xlabel('Navigation Method')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    autolabel(rect, ax)

    mins = min(results)
    maxs = max(results)

    annotation_percentage = 1.0/8.0


    y_limits = ax.get_ylim()

    if mins >= 0.0:
        y_limits = [0.0, maxs/(1-annotation_percentage)]
    elif maxs <= 0.0:
        y_limits = [mins/(1-annotation_percentage), 0.0]
    else:
        y_limits = [mins/(1-annotation_percentage), maxs/(1-annotation_percentage)]

    ax.set_ylim(y_limits)
    fig.tight_layout()
    plt.savefig(bar_plot_file_path + filename)
    plt.cla()
    plt.clf()
    plt.close()

#####################################################

'''
doneish
'''
def generate_bar_plots(bar_plots_file_path, relevant_testing_results, all_measurements, all_params):
    global meas

    interests = ['model', 'greedy', 'random', 'final_model'] #['model', 'greedy', 'random', 'final_model', 'teleport']
    removals = [(2, "Random"), (3, "Final Model")]
    
    mkdir([bar_plots_file_path])

    for starting_method in relevant_testing_results:
        starting_method_directory = bar_plots_file_path + f"{starting_method}/"
        mkdir([starting_method_directory])

        for mea in meas:
            mea_name = mea[0]
            mea_print = mea[1]
            middle_directory = starting_method_directory + f"{mea_name}/"
            mkdir([middle_directory])
            
            average_directory = middle_directory + f"averages/"
            total_directory = middle_directory + f"totals/"

            agent_num = all_params['sim_params']['agent_num']

            tot_results = [  0.0, all_measurements[mea_name][relevant_testing_results[starting_method]['greedy']['epi']], \
                                  all_measurements[mea_name][relevant_testing_results[starting_method]['random']['epi']], \
                                  all_measurements[mea_name][relevant_testing_results[starting_method]['explotation']['epi']] ]

            if starting_method == "standard_starts":
                tot_results[0] = max(all_measurements[mea_name][:all_params['tra_params']['episodes']])
                #print(mea_name, tot_results[0])

            elif starting_method != "random_starts":
                tot_results[0] = all_measurements[mea_name][relevant_testing_results[starting_method]['greedy']['original_epi']]
        
            ave_results = [tot/agent_num for tot in tot_results]

            rounding = 4            
            
            """
            ave_filename = f"Agent Average {mea_print} vs Navigation Method"
            tot_filename = f"Total {mea_print} vs Navigation Method"

            generate_single_bar_plot(average_directory, interests, tot_results, rounding, averages=True,  mea_print=mea_print, filename=ave_filename)
            generate_single_bar_plot(total_directory,   interests, ave_results, rounding, averages=False, mea_print=mea_print, filename=tot_filename)
            print('hi2')
            """
            for removal_counter in range(2**len(removals)):
                removal_indicators = format(removal_counter, '0' + str(len(removals)) + 'b')

                unwanted_ind = []
                unwanted_num = []
                for i in range(len(removals)):
                    if removal_indicators[i] == '1':
                        unwanted_ind.append(i)
                        unwanted_num.append(removals[i][0])

                cur_interests = copy.deepcopy(interests)
                cur_ave = copy.deepcopy(ave_results)
                cur_tot = copy.deepcopy(tot_results)

                unwanted_num.reverse()
                for unwanted in unwanted_num:
                    del cur_interests[unwanted]
                    del cur_ave[unwanted]
                    del cur_tot[unwanted]
                
                ave_filename = f"Agent Average {mea_print} vs Navigation Method"
                tot_filename = f"Total {mea_print} vs Navigation Method"

                if len(unwanted_ind) > 0:
                    ave_filename = ave_filename + f" Without {removals[unwanted_ind[0]][1]}"
                    tot_filename = tot_filename + f" Without {removals[unwanted_ind[0]][1]}"

                for i in range(1, len(unwanted_ind)):
                    ave_filename = ave_filename + f" and {removals[unwanted_ind[i]][1]}"
                    tot_filename = tot_filename + f" and {removals[unwanted_ind[i]][1]}"

                ave_filename = ave_filename + ".png"
                tot_filename = tot_filename + ".png"

                generate_single_bar_plot(average_directory, cur_interests, cur_ave, rounding, averages=True,  mea_print=mea_print, filename=ave_filename)
                generate_single_bar_plot(total_directory,   cur_interests, cur_tot, rounding, averages=False, mea_print=mea_print, filename=tot_filename)

#####################################################

def generate_outputs(scenario_file_path, pickle_content, all_vars_listed, index, files_generated):
    num = index
    PICKLE_FILE = files_generated['PICKLE_FILE']
    SETTINGS_FILE = files_generated['SETTINGS_FILE']
    LINE_PLOTS = files_generated['LINE_PLOTS']
    TRAINING_VIDS = files_generated['TRAINING_VIDS']
    TESTING_VIDS = files_generated['TESTING_VIDS']
    BAR_PLOTS = files_generated['BAR_PLOTS']
    PERFORM_TRAINING = files_generated['PERFORM_TRAINING']
    POST_TESTS = files_generated['POST_TESTS']
    SAVE_NETS = files_generated['SAVE_NETS']

    relevant_training_episodes = pickle_content["relevant_training_episodes"]
    relevant_testing_results = pickle_content["relevant_testing_results"]
    all_params = pickle_content["all_params"]
    all_measurements = pickle_content["all_measurements"]

    bar_plots_file_path = scenario_file_path + f"bar_plots/"
    net_file_path = scenario_file_path + f"agent_networks/"
    scatter_file_path = scenario_file_path + f"line_plots/"
    video_file_path = scenario_file_path + f"videos/"

    #saving the neural networks that were significant during training in that state and the final model/set of networks once training is done
    if SAVE_NETS:
        save_networks(net_file_path, relevant_training_episodes, relevant_testing_results, delete_networks=True, post_tests=POST_TESTS, perform_training=PERFORM_TRAINING, episode=all_params['tra_params']['episodes'])

    #saving the measurement information in a binary file
    if PICKLE_FILE:
        pickle_write(scenario_file_path, pickle_content)

    #saving the measurement information in a text file
    if SETTINGS_FILE:
        setting_write(scenario_file_path, all_vars_listed, all_params)

    #saving images of bar plots that compare random, greedy, final model and training model results
    if (BAR_PLOTS and POST_TESTS):
        generate_bar_plots(bar_plots_file_path, relevant_testing_results, all_measurements, all_params)

    #generating images of plots of all the measurements (rewards, variance, mean error, mean absolute error, root mean square error, execution time and training time)
    if LINE_PLOTS and PERFORM_TRAINING:
        generate_line_plots(scatter_file_path, all_measurements, all_params, scatter=False, degree=0)

    #generating videos of the paths taken by the relevant training episodes and relevant testing episodes
    if (TRAINING_VIDS and PERFORM_TRAINING) or (TESTING_VIDS and POST_TESTS):
        generate_video(video_file_path, pickle_content, PERFORM_TRAINING, TRAINING_VIDS, TESTING_VIDS*POST_TESTS, agent_perspective_count=0)

#####################################################

def mkdir(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)