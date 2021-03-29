#####################################################

'''Modules'''

# Roll-out k-steps implementation
import csv, os, sys, datetime, math, logging, warnings, random, time, shutil, pickle, copy

import pandas as pd, numpy as np, seaborn as sns

from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#Our libraries
from ipp_model.ipp import Scenario
from visuals.visual_results import generate_outputs, setting_write

#####################################################

INPUT_PICKLE_DIR = r"/run/user/1000/gvfs/sftp:host=newton,user=skhodadadeh/home/skhodadadeh/MFG-MARL-IPP/data/_outputs/Large.Tests.12/mixed_variables/scenario.0.1613758425.6972826/intermediate_saves/epi4999"
INPUT_PICKLE_NAME= r"pickle.pkl"

'''Data'''
#input and output directories
BASE_DIR = "./data/"
ENV_PATH = "_inputs/env200.csv"
OUT_PATH = BASE_DIR + "_outputs/"
case = 1
while(True):
    if not os.path.exists(OUT_PATH + f"Large.Tests.{case}/"):
        break
    case = case + 1
TEST_PATH = OUT_PATH + f"Large.Tests.{case}/"
SIN_PATH = TEST_PATH + "single_variables/"
MIX_PATH = TEST_PATH + "mixed_variables/"

#####################################################

'''RESULTS DISPLAYED'''

EXPERIMENT_RUNS = 3
SHUFFLE_TEST_VARS = False
SINGLE_VARIABLE = True
MULTIPLE_VARIABLE = False
PICKLE_FILE = True
SETTINGS_FILE = True
PERFORM_TRAINING = True
POST_TESTS = True
SAVE_NETS = True

LINE_PLOTS = False
TRAINING_VIDS = False
TESTING_VIDS = False
BAR_PLOTS = False

FIXED_BUDGET = True # otherwise the size of the environemt is considered fix and the budget will become variable
FIXED_BUDGET_AMOUNT = 20 # must be at least two

'''GLOBAL VARIABLES'''
num = 0
base_parameters = {}
files_generated = {
    'PICKLE_FILE': PICKLE_FILE,
    'SETTINGS_FILE': SETTINGS_FILE,
    'LINE_PLOTS': LINE_PLOTS,
    'TRAINING_VIDS': TRAINING_VIDS,
    'TESTING_VIDS': TESTING_VIDS,
    'BAR_PLOTS': BAR_PLOTS,
    'PERFORM_TRAINING': PERFORM_TRAINING,
    'POST_TESTS': POST_TESTS,
    'SAVE_NETS': SAVE_NETS
}
additional_settings_to_save = {'sim_params': ['agent_num', 'queue_len', 'queue_len_percent', 'world_dim', 'comm_range', 'same_gpr', 'same_init_neural_network', 'middle_starting_area_percentage'], 
                               'net_params': ['net_type', 'nonlinearity'], 
                               'gau_params': ['ell', 'std'], 
                               'tra_params': ['budget', 'budget_percent', 'episodes', 'PUNISH', 'reward_mechanism']}

#####################################################

# Retreiving the environment data
ground_truth = pd.read_csv(BASE_DIR+ENV_PATH, header=None).values
sim_params = {}
net_params = {}
gau_params = {}
tra_params = {}

sim_vars = {}
net_vars = {}
gau_vars = {}
tra_vars = {}

#####################################################

'''Simulation Parameters'''
sim_params['comm_range'] = 1.0 # 0.0 <= comm_range <= 1.0
sim_params['world_dim'] = [24,24] #this is a 50 x 50 env
sim_params['data_sim'] = np.array(ground_truth)[np.ix_(range(0, sim_params['world_dim'][0]), range(0, sim_params['world_dim'][1]))].flatten()
sim_params['agent_num'] = 10
sim_params['queue_len'] = 10
sim_params['queue_len_percent'] = 0.5
sim_params['scale_dim'] = [1,1]
sim_params['action_num'] = 8 #can be 4 or 8
sim_params['prev_bad_actions'] = True
sim_params['mean_field'] = True
sim_params['same_node_visits'] = True
sim_params['same_init_neural_network'] = True
sim_params['same_gpr'] = True #implements a central gpr
sim_params['mean_method'] = 'mean' #can be 'mean', 'mode', 'rounded-mean', 'floor-mean', 'ceil-mean'
sim_params['action_method'] = 'training' #can be training, explotation, greedy, random
sim_params['no_neighbors_method'] = 'zero' #can be 'zero', 'last_mean', 'past_ave'; ONLY USE 'zero'
sim_params["middle_starting_area_percentage"] = 0.5 #amount of the middle area used for random starts


'''Simulation Varying Parameters'''
#sim_vars['agent_num'] = [3,7,10]
sim_vars['queue_len_percent'] = [0.25, 0.5, 0.75]

#####################################################

'''Neural Network Parameters'''
#in effect
net_params['input_length'] = len(sim_params['world_dim']) + sim_params['mean_field']
net_params['output_length'] = sim_params['action_num']
net_params['hidden_length'] = 10
net_params['internal_lengths'] = 5
net_params['dev'] = "cuda"
net_params['name'] = "test"
net_params['sequence_length'] = sim_params['queue_len'] #MANDATORY EQUALITY
net_params['nonlinearity'] = 'tanh' #can be 'relu' or 'tanh', only affects 'RNN'
net_params['dropout'] = 0.0 #should be between 0.0 and 1
net_params['net_type'] = "LSTM.tanh" #can be 'LSTM', 'RNN', or 'GRU'

'''Neural Network Varying Paramaters'''
#net_vars['internal_lengths'] = [1, 5, 10, 20, 30, 50]

#####################################################

'''Gaussian Process Parameters'''
gau_params['init_sample_percent'] = 0.1
gau_params['ell'] = 0.5
gau_params['std'] = 1.0

'''Gaussian Process Varying Parameters'''
gau_vars = {}

#####################################################

'''Agent Training Parameters'''
tra_params['episodes'] = 5000
tra_params['episode_save_frequency'] = 500 # save the agent networks every 500 episodes
tra_params['episode_sample_training_size'] = 25 # amount of episodes that the networks are sampled from (you can assume one would be added for the current episode)
tra_params['budget'] = 20
tra_params['budget_percent'] = 0.3 #should be a value between 0.0 and 1.0
tra_params['epochs'] = 5
tra_params['batch_size'] = 64
tra_params['epsilon'] = 1.0
tra_params['epsilon_decay'] = 0.999
tra_params['new_starts'] = False
tra_params['new_GP'] = True
tra_params['group_exp'] = True
tra_params['trim_data'] = True
tra_params['update_gpr'] = True
tra_params['SHIFT'] = sys.float_info.min
tra_params['PUNISH'] = 0.0 #math.log(sys.float_info.min)
tra_params['PREVENT'] = -1 * sys.float_info.max
tra_params['reward_mechanism'] = 'mut-dia' #can be 'var', 'ent', 'info-dia', 'info-det', 'mut-det', 'mut-dia'
tra_params['train_percent'] = 1 #training on the recent tra_params['episodes']*tra_params['train_percent'] episodes

'''Agent Training Varying Parameters'''
tra_vars['budget_percent'] = [0.4, 0.5] #[40, 0.50, 0.60]
#tra_vars['PUNISH'] = [0.0, math.log(sys.float_info.min)]
#####################################################

def display_configuration(all_params, all_vars_listed):
    print(f"\nCurrent Experiment Settings.")
    for param_key in all_vars_listed:
        for variable_key in all_vars_listed[param_key]:
            print(f"{param_key} - {variable_key} - {all_params[param_key][variable_key]}")
            if param_key == "net_params" and variable_key == "net_type":
                print(f"{param_key} - {variable_key} - {all_params[param_key]['nonlinearity']}")
    print()

#####################################################

def finalize_parameters(all_params, all_vars_listed):
    global ground_truth, FIXED_BUDGET, FIXED_BUDGET_AMOUNT

    if FIXED_BUDGET:
        all_params['tra_params']['budget'] = FIXED_BUDGET_AMOUNT
        environment_side_length = min(math.ceil(math.sqrt(all_params['sim_params']['agent_num'] * all_params['tra_params']['budget']/all_params['tra_params']['budget_percent'])), ground_truth.shape[0])
        all_params['sim_params']['world_dim'] = [environment_side_length - 1, environment_side_length - 1]
    
    else:
        all_params['tra_params']['budget'] = max(int(all_params['tra_params']['budget_percent'] * (all_params['sim_params']['world_dim'][0] + 1) * (all_params['sim_params']['world_dim'][1] + 1) / all_params['sim_params']['agent_num']), 2)
    
    all_params['sim_params']['data_sim'] = np.array(ground_truth)[np.ix_(range(0, all_params['sim_params']['world_dim'][0]), range(0, all_params['sim_params']['world_dim'][1]))].flatten()
    all_params['sim_params']['queue_len'] = max(int(all_params['tra_params']['budget'] * all_params['sim_params']['queue_len_percent']),1)
    all_params['net_params']['input_length'] = max(int(len(all_params['sim_params']['world_dim']) + all_params['sim_params']['mean_field']),1)
    all_params['net_params']['sequence_length'] = all_params['sim_params']['queue_len']

    if '.' in all_params['net_params']['net_type']:
        parts = all_params['net_params']['net_type'].split('.')
        all_params['net_params']['net_type'] = parts[0]
        all_params['net_params']['nonlinearity'] = parts[1]
    
    display_configuration(all_params, all_vars_listed)

#####################################################

def sin_vars_test(all_params, all_vars, all_vars_listed, output_directory):
    global num, base_parameters, files_generated

    for parameter in all_vars:
        for variable in all_vars[parameter]:
            base_file_path1 = output_directory + f"{parameter}.{variable}/"
            if not os.path.exists(base_file_path1):
                os.makedirs(base_file_path1)
            for value in all_vars[parameter][variable]:
                all_params[parameter][variable] = value
                
                start_time_exe = time.time()

                base_file_path = base_file_path1 + f"{value}/"
                if not os.path.exists(base_file_path):
                    os.makedirs(base_file_path)

                scenario_file_path = base_file_path + f"scenario.{num}.{start_time_exe}/"
                if not os.path.exists(scenario_file_path):
                    os.makedirs(scenario_file_path)

                finalize_parameters(all_params, all_vars_listed)

                setting_write(scenario_file_path, all_vars_listed, all_params)

                print(f"Generating scenario-{num}.")
                sen = Scenario(sim_params=all_params['sim_params'], net_params=all_params['net_params'], gau_params=all_params['gau_params'], tra_params=all_params['tra_params'])
                print(f"Starting scenario-{num}.")
                relevant_training_episodes, relevant_testing_results, all_measurements = sen.simulation(PERFORM_TRAINING, POST_TESTS, scenario_file_path + "intermediate_saves/")
                print(f"Total time for scenario-{num} execution is {time.time() - start_time_exe} seconds.")

                start_time_vis = time.time()

                pickle_content = {'relevant_training_episodes': relevant_training_episodes, 'relevant_testing_results': relevant_testing_results, 'all_measurements':all_measurements, 'all_params':all_params, 'starting_nodes':sen.starting_nodes, 'initial_sample': sen.world.SAMPLE}
                generate_outputs(scenario_file_path, pickle_content, all_vars_listed, num, files_generated)

                print(f"Total time for scenario-{num} visualization is {time.time() - start_time_vis} seconds.")
                print(f"Total time for scenario-{num} is {time.time() - start_time_exe} seconds.")

                num = num + 1
                
            all_params = copy.deepcopy(base_parameters)

#####################################################

def mix_vars_test(all_params, all_vars, all_vars_listed, output_directory):
    global num, base_parameters, files_generated

    for k in all_vars:
        cur_vars = all_vars[k]
        cur_params = all_params[k]
        if len(cur_vars) != 0:
            for key in cur_vars:
                for val in cur_vars[key]:
                    cur_params[key] = val
                    all_vars_copy = copy.deepcopy(all_vars)
                    del all_vars_copy[k][key]
                    mix_vars_test(all_params, all_vars_copy, all_vars_listed, output_directory)
        else:
            return

    start_time_exe = time.time()
    scenario_file_path = output_directory + f"scenario.{num}.{start_time_exe}/"

    finalize_parameters(all_params, all_vars_listed)

    setting_write(scenario_file_path, all_vars_listed, all_params)

    sen = Scenario(sim_params=all_params['sim_params'], net_params=all_params['net_params'], gau_params=all_params['gau_params'], tra_params=all_params['tra_params'])
    relevant_training_episodes, relevant_testing_results, all_measurements = sen.simulation(PERFORM_TRAINING, POST_TESTS, scenario_file_path + "intermediate_saves/")

    print(f"Total time for scenario-{num} execution is {time.time() - start_time_exe} seconds.")

    start_time_vis = time.time()

    pickle_content = {'relevant_training_episodes': relevant_training_episodes, 'relevant_testing_results': relevant_testing_results, 'all_measurements':all_measurements, 'all_params':all_params, 'starting_nodes':sen.starting_nodes, 'initial_sample': sen.world.SAMPLE}
    generate_outputs(scenario_file_path, pickle_content, all_vars_listed, num, files_generated)

    print(f"Total time for scenario-{num} visualization is {time.time() - start_time_vis} seconds.")
    print(f"Total time for scenario-{num} is {time.time() - start_time_exe} seconds.")

    num = num + 1

#####################################################

def single_run(all_params, all_vars, all_vars_listed, output_directory):
    global num, base_parameters, files_generated

    start_time_exe = time.time()
    scenario_file_path = output_directory + f"scenario.{num}.{start_time_exe}/"

    finalize_parameters(all_params, all_vars_listed)

    setting_write(scenario_file_path, all_vars_listed, all_params)
    if not os.path.exists(scenario_file_path):
        os.makedirs(scenario_file_path)

    sen = Scenario(sim_params=all_params['sim_params'], net_params=all_params['net_params'], gau_params=all_params['gau_params'], tra_params=all_params['tra_params'])
    print(f"Starting scenario-{num}.")
    relevant_training_episodes, relevant_testing_results, all_measurements = sen.simulation(PERFORM_TRAINING, POST_TESTS, scenario_file_path + "intermediate_saves/")
    print(f"Total time for scenario-{num} execution is {time.time() - start_time_exe} seconds.")

    start_time_vis = time.time()

    pickle_content = {'relevant_training_episodes': relevant_training_episodes, 'relevant_testing_results': relevant_testing_results, 'all_measurements':all_measurements, 'all_params':all_params, 'starting_nodes':sen.starting_nodes, 'initial_sample': sen.world.SAMPLE}
    generate_outputs(scenario_file_path, pickle_content, all_vars_listed, num, files_generated)

    print(f"Total time for scenario-{num} visualization is {time.time() - start_time_vis} seconds.")
    print(f"Total time for scenario-{num} is {time.time() - start_time_exe} seconds.")

    num = num + 1

#####################################################

def single_run_with_pickle(pickle_directory, pickle_file_name):
    global num, base_parameters, files_generated

    pickle_file_location = pickle_directory + "/" + pickle_file_name
    print(pickle_file_location)
    start_time_exe = time.time()
    scenario_file_path = pickle_directory + "/" + f"scenario.{num}.{start_time_exe}/"
    
    if not os.path.exists(scenario_file_path):
        os.makedirs(scenario_file_path)
    
    pickle_content = None
    
    with open(pickle_file_location, 'rb') as file:
        pickle_content = pickle.load(file)
    
    all_params = pickle_content['all_params']
    starting_locations = pickle_content['relevant_training_episodes']['hi_var']['loc']
    sen = Scenario(sim_params=all_params['sim_params'], net_params=all_params['net_params'], gau_params=all_params['gau_params'], tra_params=all_params['tra_params'], starting_locations=starting_locations)
    relevant_training_episodes, relevant_testing_results, all_measurements = sen.simulation(perform_training=False, post_tests=True, intermediate_save_directory=scenario_file_path + "intermediate_saves/")

    new_pickle_content = {'relevant_training_episodes': relevant_training_episodes, 'relevant_testing_results': relevant_testing_results, 'all_measurements':all_measurements, 'all_params':all_params, 'starting_nodes':sen.starting_nodes, 'initial_sample': sen.world.SAMPLE}
    
    with open(scenario_file_path + "pickle.pkl", "wb") as f:
        pickle.dump(new_pickle_content, f, pickle.HIGHEST_PROTOCOL)

    print()
    print(all_measurements['mut-dia'])

#####################################################

single_run_with_pickle(INPUT_PICKLE_DIR, INPUT_PICKLE_NAME)
quit()
"""
start = time.time()

all_params = {'sim_params': sim_params, 'net_params': net_params, 'gau_params': gau_params, 'tra_params': tra_params}
base_parameters = copy.deepcopy(all_params)
all_vars = {'sim_params': sim_vars, 'net_params': net_vars, 'gau_params': gau_vars, 'tra_params': tra_vars}
all_vars_listed = {'sim_params': set(), 'net_params': set(), 'gau_params': set(), 'tra_params': set()}

add = 0
mul = 1
for param_key in all_vars:
    for var_key in all_vars[param_key]:
        all_vars_listed[param_key].add(var_key)
        add = add + len(all_vars[param_key][var_key])
        mul = mul * len(all_vars[param_key][var_key])
        if SHUFFLE_TEST_VARS:
            random.shuffle(all_vars[param_key][var_key])

for param_key in additional_settings_to_save:
    for var_key in additional_settings_to_save[param_key]:
        all_vars_listed[param_key].add(var_key)


print(f"There are {add*SINGLE_VARIABLE} unique single variable tests and {mul*MULTIPLE_VARIABLE} unique multiple variable tests that are going to be conducted that are repeated {EXPERIMENT_RUNS} times.")

if not os.path.exists(TEST_PATH):
    os.makedirs(TEST_PATH)

if (not os.path.exists(SIN_PATH)) and SINGLE_VARIABLE:
    os.makedirs(SIN_PATH)

if (not os.path.exists(MIX_PATH)) and MULTIPLE_VARIABLE:
    os.makedirs(MIX_PATH)

for i in range(EXPERIMENT_RUNS):
    for queue_len_percent in sim_vars['queue_len_percent']:
        for budget_percent in tra_vars['budget_percent']:
            all_params['sim_params']['queue_len_percent'] = queue_len_percent
            all_params['tra_params']['budget_percent'] = budget_percent
            single_run(all_params, all_vars, all_vars_listed, MIX_PATH)


end = time.time()
print(f"All scenarios completed. Total time delta = {end - start} seconds.")

for i in range(EXPERIMENT_RUNS):
    if SINGLE_VARIABLE:
        sin_vars_test(all_params, all_vars, all_vars_listed, SIN_PATH)

    if MULTIPLE_VARIABLE:
        mix_vars_test(all_params, all_vars, all_vars_listed, MIX_PATH)

end = time.time()
print(f"All scenarios completed. Total time delta = {end - start} seconds.")
"""