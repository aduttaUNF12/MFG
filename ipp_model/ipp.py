import numpy as np, random, copy, sys, time, math, torch

from collections import deque
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from .multiagent.core import IPP_Node, IPP_Agent, IPP_Grid_World, IPP_Experience
from .multiagent.tools import iterable, Fixed_Queue, squared_euclidean
from .multiagent.UpdatableGPR import UpdatableGPR
from .multiagent.TorchNN2 import RNNet

import os
import sys
sys.path.append(os.path.realpath('visuals'))
from visuals import visual_results

#NOTE: May move into class some point in the future
"""
Inputs: 
    s: the number of points of interest (POI) along the side of a square environment
    p: the percentage of the are of the middle region wanted
Outputs: the ids of the middle region POI
"""
def midRegion(s, p):
    a = math.ceil(s*(1-math.sqrt(1-p)))
    k = math.floor(a/2)

    w = (k-1)*(s+1)
    x = k*(s-1)
    y = (s-1)*(s-k+1)
    z = (s+1)*(s-k)

    out = []
    for y1 in range(w, y+s, s):
        for x1 in range(y1, y1+(x-w)+1, 1):
            out.append(x1)
    
    return out

class Scenario:

    #doesn't check if starting locations are valid
    def __init__(self, sim_params, net_params, gau_params, tra_params, starting_locations=[]):
        self.sim_params = sim_params
        self.net_params = net_params
        self.gau_params = gau_params
        self.tra_params = tra_params
        self.starting_locations = starting_locations
        self.world = IPP_Grid_World()
        self.starting_nodes = []
        self.default_gpr = None
        self.default_net = RNNet(input_length=self.net_params['input_length'],output_length=self.net_params['output_length'], hidden_length=self.net_params['hidden_length'], \
                                internal_lengths=self.net_params['internal_lengths'], dev=self.net_params['dev'], \
                                sequence_length=net_params['sequence_length'], nonlinearity=net_params['nonlinearity'], dropout=net_params['dropout'], net_type=net_params['net_type'], name=self.net_params['name'])
        self.make_world()

        self.sim_params['initial_same_gpr'] = self.sim_params['same_gpr']


    def make_world(self):
        self.create_actions()
        self.create_nodes()
        self.create_agents()

        print("Generating world.")
        self.world.true_comm_range = self.sim_params['comm_range'] * squared_euclidean(self.world.nodes[0].real_coor, self.world.nodes[-1].real_coor)
        self.world.prevent_bad_actions = self.sim_params['prev_bad_actions']
        self.world.mean_field = self.sim_params['mean_field']
        self.world.same_node_visits = self.sim_params['same_node_visits']
        self.world.mean_method = self.sim_params['mean_method']
        self.world.action_method = self.sim_params['action_method']
        self.world.no_neighbors_method = self.sim_params['no_neighbors_method']
        self.world.SHIFT = self.tra_params['SHIFT']
        self.world.PUNISH = self.tra_params['PUNISH']
        self.reset_world(new_starts=True, new_GP=True, new_agents=True)


    def reset_world(self, new_starts=False, new_agents=False, new_GP=False):
        
        update_gpr_with_sample=False

        #remove the visitor memory from the nodes
        for node in self.world.nodes:
            node.visitors = []

        #get new starting locations and generate a new GP
        if new_starts:
            print("Assigning new starting locations to agents.")
            sample_size = int(self.gau_params['init_sample_percent'] * len(self.sim_params['data_sim']))
            if len(self.starting_locations) == 0:
                self.starting_nodes = random.sample(midRegion(self.sim_params['world_dim'][0], self.sim_params["middle_starting_area_percentage"]), max(sample_size, self.sim_params['agent_num'])) #self.starting_nodes = random.sample(range(len(self.sim_params['data_sim'])), self.sim_params['agent_num'])
            else:
                self.starting_nodes = self.starting_locations
                if sample_size > self.sim_params['agent_num']:
                    in_starts = [False]*len(self.world.nodes)
                    not_in = []
                    for node_id in self.starting_nodes:
                        in_starts[node_id] = True
                    for node_id,node_in_start in enumerate(in_starts):
                        if not node_in_start:
                            not_in.append(node_id)
                    self.starting_nodes.extend(random.sample(not_in, sample_size-self.sim_params['agent_num']))

            if new_GP:
                print("Generating a new default GPR for the agents.")
                self.world.SAMPLE = self.starting_nodes[:sample_size]
                #locs = [self.world.nodes[i].real_coor for i in self.world.SAMPLE]
                vals = [self.world.nodes[i].val for i in self.world.SAMPLE]
                std = np.std(vals)
                self.default_gpr = UpdatableGPR(loc=[node.real_coor for node in self.world.nodes], sig=std, ell=0.5)
                
                if update_gpr_with_sample:
                    self.default_gpr.update([(int(id), self.world.nodes[id].val) for id in self.world.SAMPLE])

        
        central_gpr = None
        if self.sim_params['same_gpr']:
            central_gpr = copy.deepcopy(self.default_gpr)       

        #restarting everything about the agents aside from their neural networks
        for agent in self.world.agents:
            #id not reset

            if new_agents:
                print(f"Resetting agent-{agent.id}'s GPR and RNN.")
                if self.sim_params['same_init_neural_network']:
                    agent.net = self.default_net
                else:
                    agent.net = RNNet(input_length=self.net_params['input_length'],output_length=self.net_params['output_length'], hidden_length=self.net_params['hidden_length'], \
                                internal_lengths=self.net_params['internal_lengths'], dev=self.net_params['dev'], \
                                sequence_length=self.net_params['sequence_length'], nonlinearity=self.net_params['nonlinearity'], dropout=self.net_params['dropout'], net_type=self.net_params['net_type'], name=self.net_params['name'])
            else:
                print(f"Resetting agent-{agent.id}'s GPR.")
            
            if central_gpr is None:
                agent.gpr = copy.deepcopy(self.default_gpr)
            else:
                agent.gpr = central_gpr
            
            agent.curr = IPP_Experience()
            agent.curr.state = Fixed_Queue(self.sim_params['queue_len'])
            for i in range(self.sim_params['queue_len']):
                if self.sim_params['mean_field']:
                    agent.curr.state.enq((self.world.nodes[self.starting_nodes[agent.id]].real_coor[0], self.world.nodes[self.starting_nodes[agent.id]].real_coor[1], 0.0))
                else:
                    agent.curr.state.enq((self.world.nodes[self.starting_nodes[agent.id]].real_coor[0], self.world.nodes[self.starting_nodes[agent.id]].real_coor[1]))
            agent.past = []
            agent.loc = self.starting_nodes[agent.id]
            agent.visits = [False]*len(self.world.nodes)
            agent.path = [agent.loc]
            agent.visits[agent.loc] = True
            agent.total_rewards = 0
        
        if update_gpr_with_sample:
            for id in self.world.SAMPLE:
                self.world.nodes[id].visitors.append(-1)
                if self.sim_params['same_node_visits']:
                    for agent in self.world.agents:                        
                        agent.visits[id] = True


    def create_nodes(self):
        print("Generating nodes/POIs.")
        data = np.array(self.sim_params['data_sim']).reshape(self.sim_params['world_dim'])

        nodes = []

        #TODO: Add scaling to the second set of parameters
        cnt = 0
        if len(self.sim_params['world_dim'])==2:
            for j in range(self.sim_params['world_dim'][1]):
                for i in range(self.sim_params['world_dim'][0]):
                    
                    node = IPP_Node(cnt, (i,j), (i*self.sim_params['scale_dim'][0],j*self.sim_params['scale_dim'][1]), data[j][i])

                    if self.sim_params['action_num'] == 4:
                        node.valid_actions = {0: node.id+1, 1: node.id-self.sim_params['world_dim'][0], 2: node.id-1, 3: node.id+self.sim_params['world_dim'][0]}
                        if i == 0:
                            node.valid_actions.pop(2, None)
                        elif i == self.sim_params['world_dim'][0]-1:
                            node.valid_actions.pop(0, None)
                        if j == 0:
                            node.valid_actions.pop(1, None)
                        elif j == self.sim_params['world_dim'][1]-1:
                            node.valid_actions.pop(3, None)
                    else:
                        node.valid_actions = {0: node.id+1, 1: node.id-self.sim_params['world_dim'][0]+1, 2: node.id-self.sim_params['world_dim'][0], 3: node.id-self.sim_params['world_dim'][0]-1, \
                                            4: node.id-1, 5: node.id+self.sim_params['world_dim'][0]-1, 6: node.id+self.sim_params['world_dim'][0], 7: node.id+self.sim_params['world_dim'][0]+1}
                        if i == 0:
                            node.valid_actions.pop(3, None)
                            node.valid_actions.pop(4, None)
                            node.valid_actions.pop(5, None)
                        elif i == self.sim_params['world_dim'][0]-1:
                            node.valid_actions.pop(0, None)
                            node.valid_actions.pop(1, None)
                            node.valid_actions.pop(7, None)
                        if j == 0:
                            node.valid_actions.pop(1, None)
                            node.valid_actions.pop(2, None)
                            node.valid_actions.pop(3, None)
                        elif j == self.sim_params['world_dim'][1]-1:
                            node.valid_actions.pop(5, None)
                            node.valid_actions.pop(6, None)
                            node.valid_actions.pop(7, None)

                    nodes.append(node)
                    cnt = cnt + 1
        self.world.nodes = nodes


    def create_agents(self):
        print("Generating agents.")
        self.world.agents = [IPP_Agent(i) for i in range(self.sim_params['agent_num'])]


    def create_actions(self):
        print("Generating actions.")
        self.world.actions = [float(i) for i in range(self.sim_params['action_num'])]


    def simulation(self, perform_training, post_tests, intermediate_save_directory):
        
        x = -1

        episodes = self.tra_params['episodes']
        budget = self.tra_params['budget']
        epochs = self.tra_params['epochs']
        batch_size = self.tra_params['batch_size']
        new_starts = self.tra_params['new_starts']
        new_GP = self.tra_params['new_GP']
        epsilon = self.tra_params['epsilon']
        epsilon_decay = self.tra_params['epsilon_decay']
        group_exp = self.tra_params['group_exp']
        trim_data = self.tra_params['trim_data']
        reward_mechanism = self.tra_params['reward_mechanism']
        train_percent = self.tra_params['train_percent']
        sample_size = self.tra_params['episode_sample_training_size']

        shift = self.sim_params['queue_len'] if trim_data else 0

        assert 0.0 <= epsilon <= 1.0 and 0.0 <= epsilon_decay <= 1.0, "Epsilon and Epsilon Decay should be between 0.0 and 1.0."
        # these will be used to maintain all the information relating to the episodes with the highest/lowest rewards collected, resulting variance, execution time (doesn't include training time), computed RSME
        base_information = {'epi': -1, 'loc':None, 'exp': None, 'net': None, 'original_epi': None} #[episode number that is was last updated from, the starting locations of the location, the experience that object held]
        relevant_training_episodes = {'hi_var':copy.deepcopy(base_information), 'lo_var':copy.deepcopy(base_information), \
                                      'hi_ent':copy.deepcopy(base_information), 'lo_ent':copy.deepcopy(base_information), \
                                      'hi_mut-det':copy.deepcopy(base_information), 'lo_mut-det':copy.deepcopy(base_information), \
                                      'hi_mut-dia':copy.deepcopy(base_information), 'lo_mut-dia':copy.deepcopy(base_information), \
                                      'hi_prio-det':copy.deepcopy(base_information), 'lo_prio-det':copy.deepcopy(base_information), \
                                      'hi_prio-dia':copy.deepcopy(base_information), 'lo_prio-dia':copy.deepcopy(base_information), \
                                      'hi_marg-det':copy.deepcopy(base_information), 'lo_marg-det':copy.deepcopy(base_information), \
                                      'hi_marg-dia':copy.deepcopy(base_information), 'lo_marg-dia':copy.deepcopy(base_information), \
                                      'hi_post-det':copy.deepcopy(base_information), 'lo_post-det':copy.deepcopy(base_information), \
                                      'hi_post-dia':copy.deepcopy(base_information), 'lo_post-dia':copy.deepcopy(base_information), \
                                      'hi_me':copy.deepcopy(base_information), 'lo_me':copy.deepcopy(base_information), \
                                      'hi_mae':copy.deepcopy(base_information), 'lo_mae':copy.deepcopy(base_information), \
                                      'hi_rsme':copy.deepcopy(base_information), 'lo_rsme':copy.deepcopy(base_information), \
                                      'hi_exe_tim':copy.deepcopy(base_information), 'lo_exe_tim':copy.deepcopy(base_information) }

        #these will be used to maintain all the information relating to different traversal techniques reaction to the different starting locations
        
        base_tests = {'explotation':copy.deepcopy(base_information), 'random':copy.deepcopy(base_information), 'greedy':copy.deepcopy(base_information), 'good_teleport':copy.deepcopy(base_information), 'bad_teleport':copy.deepcopy(base_information)}
        
        relevant_testing_results = {}

        if new_starts:
            relevant_testing_results['random_starts'] = copy.deepcopy(base_tests)
            for key in relevant_training_episodes:
                relevant_testing_results[key] = copy.deepcopy(base_tests)
        else:
            relevant_testing_results['standard_starts'] = copy.deepcopy(base_tests)

        total_measurements = episodes*perform_training + len(base_information)*len(base_tests)*len(relevant_testing_results)*post_tests

        all_measurements = {'var':          [0.0]*(total_measurements), \
                            'ent':          [0.0]*(total_measurements), \
                            'mut-det':      [0.0]*(total_measurements), \
                            'mut-dia':      [0.0]*(total_measurements), \
                            'prio-det':     [0.0]*(total_measurements), \
                            'prio-dia':     [0.0]*(total_measurements), \
                            'marg-det':     [0.0]*(total_measurements), \
                            'marg-dia':     [0.0]*(total_measurements), \
                            'post-det':     [0.0]*(total_measurements), \
                            'post-dia':     [0.0]*(total_measurements), \
                            'me':           [0.0]*(total_measurements), \
                            'mae':          [0.0]*(total_measurements), \
                            'rsme':         [0.0]*(total_measurements), \
                            'exe_tim':      [0.0]*(total_measurements), \
                            'train_tim':    [0.0]*(total_measurements), \
                            'sen_tim':      [0.0]*(total_measurements)  }
        
        all_measurements['standard_deviation'] = copy.deepcopy(all_measurements)
        all_measurements['individual_agent_values'] = copy.deepcopy(all_measurements['standard_deviation'])

        for key in all_measurements['individual_agent_values']:
            all_measurements['individual_agent_values'][key] = [[]]*(total_measurements)

        self.world.epsilon = epsilon

        episode_experience_limit = int(episodes * train_percent)
        data_len = episode_experience_limit * (budget - shift)
        data = [deque(maxlen=int(data_len)) for agent in self.world.agents]

        visual_results.mkdir([intermediate_save_directory])

        if perform_training:
            self.world.action_method = 'training'
            for x in range(episodes):
                self.tra_params['current_episode'] = x
                self.tra_params['current_epsilon'] = self.world.epsilon

                episode_start_time = exec_start = time.time()
                self.reset_world(new_starts=new_starts, new_GP=new_GP)

                print(f"\nStarting Episode {x} exploration.")

                for y in range(budget-1):
                    print(f"Episode {x}, Step {y}.")
                    self.world.step()
                print(f"Episode {x}, Step {budget-1}.")
                self.world.step(last_step=True)
                
                total_execution_time = time.time() - exec_start

                print(f"Episode {x} exploration complete. Total exploration time was {total_execution_time} seconds.")
                
                agent_experiences = self.compute_metrics(all_measurements=all_measurements, x=x, total_execution_time=total_execution_time)

                if x == 0:
                    for key in relevant_training_episodes:
                        relevant_training_episodes[key]['epi'] = 0
                        relevant_training_episodes[key]['loc'] = copy.deepcopy(self.starting_nodes[:len(self.world.agents)])
                        relevant_training_episodes[key]['exp'] = copy.deepcopy(agent_experiences)
                        if self.sim_params['same_init_neural_network']:
                            relevant_training_episodes[key]['net'] = copy.deepcopy([self.world.agents[0].net])
                        else:
                            relevant_training_episodes[key]['net'] = copy.deepcopy([agent.net for agent in self.world.agents])

                    print(f"Episode {x} finished executing and will be used as the initial baseline.\n"+\
                        f"The total variance computed (for the final measurements) was {all_measurements['var'][x]} units^2.\n"+\
                        f"The total entropy collected was {all_measurements['ent'][x]} nats.\n"+\
                        f"The total determinant mutual-information collected was {all_measurements['mut-det'][x]} nats.\n"+\
                        f"The total diagonal mutual-information collected was {all_measurements['mut-dia'][x]} nats.\n"+\
                        f"The total determinant prior entropy collected was {all_measurements['prio-det'][x]} nats.\n"+\
                        f"The total diagonal prior entropy collected was {all_measurements['prio-dia'][x]} nats.\n"+\
                        f"The total determinant marginal entropy collected was {all_measurements['marg-det'][x]} nats.\n"+\
                        f"The total diagonal marginal entropy collected was {all_measurements['marg-dia'][x]} nats.\n"+\
                        f"The total determinant posterior entropy collected was {all_measurements['post-det'][x]} nats.\n"+\
                        f"The total diagonal posterior entropy collected collected was {all_measurements['post-dia'][x]} nats.\n"+\
                        f"The total execution time for the entire traversal was {all_measurements['exe_tim'][x]} seconds.\n"+\
                        f"The total MEAN ERROR compute (for the final measurements) was {all_measurements['me'][x]} units.\n"+\
                        f"The total MEAN ABSOLUTE ERROR compute (for the final measurements) was {all_measurements['mae'][x]} units.\n"+\
                        f"The total ROOT MEAN SQUARE ERROR compute (for the final measurements) was {all_measurements['rsme'][x]} units.\n")
                        
                else:
                    names = [('var', 'VARIANCE', 'varying'), ('ent',  'ENTROPY', 'individually information dense'), 
                            ('mut-det', 'DETERMINANT MUTUAL-INFORMATION', 'mutually information dense (determinant-wise)'), ('mut-dia', 'DIAGONAL MUTUAL-INFORMATION', 'mutually information dense (diagonal-wise)'),\
                            ('prio-det', 'DETERMINANT PRIOR ENTROPY', 'previously information dense (determinant-wise)'), ('prio-dia', 'DIAGONAL PRIOR ENTROPY', 'previously information dense (diagonal-wise)'),\
                            ('marg-det', 'DETERMINANT MARGINAL ENTROPY', 'marginally information dense (determinant-wise)'), ('marg-dia', 'DIAGONAL MARGINAL ENTROPY', 'marginally information dense (diagonal-wise)'),\
                            ('post-det', 'DETERMINANT POSTERIOR ENTROPY', 'post information dense (determinant-wise)'), ('post-dia', 'DIAGONAL POSTERIOR ENTROPY', 'post information dense (diagonal-wise)'),\
                            ('exe_tim',  'EXECUTION_TIME', 'time consuming'), \
                            ('me', 'MEAN ERROR', 'error prone'), ('mae', 'MEAN ABSOLUTE ERROR', 'error prone'), ('rsme', 'ROOT MEAN SQUARE ERROR', 'error prone')]
                    
                    for name_tup in names:
                        name = name_tup[0]
                        print_name = name_tup[1]
                        descipt_name = name_tup[2]

                        if all_measurements[name][relevant_training_episodes['hi_'+name]['epi']] < all_measurements[name][x]:
                            lo_hi = 'hi_'
                            print(f"HIGHEST {print_name} UPDATE: Episode {x} is the most {descipt_name} episode so far with a total {print_name} of {round(all_measurements[name][x],5)}. "+\
                                f"This replaces Episode {relevant_training_episodes[lo_hi+name]['epi']} with a {print_name} of {round(all_measurements[name][relevant_training_episodes[lo_hi+name]['epi']],5)}.")
                            relevant_training_episodes[lo_hi+name]['epi'] = x
                            relevant_training_episodes[lo_hi+name]['loc'] = copy.deepcopy(self.starting_nodes[:len(self.world.agents)])
                            relevant_training_episodes[lo_hi+name]['exp'] = copy.deepcopy(agent_experiences)

                            if self.sim_params['same_init_neural_network']:
                                relevant_training_episodes[lo_hi+name]['net'] = copy.deepcopy([self.world.agents[0].net])
                            else:
                                relevant_training_episodes[lo_hi+name]['net'] = copy.deepcopy([agent.net for agent in self.world.agents])

                        elif all_measurements[name][relevant_training_episodes['lo_'+name]['epi']] > all_measurements[name][x]:
                            lo_hi = 'lo_'
                            print(f"LOWEST {print_name} UPDATE: Episode {x} is the least {descipt_name} episode so far with a total {print_name} of {round(all_measurements[name][x],5)}. "+\
                                f"This replaces Episode {relevant_training_episodes[lo_hi+name]['epi']} with a {print_name} of {round(all_measurements[name][relevant_training_episodes[lo_hi+name]['epi']],5)}.")
                            relevant_training_episodes[lo_hi+name]['epi'] = x
                            relevant_training_episodes[lo_hi+name]['loc'] = copy.deepcopy(self.starting_nodes[:len(self.world.agents)])
                            relevant_training_episodes[lo_hi+name]['exp'] = copy.deepcopy(agent_experiences)

                            if self.sim_params['same_init_neural_network']:
                                relevant_training_episodes[lo_hi+name]['net'] = copy.deepcopy([self.world.agents[0].net])
                            else:
                                relevant_training_episodes[lo_hi+name]['net'] = copy.deepcopy([agent.net for agent in self.world.agents])

                        else:
                            #  have  of {}, . "+\
                            print(f"NO {print_name} UPDATE. " +\
                                f"EPISODES {x}, {relevant_training_episodes['hi_'+name]['epi']} and {relevant_training_episodes['lo_'+name]['epi']} have {print_name}S of " +\
                                f"{round(all_measurements[name][x],5)}, {round(all_measurements[name][relevant_training_episodes['hi_'+name]['epi']],5)} and {round(all_measurements[name][relevant_training_episodes['lo_'+name]['epi']],5)} (respectively).")

                print(f"Starting Episode {x} training.")
                train_start = time.time()
                
                if epochs > 0 and batch_size > 0:
                    batches = min(budget, batch_size)

                    past_sample_indexes = random.sample( range( min(episode_experience_limit, x) ), min(sample_size, x) )
                    #print("sample_indexes", past_sample_indexes)
                    if group_exp:

                        inputs = []
                        outputs = []

                        #gathering a sample past experiences
                        for index in past_sample_indexes:
                            start = int((budget - shift)*index)
                            end = int(start + (budget - shift))
                            for agent in self.world.agents:
                                for exp_index in range(start, end):
                                    exp = data[agent.id][exp_index]
                                    inputs.append(exp.state.queue)
                                    outputs.append(exp.reward[reward_mechanism])
                        
                        #gathering recent experiences
                        for agent in self.world.agents:
                            for exp in agent.past[shift:]:
                                inputs.append(exp.state.queue)
                                outputs.append(exp.reward[reward_mechanism])

                        if len(self.world.agents) > 0:
                            inputs = self.world.agents[0].net.prep_training_data(inputs)
                            outputs = self.world.agents[0].net.prep_training_data(outputs)

                        if self.sim_params['same_init_neural_network']:
                            self.world.agents[0].net.train_net(input=inputs, output=outputs, epochs=epochs, batch_size=batches)

                        else:
                            for agent in self.world.agents:
                                agent.net.train_net(input=inputs, output=outputs, epochs=epochs, batch_size=batches)

                    else:
                        for agent in self.world.agents:
                            inputs = []
                            outputs = []

                            #gathering a sample of past experiences
                            for index in past_sample_indexes:
                                start = int((budget - shift)*index)
                                end = int(start + (budget - shift))
                                for exp_index in range(start, end):
                                    exp = data[agent.id][exp_index]
                                    inputs.append(exp.state.queue)
                                    outputs.append(exp.reward[reward_mechanism])

                            #gathering recent experiences
                            for exp in agent.past[shift:]:
                                inputs.append(exp.state.queue)
                                outputs.append(exp.reward[reward_mechanism])
                            
                            agent.net.train_net(input=inputs, output=outputs, epochs=epochs, batch_size=batches)

                for agent in self.world.agents:
                    data[agent.id].extend(agent.past[shift:])

                train_time = time.time() - train_start
                all_measurements['train_tim'][x] = train_time
                print(f"Episode {x} training complete. Total training time was {train_time} seconds.")

                scenario_time = time.time() - episode_start_time
                all_measurements['sen_tim'][x] = scenario_time
                print(f"Total elapsed time for Episode {x} was {scenario_time} seconds.")

                if ((x + 1) % self.tra_params['episode_save_frequency']) == 0:
                    save_num = int((x+1)/self.tra_params['episode_save_frequency'])
                    print(f"Performing intermediate save number {save_num}.")
                    current_directory = intermediate_save_directory + f"epi{x}/"
                    visual_results.mkdir([current_directory])
                    all_params = {'sim_params': self.sim_params, 'net_params': self.net_params, 'gau_params': self.gau_params, 'tra_params': self.tra_params}
                    agent_networks = current_directory + "agent_networks/"
                    visual_results.mkdir([agent_networks])
                    visual_results.save_multiple_networks(agent_networks+"_current/", [agent.net for agent in self.world.agents], x)
                    visual_results.save_networks(net_file_path=agent_networks, relevant_training_episodes=relevant_training_episodes, relevant_testing_results=None, delete_networks=False, post_tests=False, perform_training=True, episode=x)
                    pickle_content = {'relevant_training_episodes': relevant_training_episodes, 'relevant_testing_results': relevant_testing_results, 'all_measurements': all_measurements, 'all_params': all_params}
                    visual_results.pickle_write(current_directory, pickle_content)
                    print(f"Intermediate saving number {save_num} complete.")
                    
                self.world.epsilon = self.world.epsilon*epsilon_decay
        
        self.world.epsilon = -1.1

        if post_tests:
            
            if not new_starts:
                if len(self.starting_locations) == 0: 
                    starting_locations = relevant_training_episodes["hi_mut-dia"]['loc']
                else:
                    starting_locations = self.starting_locations
                for test_result_key in relevant_testing_results:
                    for traverse_method_key in relevant_testing_results[test_result_key]:
                        x = x + 1
                        print(f"\nConducting a {traverse_method_key} walk with the same starting nodes as the episode with {test_result_key}.")
                        self.model_testing(x, starting_locations, traverse_method_key, test_result_key, relevant_testing_results, relevant_training_episodes, all_measurements)
            
            else:
                predefined_random_starts = random.sample(range(len(self.sim_params['data_sim'])), self.sim_params['agent_num'])

                for test_result_key in relevant_testing_results:
                    for traverse_method_key in relevant_testing_results[test_result_key]:
                        starting_locations = []
                        if test_result_key != "random_starts":
                            print(f"\nConducting a {traverse_method_key} walk with the same starting nodes as the episode with {test_result_key}.")
                            starting_locations = relevant_training_episodes[test_result_key]['loc']
                        else:
                            print(f"\nConducting a {traverse_method_key} walk with new random starting points.")
                            starting_locations = predefined_random_starts
                        x = x + 1
                        self.model_testing(x, starting_locations, traverse_method_key, test_result_key, relevant_testing_results, relevant_training_episodes, all_measurements)
                      
        return relevant_training_episodes, relevant_testing_results, all_measurements

    def model_testing(self, x, starting_locations, traverse_method_key, test_result_key, relevant_testing_results, relevant_training_episodes, all_measurements):

        budget = self.tra_params['budget']
        new_starts = self.tra_params['new_starts']
        new_GP = self.tra_params['new_GP']

        self.sim_params['same_gpr'] = self.sim_params['initial_same_gpr']
        if traverse_method_key == "greedy":
            self.sim_params['same_gpr'] = False
        
        self.reset_world(new_starts=new_starts, new_GP=new_GP)
        start = time.time()
        saving_results = True

        if traverse_method_key == "explotation":
            self.world.action_method = traverse_method_key
            for y in range(budget-1):
                self.world.step()
            self.world.step(last_step=True)

        elif traverse_method_key == "random":
            self.world.action_method = traverse_method_key
            for y in range(budget-1):
                self.world.step()
            self.world.step(last_step=True)

        elif traverse_method_key == "greedy":
            self.world.action_method = traverse_method_key
            for y in range(budget-1):
                self.world.step()
            self.world.step(last_step=True)
            
        elif traverse_method_key == "good_teleport":
            if self.sim_params['agent_num'] <= 7:            
                self.world.teleport_step(first_step=True)
                for y in range(budget-2):
                    self.world.teleport_step()
                self.world.teleport_step(last_step=True)
            else:
                saving_results = False

        elif traverse_method_key == "bad_teleport":
            if self.sim_params['agent_num'] <= 7:
                self.world.teleport_step(first_step=True, good_teleports=False)
                for y in range(budget-2):
                    self.world.teleport_step(good_teleports=False)
                self.world.teleport_step(last_step=True, good_teleports=False)
            else:
                saving_results = False

        total_execution_time = time.time() - start
        if saving_results:
            print(f"Episode {x} exploration complete. Total exploration time was {total_execution_time} seconds.")
            
            agent_experiences = self.compute_metrics(all_measurements=all_measurements, x=x, total_execution_time=total_execution_time)

            if test_result_key in relevant_training_episodes:
                relevant_testing_results[test_result_key][traverse_method_key]['original_epi'] = relevant_training_episodes[test_result_key]['epi']
            relevant_testing_results[test_result_key][traverse_method_key]['epi'] = x
            relevant_testing_results[test_result_key][traverse_method_key]['loc'] = copy.deepcopy(starting_locations)
            relevant_testing_results[test_result_key][traverse_method_key]['exp'] = copy.deepcopy(agent_experiences)

            if self.sim_params['same_init_neural_network']:
                relevant_testing_results[test_result_key][traverse_method_key]['net'] = copy.deepcopy([self.world.agents[0].net])
            else:
                relevant_testing_results[test_result_key][traverse_method_key]['net'] = copy.deepcopy([agent.net for agent in self.world.agents])


    def compute_metrics(self, all_measurements, x, total_execution_time):

        variances = [0.0]*(len(self.world.agents))
        entropies = [0.0]*(len(self.world.agents))
        mutual_determinants = [0.0]*(len(self.world.agents))
        mutual_diagonals = [0.0]*(len(self.world.agents))
        prior_determinants = [0.0]*(len(self.world.agents))
        prior_diagonals = [0.0]*(len(self.world.agents))
        marginal_determinants = [0.0]*(len(self.world.agents))
        marginal_diagonals = [0.0]*(len(self.world.agents))
        posterior_determinants = [0.0]*(len(self.world.agents))
        posterior_diagonals = [0.0]*(len(self.world.agents))
        MEs = [0.0]*(len(self.world.agents))
        MAEs = [0.0]*(len(self.world.agents))
        RSMEs = [0.0]*(len(self.world.agents))

        agent_experiences = [None]*(len(self.world.agents))
        
        for agent in self.world.agents:
            id = agent.id
            for node in self.world.nodes:
                
                variances[id] = variances[id] + agent.past[-1].world_understanding['var'][node.id]

            for exp in agent.past:
                entropies[id] = entropies[id] + exp.reward['ent'][int(exp.action)]
                mutual_determinants[id] = mutual_determinants[id] + exp.reward['mut-det'][int(exp.action)]
                mutual_diagonals[id] = mutual_diagonals[id] + exp.reward['mut-dia'][int(exp.action)]
                prior_determinants[id] = prior_determinants[id] + exp.reward['info-det']['prioH'][int(exp.action)]
                prior_diagonals[id] = prior_diagonals[id] + exp.reward['info-dia']['prioH'][int(exp.action)]
                marginal_determinants[id] = marginal_determinants[id] + exp.reward['info-det']['margH'][int(exp.action)]
                marginal_diagonals[id] = marginal_diagonals[id] + exp.reward['info-dia']['margH'][int(exp.action)]
                posterior_determinants[id] = posterior_determinants[id] + exp.reward['info-det']['postH'][int(exp.action)]
                posterior_diagonals[id] = posterior_diagonals[id] + exp.reward['info-dia']['postH'][int(exp.action)]

            MEs[id], MAEs[id], RSMEs[id] = self.compute_errors(agent.past[-1].world_understanding['temp'])

            agent_experiences[id] = copy.deepcopy(agent.past)

        all_measurements['var'][x] = sum(variances)
        all_measurements['ent'][x] = sum(entropies)
        all_measurements['mut-det'][x] = sum(mutual_determinants)
        all_measurements['mut-dia'][x] = sum(mutual_diagonals)
        all_measurements['prio-det'][x] = sum(prior_determinants)
        all_measurements['prio-dia'][x] = sum(prior_diagonals)
        all_measurements['marg-det'][x] = sum(marginal_determinants)
        all_measurements['marg-dia'][x] = sum(marginal_diagonals)
        all_measurements['post-det'][x] = sum(posterior_determinants)
        all_measurements['post-dia'][x] = sum(posterior_diagonals)
        all_measurements['me'][x] = sum(MEs)
        all_measurements['mae'][x] = sum(MAEs)
        all_measurements['rsme'][x] = sum(RSMEs)
        all_measurements['exe_tim'][x] = total_execution_time

        all_measurements['standard_deviation']['var'][x] = np.std(variances)
        all_measurements['standard_deviation']['ent'][x] = np.std(entropies)
        all_measurements['standard_deviation']['mut-det'][x] = np.std(mutual_determinants)
        all_measurements['standard_deviation']['mut-dia'][x] = np.std(mutual_diagonals)
        all_measurements['standard_deviation']['prio-det'][x] = np.std(prior_determinants)
        all_measurements['standard_deviation']['prio-dia'][x] = np.std(prior_diagonals)
        all_measurements['standard_deviation']['marg-det'][x] = np.std(marginal_determinants)
        all_measurements['standard_deviation']['marg-dia'][x] = np.std(marginal_diagonals)
        all_measurements['standard_deviation']['post-det'][x] = np.std(posterior_determinants)
        all_measurements['standard_deviation']['post-dia'][x] = np.std(posterior_diagonals)
        all_measurements['standard_deviation']['me'][x] = np.std(MEs)
        all_measurements['standard_deviation']['mae'][x] = np.std(MAEs)
        all_measurements['standard_deviation']['rsme'][x] = np.std(RSMEs)
        all_measurements['standard_deviation']['exe_tim'][x] = 0.0

        all_measurements['individual_agent_values']['var'][x] = variances
        all_measurements['individual_agent_values']['ent'][x] = entropies
        all_measurements['individual_agent_values']['mut-det'][x] = mutual_determinants
        all_measurements['individual_agent_values']['mut-dia'][x] = mutual_diagonals
        all_measurements['individual_agent_values']['prio-det'][x] = prior_determinants
        all_measurements['individual_agent_values']['prio-dia'][x] = prior_diagonals
        all_measurements['individual_agent_values']['marg-det'][x] = marginal_determinants
        all_measurements['individual_agent_values']['marg-dia'][x] = marginal_diagonals
        all_measurements['individual_agent_values']['post-det'][x] = posterior_determinants
        all_measurements['individual_agent_values']['post-dia'][x] = posterior_diagonals
        all_measurements['individual_agent_values']['me'][x] = MEs
        all_measurements['individual_agent_values']['mae'][x] = MAEs
        all_measurements['individual_agent_values']['rsme'][x] = RSMEs
        all_measurements['individual_agent_values']['exe_tim'][x] = [total_execution_time/len(self.world.agents) for i in range(len(self.world.agents))]

        return agent_experiences


    def compute_errors(self, estimates):
        ME =   0.0
        MAE =  0.0
        RSME = 0.0
        n = float(len(self.world.nodes))

        for node in self.world.nodes:
            diff = (node.val - estimates[node.id])
            ME = ME + diff
            MAE = MAE + abs(diff)
            RSME = RSME + diff*diff
            
        return ME/n, MAE/n, math.sqrt(RSME/n)
