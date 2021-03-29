import numpy as np
import sklearn as sk
from .tools import Fixed_Queue, iterable, minMax, normalize, squared_euclidean
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import copy
from scipy import stats
import math
import sys
import random

class IPP_Experience:

    REWARD_KEYS = ('var', 'ent', 'info-dia', 'info-det', 'mut-det', 'mut-dia')
    UNDERSTANDING_KEYS = ('var', 'ent', 'info-dia', 'info-det', 'temp', 'mut-det', 'mut-dia')

    def __init__(self):
        self.state = None #Fixed_Queue, each element of the queue is a tuple of length 2 consisting of an IPP_Node.id and an action mean 
        self.action = 0.0 #float
        
        self.reward = {} #0.0 #float
        for key in self.REWARD_KEYS:
            if 'info' in key:
                self.reward[key] = {'prioH': 0.0, 'margH': 0.0, 'postH': 0.0}
            else:
                self.reward[key] = []

        self.world_understanding = {} #[] #List of list of floats
        for key in self.UNDERSTANDING_KEYS:
            if 'info' in key:
                self.world_understanding[key] = {'prioH': 0.0, 'margH': 0.0, 'postH': 0.0}
            else:
                self.world_understanding[key] = []


class IPP_Node:

    def __init__(self, id, grid_coor, real_coor, val):
        self.id = id
        self.grid_coor = grid_coor
        self.real_coor = real_coor
        self.val = val
        self.visitors = []
        self.valid_actions = {}


class IPP_Agent:

    def __init__(self, id):
        self.id = id #unique identifier
        self.net = None #PyTorch Neural Network
        self.gpr = None #Gaussian Process Regressor that will be updated over time
        self.curr = IPP_Experience() #IPP_Experience that models an agent's current situation
        self.past = [] #List of IPP_Experience
        self.loc = 0 #IPP_Node.id
        self.visits = [] #List of IPP_Node.id
        self.path = [] #Ordered List of visits
        self.previous_world_understanding = []

class IPP_Grid_World:

    def __init__(self):
        self.agents = []
        self.nodes = []
        self.actions = []
        self.SAMPLE = []

        self.true_comm_range = 0.0 #TODO: MAKE SURE THIS GET'S MODIFIED OR THIS WON'T WORK
        self.epsilon = 1.0
        self.SHIFT = sys.float_info.min #1 #sys.float_info.min #0 #sys.float_info.min
        self.PUNISH = math.log(sys.float_info.min) #0 #-800.0 #-1*sys.float_info.max #-10 #0.0 #-1*sys.float_info.max
        
        self.prevent_bad_actions = True
        self.PREVENTION = -1 * sys.float_info.max
        self.mean_field = True
        self.same_node_visits = True
        self.update_gpr = True

        self.action_method = 'training' #can be training, explotation, greedy, random
        self.reward_mechanism = 'ent' #can be 'var', 'ent', 'mut-dia', 'mut-det'
        self.mean_method = 'mean'
        self.no_neighbors_method = 'zero'

    def step(self, last_step=False):

        for agent in self.agents:
            agent.curr.reward = self.reward(agent)
            agent.curr.action = self.action_selection(agent)
            agent.loc, new_loc = self.next_cell(agent)
            self.update_cell_information(agent)
            if new_loc:
                self.update_agent_gpr(agent)
            agent.curr.world_understanding = self.agent_perspective(agent)
            agent.past.append(agent.curr)
        
        if not last_step:
            for agent in self.agents:
                agent.curr = IPP_Experience()

            assert self.true_comm_range > 0.0, "No communication will be conducted."
            #finding the agents within a communication range and computing the mean

            if self.mean_field:
                for agent in self.agents:
                    neighbor_actions = []
                    for agent_1 in self.agents:
                        if agent.id != agent_1.id:
                            if squared_euclidean(self.nodes[agent.loc].real_coor, self.nodes[agent_1.loc].real_coor) < self.true_comm_range:
                                neighbor_actions.append(agent_1.past[-1].action)
                    
                    agent.curr.state = copy.deepcopy(agent.past[-1].state)
                    if len(neighbor_actions) == 0:
                        if self.no_neighbors_method ==  'zero':
                            agent.curr.state.enq((self.nodes[agent.loc].real_coor[0], self.nodes[agent.loc].real_coor[1], 0.0))
                        elif self.no_neighbors_method == 'last_mean':
                            agent.curr.state.enq((self.nodes[agent.loc].real_coor[0], self.nodes[agent.loc].real_coor[1], 0.0))
                        elif self.no_neighbors_method == 'past_ave':
                            agent.curr.state.enq((self.nodes[agent.loc].real_coor[0], self.nodes[agent.loc].real_coor[1], 0.0))
                    else:
                        agent.curr.state.enq((self.nodes[agent.loc].real_coor[0], self.nodes[agent.loc].real_coor[1], self.mean_computation(neighbor_actions)))
            else:
                for agent in self.agents:
                    agent.curr.state = copy.deepcopy(agent.past[-1].state)
                    agent.curr.state.enq((self.nodes[agent.loc].real_coor[0], self.nodes[agent.loc].real_coor[1]))

    """
    Parameters
    agent: IPP_Agent

    Returns
    reward: list of floats
    """
    def reward(self, agent):
        reward = {}

        for key in IPP_Experience.REWARD_KEYS:
            if 'info' in key:
                reward[key] = {'prioH': [self.PUNISH]*len(self.actions), 'margH': [self.PUNISH]*len(self.actions), 'postH': [self.PUNISH]*len(self.actions)}
            elif key == 'var':
                reward[key] = [0.0]*len(self.actions)
            else:
                reward[key] = [self.PUNISH]*len(self.actions)

        node = self.nodes[agent.loc]

        for i in self.actions:
            if i in node.valid_actions:
                node_id = node.valid_actions[i]
                if (not agent.visits[node_id]) or (self.same_node_visits and len(self.nodes[node_id].visitors)==0) or (self.action_method == "greedy"):
                    
                    reward['var'][int(i)] = agent.gpr.variance(node_id)
                    reward['ent'][int(i)] = agent.gpr.entropy(node_id, self.SHIFT)

                    dia = agent.gpr.mutual_metrics(id=node_id, method='diagonal', shift=self.SHIFT)
                    reward['info-dia']['prioH'][int(i)] = dia[0]
                    reward['info-dia']['margH'][int(i)] = dia[1]
                    reward['info-dia']['postH'][int(i)] = dia[2]

                    #det = agent.gpr.mutual_metrics(id=id, method='determinant', shift=self.SHIFT)
                    reward['info-det']['prioH'][int(i)] = 0.0
                    reward['info-det']['margH'][int(i)] = 0.0
                    reward['info-det']['postH'][int(i)] = 0.0

                    reward['mut-det'][int(i)] = reward['info-det']['margH'][int(i)] - reward['info-det']['postH'][int(i)]
                    reward['mut-dia'][int(i)] = reward['info-dia']['margH'][int(i)] - reward['info-dia']['postH'][int(i)]
        return reward

    """
    """
    def teleport_reward(self, agent):
        reward = {}

        for key in IPP_Experience.REWARD_KEYS:
            if 'info' in key:
                reward[key] = {'prioH': [self.PUNISH], 'margH': [self.PUNISH], 'postH': [self.PUNISH]}
            elif key == 'var':
                reward[key] = [0.0]
            else:
                reward[key] = [self.PUNISH]
            
        reward['var'][0] = agent.previous_world_understanding['var'][agent.loc]
        reward['ent'][0] = agent.previous_world_understanding['ent'][agent.loc]
        
        reward['info-dia']['prioH'][0] = agent.previous_world_understanding['info-dia']['prioH'][agent.loc]
        reward['info-dia']['margH'][0] = agent.previous_world_understanding['info-dia']['margH'][agent.loc]
        reward['info-dia']['postH'][0] = agent.previous_world_understanding['info-dia']['postH'][agent.loc]
        
        reward['info-det']['prioH'][0] = 0.0 #agent.previous_world_understanding['info-det']['prioH'][agent.loc]
        reward['info-det']['margH'][0] = 0.0 #agent.previous_world_understanding['info-det']['margH'][agent.loc]
        reward['info-det']['postH'][0] = 0.0 #agent.previous_world_understanding['info-det']['postH'][agent.loc]
        
        reward['mut-dia'][0] = reward['info-dia']['margH'][0] - reward['info-dia']['postH'][0]
        reward['mut-det'][0] = reward['info-det']['margH'][0] - reward['info-det']['postH'][0]

        return reward

    """
    Parameters
    agent: IPP_Agent

    Returns
    action: int
    """
    def action_selection(self, agent):
        action = 0.0

        num = random.uniform(0.0, 1.0)
        if (self.action_method == 'training' and num <= self.epsilon) or self.action_method == 'random':
            action = np.random.random_integers(low=0, high=len(self.actions)-1)

        else:
            pots = None

            if self.action_method=='greedy':
                pots = copy.deepcopy(agent.curr.reward[self.reward_mechanism])
            else:
                pots = agent.net.forward([agent.curr.state.queue]).detach().tolist()[0]
                if self.prevent_bad_actions:
                    for i in range(len(self.actions)):
                        if float(i) not in self.nodes[agent.loc].valid_actions:
                            pots[i] = -1*sys.float_info.max
                        elif agent.visits[self.nodes[self.nodes[agent.loc].valid_actions[float(i)]].id] or (self.same_node_visits and len(self.nodes[self.nodes[agent.loc].valid_actions[float(i)]].visitors) > 0):
                            pots[i] = -1*sys.float_info.max
            action = np.argmax(pots)
            if iterable(action):
                action = action[0]
                
        return action

    """
    Parameters
    agent: IPP_Agent

    Returns
    id: int
    new_loc: boolean
    """
    def next_cell(self, agent):
        id = agent.loc
        node = self.nodes[agent.loc]
        action = agent.curr.action
        new_loc = False

        if action in node.valid_actions:
            id = node.valid_actions[action]
            new_loc = True

        return id, new_loc

    """
    Parameters
    agent: IPP_Agent

    Returns
    nothing
    """
    def update_cell_information(self, agent):
        agent.path.append(agent.loc)
        self.nodes[agent.loc].visitors.append(agent.id)
        if self.same_node_visits:
            for agent1 in self.agents:
                agent1.visits[agent.loc] = True
        else:
            agent.visits[agent.loc] = True

    """
    Parameters
    agent: IPP_Agent

    Returns
    nothing
    """
    def update_agent_gpr(self, agent):
        zB = [(agent.loc, self.nodes[agent.loc].val)]
        agent.gpr.update(zB)

    
    """
    Parameters
    neighbor_actions: List of floats

    Returns
    mean: float
    """
    def mean_computation(self, neighbor_actions):
        mean = 0.0        
        if self.mean_method == 'mean':
            mean = np.mean(neighbor_actions)
        elif self.mean_method == 'mode':
            mean = stats.mode(neighbor_actions).mode[0]
        elif self.mean_method == 'rounded-mean':
            mean = round(np.mean(neighbor_actions))
        elif self.mean_method == 'floor-mean':
            mean = math.floor(np.mean(neighbor_actions))
        elif self.mean_method == 'ceil-mean':
            mean = math.ceil(np.mean(neighbor_actions))
        return mean

    """
    Parameters
    agent: IPP_Node

    Returns
    global_estimate: List of floats
    estimate_variances: List of floats
    """
    def agent_perspective(self, agent, include_mut_dia=False, include_mut_det=False):

        world_understanding = {}
        for key in IPP_Experience.UNDERSTANDING_KEYS:
            if 'info' in key:
                world_understanding[key] = {'prioH': [0]*len(self.nodes), 'margH': [0]*len(self.nodes), 'postH': [0]*len(self.nodes)}
            else:
                world_understanding[key] = [0]*len(self.nodes)
        
        for node in self.nodes:
            id = node.id
            world_understanding['temp'][id] = agent.gpr.predict(id)

            if (not agent.visits[id]) or (self.same_node_visits and len(self.nodes[id].visitors)==0):
                world_understanding['var'][id] = agent.gpr.variance(id)
                world_understanding['ent'][id] = agent.gpr.entropy(id, self.SHIFT)

                if include_mut_dia:
                    dia = agent.gpr.mutual_metrics(id=id, method='diagonal', shift=self.SHIFT)
                    world_understanding['info-dia']['prioH'][id] = dia[0]
                    world_understanding['info-dia']['margH'][id] = dia[1]
                    world_understanding['info-dia']['postH'][id] = dia[2]
                    world_understanding['mut-dia'][id] = world_understanding['info-dia']['margH'][id] - world_understanding['info-dia']['postH'][id]
                
                if include_mut_det:
                    det = agent.gpr.mutual_metrics(id=id, method='determinant', shift=self.SHIFT)
                    world_understanding['info-det']['prioH'][id] = det[0]
                    world_understanding['info-det']['margH'][id] = det[1]
                    world_understanding['info-det']['postH'][id] = det[2]
                    world_understanding['mut-det'][id] = world_understanding['info-det']['margH'][id] - world_understanding['info-det']['postH'][id]
            else:
                world_understanding['var'][id] = 0.0
                world_understanding['ent'][id] = self.PUNISH

                if include_mut_dia:
                    world_understanding['info-dia']['prioH'][id] = self.PUNISH
                    world_understanding['info-dia']['margH'][id] = self.PUNISH
                    world_understanding['info-dia']['postH'][id] = self.PUNISH
                    world_understanding['mut-dia'][id] = self.PUNISH
                
                if include_mut_det:
                    world_understanding['info-det']['prioH'][id] = self.PUNISH
                    world_understanding['info-det']['margH'][id] = self.PUNISH
                    world_understanding['info-det']['postH'][id] = self.PUNISH
                    world_understanding['mut-det'][id] = self.PUNISH

        return world_understanding


    def teleport_step(self, first_step=False, last_step=False, good_teleports=True):

        if first_step:
            for agent in self.agents:
                agent.previous_world_understanding = self.agent_perspective(agent, include_mut_dia=True)

        for agent in self.agents:
            if good_teleports:
                potential_future_locations = copy.deepcopy(agent.previous_world_understanding[self.reward_mechanism])
                for node_id in range(len(self.nodes)):
                    if len(self.nodes[node_id].visitors) != 0:
                        potential_future_locations[node_id] = self.PUNISH
                agent.loc = np.argmax(potential_future_locations)
            else:
                agent.loc = 0
                worst_reward_above_zero = agent.previous_world_understanding[self.reward_mechanism][agent.loc]
                for node_id in range(len(self.nodes)):
                    cur = agent.previous_world_understanding[self.reward_mechanism][node_id]
                    if ((worst_reward_above_zero > cur and cur > 0.0) or (worst_reward_above_zero <= 0.0)) and len(self.nodes[node_id].visitors) == 0:
                        agent.loc = node_id
                        worst_reward_above_zero = cur                        
            agent.curr.reward = self.teleport_reward(agent)
            agent.curr.action = 0
            self.update_cell_information(agent)
            self.update_agent_gpr(agent)
            agent.curr.world_understanding = self.agent_perspective(agent, include_mut_dia=True)
            agent.previous_world_understanding = agent.curr.world_understanding
            agent.past.append(agent.curr)
        
        if not last_step:
            for agent in self.agents:
                agent.curr = IPP_Experience()

            for agent in self.agents:
                agent.curr.state = copy.deepcopy(agent.past[-1].state)
                agent.curr.state.enq((self.nodes[agent.loc].real_coor[0], self.nodes[agent.loc].real_coor[1], 0.0))