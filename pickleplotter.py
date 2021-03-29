'''
Created on Mar 20, 2021

@author: Jeffery Wolbert
'''
import pickle
import numpy as np
from matplotlib import pyplot as plt

def data_from_exp(agents_experience, mean_field):
    paths_with_means = []
    heatmaps = []

    for agent in agents_experience:
        path = []
        heatmap = []
        for exp in agent:
            pos = exp.state.last()
            x = int(pos[0])
            y = int(pos[1])
            angle = 0 if not mean_field else pos[2]*45
            
            path.append((x, y, angle))
            heatmap.append([exp.world_understanding['var']])#this needs to be changed in order to get different data types
            #heatmap.append([exp.world_understanding['ent']])

        paths_with_means.append(path)
        heatmaps.append(heatmap)

    return paths_with_means, heatmaps

##########load pickle file##########
ests = pickle.load( open( "/run/user/1000/gvfs/sftp:host=newton,user=skhodadadeh/home/skhodadadeh/MFG-MARL-IPP/data/_outputs/Large.Tests.5/mixed_variables/scenario.0.1611694287.0894256/pickle.pkl", "rb" ) )

#one of these keys can be used in line 38 to get data of a certain relevant episode
print(ests['relevant_training_episodes'].keys())

#paths , maps = data_from_exp(ests['relevant_testing_results']['standard_starts']['explotation']['exp'], True)
paths , maps = data_from_exp(ests['relevant_training_episodes']['lo_var']['exp'], True)

all_params = ests['all_params']
dims = all_params['sim_params']['world_dim']

newlist = []
newlist.append([])
maps = np.array(maps)
#remove channel axis
maps = np.squeeze(maps,axis=2)

zeros = 0
min = 1
avg = 0
for x in maps[0][-1]:
    if x == 0:
        zeros += 1
    elif min > x:
        min = x
    avg += x


#some normalization
avg = avg / (dims[0] + dims[1])
avg = avg / (dims[0] + dims[1])
avg = avg / (dims[0] + dims[1])

min -= avg
    
#switch zeros to normalized values
for x in range(len(maps[0][-1])):
    if maps[0][-1][x] == 0:
        maps[0][-1][x] = min
        


last = np.reshape(maps[0][-1],(dims[0],dims[1]))

#print(last)

print("episode",ests['relevant_training_episodes']['lo_var']['epi'])
print("visited cells",zeros)

plt.imshow(last) 
plt.show()


quit()
