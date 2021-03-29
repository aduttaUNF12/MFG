import pickle, sys, os
import numpy as np
from matplotlib import pyplot as plt

# sys.path.append(os.path.realpath('../MFG-MARL-IPP'))

base_dir = r""  # directory where the file is stored
file_name = r"0.pkl"  # name of the file itself
file_path = '/run/user/1000/gvfs/sftp:host=newton,user=skhodadadeh/home/skhodadadeh/MFG-MARL-IPP/data/_outputs/Large.Tests.36/mixed_variables/'
file_path = os.path.join(file_path, os.listdir(file_path)[0], 'pickle.pkl')

print(file_path)



def get_results(file_path):
    metric = "mut-dia"
    methods = ["model", "greedy", "random"]

    pickle_contents = None
    with open(file_path, 'rb') as file:
        pickle_contents = pickle.load(file)

    results = {"model": None, "greedy": None, "random": None}

    training_results = pickle_contents['all_measurements'][metric][:pickle_contents['all_params']['tra_params']['episodes']]
    results["model"] = np.max(training_results)
    print(training_results.index(np.max(training_results)))

    epi = pickle_contents['relevant_testing_results']["standard_starts"]["greedy"]['epi']
    results["greedy"] = pickle_contents['all_measurements'][metric][epi]

    epi = pickle_contents['relevant_testing_results']["standard_starts"]["random"]['epi']
    results["random"] = pickle_contents['all_measurements'][metric][epi]

    return results

results = get_results(file_path)
print(results)
