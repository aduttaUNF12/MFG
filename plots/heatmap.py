import pickle, sys, os
import numpy as np
from matplotlib import pyplot as plt

from visuals.visual_results import generate_video


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

    return pickle_contents


def main():
    pickle_content = get_results(file_path)

    generate_video(
        'temp_video/',
        pickle_content,
        perform_training=True,
        gen_training=True,
        gen_testing=True,
        agent_perspective_count=0
    )


if __name__ == '__main__':
    main()
