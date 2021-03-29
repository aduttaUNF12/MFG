import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from plots.utils import setup_figure


def smooth_data(data, smoothing_weight=0.99):
    last = data[0]
    for i in range(1, data.shape[0]):
        data[i] = last * smoothing_weight + (1 - smoothing_weight) * data[i]
        last = data[i]

    return data


def main():
    figure_name = setup_figure(name='convergence.pdf')

    budget_percent = 0.4  # 0.4 or 0.5
    num_agents = 7  # 3, 7, 10, 15, or 20
    queue_len = 25  # 25, 50 or 75
    file_name = os.path.join(f'../results/{budget_percent}_{num_agents}_{queue_len}.txt')
    with open(file_name, 'rb') as f:
        rewards = pickle.load(f)

    random_rewards = rewards[5001]
    greedy_rewards = rewards[5002]

    if len(np.array(rewards[:5003]).shape) == 2:
        rewards = np.array(rewards[:5003])
        # teleport_rewards1 = rewards[5003]
        # teleport_rewards2 = rewards[5004]
        rewards = rewards[:5000]
    else:
        for i in range(len(rewards)):
            if len(rewards[i]) != num_agents:
                break
        rewards = np.array(rewards[:i])


    fig, ax = plt.subplots()

    for agent in range(num_agents):
        ax.plot(smooth_data(rewards[:, agent]), label=f'Agent {agent}')
        # ax.plot(5000, random_rewards[agent], marker='x', label=f'Random Agent {agent}')
        # ax.plot(5000, greedy_rewards[agent], marker='x', label=f'Greedy Agent {agent}')

    ax.legend( bbox_to_anchor=(1.05, 1))

    lgd = plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(fname=figure_name, dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
