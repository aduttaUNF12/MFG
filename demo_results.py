import os
import pickle

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def smooth_data(data, smoothing_weight=0.99):
    last = data[0]
    for i in range(1, data.shape[0]):
        data[i] = last * smoothing_weight + (1 - smoothing_weight) * data[i]
        last = data[i]

    return data


def main():
    budget_percent = st.sidebar.selectbox(label='Budget Percent', options=[0.4, 0.5])
    num_agents = st.sidebar.selectbox(label='#Agents', options=[3, 7, 10, 15, 20])
    queue_len = st.sidebar.selectbox(label='Queue length', options=[25, 50, 75])
    file_name = os.path.join(f'results/{budget_percent}_{num_agents}_{queue_len}.txt')
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
    st.pyplot(fig)

    fig, ax = plt.subplots()
    width = 0.35
    ax.bar(np.arange(num_agents) - width / 2, random_rewards, width / 2, label='Random')
    ax.bar(np.arange(num_agents), greedy_rewards, width / 2, label='Greedy')
    ax.bar(np.arange(num_agents) + width / 2, rewards[-1, :], width / 2, label='Model')
    ax.legend( bbox_to_anchor=(1.05, 1))
    ax.set_xticks(range(num_agents))
    ax.set_xticklabels([f'Agent {i}' for i in range(num_agents)])
    st.pyplot(fig)


if __name__ == '__main__':
    main()


