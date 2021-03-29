import matplotlib.pyplot as plt

from plots.utils import get_data, setup_figure


def main():
    figure_name = setup_figure(name='queue_length_impact.pdf')

    df = get_data()
    counter = 0
    for budget_percent in (0.4, 0.5):
        for agent_num in (3, 7, 10, 15, 20):
            current_df = df[(df['budget'] == budget_percent) & (df['agents'] == agent_num)]
            label = None if counter != 0 else 'RL model'
            plt.plot(
                current_df['queue_percent'],
                current_df['reward'],
                label=f'Budget: {budget_percent}, #Agents: {agent_num}'
                # label=label,
                # color='b'
            )
            # label = None if counter != 0 else 'Greedy'
            # plt.plot(
            #     current_df['queue_percent'],
            #     current_df['greedy'],
            #     # label=f'Budget: {budget_percent}, #Agents: {agent_num}'
            #     label=label,
            #     color='r'
            # )
            # label = None if counter != 0 else 'Random'
            # plt.plot(
            #     current_df['queue_percent'],
            #     current_df['random'],
            #     # label=f'Budget: {budget_percent}, #Agents: {agent_num}'
            #     label=label,
            #     color='g'
            # )
            # counter += 1

    plt.xticks(current_df['queue_percent'])
    plt.xlabel('Queue Length')
    plt.ylabel('Reward')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(fname=figure_name, dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
