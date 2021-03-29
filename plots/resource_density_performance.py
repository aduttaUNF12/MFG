import matplotlib.pyplot as plt

from plots.utils import get_data, setup_figure


def main():
    figure_name = setup_figure(name='resource_density_performance.pdf')

    fig, axes = plt.subplots(1, 3)
    df = get_data()
    for i, queue_length in enumerate((25, 50, 75)):
        current_df = df[df['queue_percent'] == queue_length]
        current_df['rs'] = current_df['budget'] * current_df['agents'] / current_df['env_size'] ** 2
        current_df = current_df.sort_values(by='rs')

        axes[i].plot(
            current_df['rs'],
            current_df['reward'],
            label=f'Total Reward'
        )
        axes[i].plot(
            current_df['rs'],
            current_df['random'],
            label=f'Random'
        )
        axes[i].plot(
            current_df['rs'],
            current_df['greedy'],
            label=f'Greedy'
        )

        axes[i].set_xticks([])
        axes[i].set_xlabel('Resource Density')
    axes[0].set_ylabel('Reward')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(fname=figure_name, dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
