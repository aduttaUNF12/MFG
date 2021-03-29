import matplotlib.pyplot as plt

from plots.utils import get_data, setup_figure


def main():
    figure_name = setup_figure(name='training_time.pdf')

    df = get_data()
    for budget_percent in (0.4, 0.5):
        for queue_length in (25, 50, 75):
            current_df = df[(df['budget'] == budget_percent) & (df['queue_percent'] == queue_length)]
            plt.plot(
                current_df['agents'],
                current_df['elapsed time'],
                label=f'Budget: {budget_percent}, Queue Length: {queue_length}'
            )

    plt.xticks(current_df['agents'])
    plt.xlabel('#Agents')
    plt.ylabel('Training Episode Time (seconds)')
    plt.legend()
    plt.savefig(fname=figure_name, dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
