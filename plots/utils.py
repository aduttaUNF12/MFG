import os

import matplotlib
import pandas


def get_data(cols=None):
    FILE_PATH = '../results.csv'
    df = pandas.read_csv(FILE_PATH)
    if cols is None:
        df = df[
            [
                'budget',
                'agents',
                'queue_percent',
                'env_size',
                'exploration time',
                'training time',
                'elapsed time',
                'reward',
                'random',
                'greedy'
            ]
        ]
    else:
        df = df[cols]
    return df


def setup_figure(name):
    font = {
        'family': 'arial',
        #     'weight' : 'bold',
        'size': 12
    }
    matplotlib.rc('font', **font)

    root_figures_folder = 'figures'
    os.makedirs(root_figures_folder, exist_ok=True)
    figure_name = os.path.join(root_figures_folder, name)

    return figure_name
