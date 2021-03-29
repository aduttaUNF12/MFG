import os
import pickle
from tqdm import tqdm

BASE_ADDRESS = '/run/user/1000/gvfs/sftp:host=newton,user=skhodadadeh/home/skhodadadeh/MFG-MARL-IPP/data/_outputs'

EXPERIMENT_FILE_PATH = {
    0.4: {
        3: {
            25: 30,
            50: 31,
            75: 32,
        },
        7: {
            25: 24,
            50: 25,
            75: 26,
        },
        10: {
            25: 18,
            50: 19,
            75: 20,
        },
        15: {
            25: os.path.join(
                BASE_ADDRESS,
                'Large.Tests.4/mixed_variables/scenario.0.1611692515.99967/intermediate_saves/epi4499/pickle.pkl',
            ),
            50: 5,
            75: 6,
        },
        20: {
            25: os.path.join(
                BASE_ADDRESS,
                'Large.Tests.14/mixed_variables/scenario.0.1613758596.0368855/intermediate_saves/epi2499/pickle.pkl',
            ),
            50: os.path.join(
                BASE_ADDRESS,
                'Large.Tests.15/mixed_variables/scenario.0.1613758695.9087696/intermediate_saves/epi3999/pickle.pkl',
            ),
            75: 16,
        },

    },
    0.5: {
        3: {
            25: 33,
            50: 34,
            75: 35,
        },
        7: {
            25: 27,
            50: 28,
            75: 29,
        },
        10: {
            25: 21,
            50: 22,
            75: 23,
        },
        15: {
            25: 10,
            50: 8,
            75: 7,
        },
        20: {
            25: os.path.join(
                BASE_ADDRESS,
                'Large.Tests.11/mixed_variables/scenario.0.1613758318.162394/intermediate_saves/epi3499/pickle.pkl',
            ),
            50: os.path.join(
                BASE_ADDRESS,
                'Large.Tests.12/mixed_variables/scenario.0.1613758425.6972826/intermediate_saves/epi4999/pickle.pkl',
            ),
            75: 13,
        },

    },

}

GREEDY_RANDOM_PICKLE = {
    os.path.join(
        BASE_ADDRESS,
        'Large.Tests.4/mixed_variables/scenario.0.1611692515.99967/intermediate_saves/epi4499/pickle.pkl',
    ): 'scenario.0.1615047852.2168562/pickle.pkl',
    os.path.join(
        BASE_ADDRESS,
        'Large.Tests.14/mixed_variables/scenario.0.1613758596.0368855/intermediate_saves/epi2499/pickle.pkl',
    ): 'scenario.0.1615048629.8639379/pickle.pkl',
    os.path.join(
        BASE_ADDRESS,
        'Large.Tests.15/mixed_variables/scenario.0.1613758695.9087696/intermediate_saves/epi3999/pickle.pkl',
    ): 'scenario.0.1615049601.5392258/pickle.pkl',
    os.path.join(
        BASE_ADDRESS,
        'Large.Tests.11/mixed_variables/scenario.0.1613758318.162394/intermediate_saves/epi3499/pickle.pkl',
    ): 'scenario.0.1615051468.6924114/pickle.pkl',
    os.path.join(
        BASE_ADDRESS,
        'Large.Tests.12/mixed_variables/scenario.0.1613758425.6972826/intermediate_saves/epi4999/pickle.pkl',
    ): 'scenario.0.1615069670.6069844/pickle.pkl',

}


def extract_experiment_file_path(budget_percent, agent_num, queue_len):
    exp_id = EXPERIMENT_FILE_PATH[budget_percent][agent_num][queue_len]
    if isinstance(exp_id, int):
        file_path = f'/run/user/1000/gvfs/sftp:host=newton,user=skhodadadeh/home/skhodadadeh/' \
                    f'MFG-MARL-IPP/data/_outputs/Large.Tests.{exp_id}/mixed_variables/'
        file_path = os.path.join(file_path, os.listdir(file_path)[0], 'pickle.pkl')
        return file_path, None
    return exp_id, exp_id.replace('pickle.pkl', GREEDY_RANDOM_PICKLE[exp_id])


def main():
    os.makedirs('results', exist_ok=True)

    progbar = tqdm(total=2 * 5 * 3)
    for budget_percent in (0.4, 0.5):
        for agent_num in (3, 7, 10, 15, 20):
            for queue_len in (25, 50, 75):
                result_file_name = os.path.join('results', f'{budget_percent}_{agent_num}_{queue_len}.txt')
                if os.path.exists(result_file_name):
                    progbar.update(1)
                file_path, greedy_random_pkl = extract_experiment_file_path(budget_percent, agent_num, queue_len)

                if greedy_random_pkl is not None:
                    with open(greedy_random_pkl, 'rb') as file:
                        gr_pickle_contents = pickle.load(file)
                        gr_pickle_contents = gr_pickle_contents['all_measurements']['individual_agent_values']['mut-dia']
                else:
                    continue

                with open(file_path, 'rb') as file:
                    pickle_contents = pickle.load(file)

                rewards = pickle_contents['all_measurements']['individual_agent_values']['mut-dia']
                rewards[5001] = gr_pickle_contents[1]
                rewards[5002] = gr_pickle_contents[2]
                print(len(rewards))

                with open(result_file_name, 'wb') as f:
                    pickle.dump(rewards, f)
                progbar.update(1)


if __name__ == '__main__':
    main()
