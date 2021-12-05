from models.train import Order_train, Sequence_train, Mix_train
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import itertools as it
from prettytable import PrettyTable
import numpy as np
from data_aug.seq_loader import Sequence

"""
symbols:
    A - Anchor image
    A1 - augmented A
    B - second component of hybrid image
    C - rest of images in the batch except A & B
    
The weights of triples follow the order:
'A1_B', 'AB_C', 'AB_A1', 'AB_B', 'A1_C'
"""


def set_triplet_weights():
    triplet_weights = (np.linspace(0, 1, 2),) * 5
    weights = [list(items) for items in it.product(*triplet_weights)]
    weights.remove([0, 0, 0, 0, 0])
    return weights


def search_hybrid_config():
    config = {}
    config['kernel_size'] = [(5, 5)]
    config['sigma'] = [(1.5, 1.5)]
    config['delta'] = [0.1]
    config['probability'] = [(1, 0, 0), (1, 1.0, 0), (1, 0, 1.0)]
    config['triple_weights'] = [(0, 0, 0, 0, 1), (0, 0, 1, 0, 1), (0, 0, 1, 1, 1), (1, 0, 0, 0, 1), (1, 0, 1, 0, 1),
                                (1, 0, 1, 1, 1)]
    config['learning_rate'] = [1e-3]
    # config['use_cosine_similarity'] = [False, True]
    flat = [[(k, v) for v in vs] for k, vs in config.items()]
    combinations = [dict(items) for items in it.product(*flat)]
    return combinations


def search_mix_config():
    config = {}
    config['delta'] = [0.1]
    config['probability'] = [(1.0, 0.5), (0.5, 1.0)]
    config['learning_rate'] = [1e-3]
    # config['use_cosine_similarity'] = [False, True]
    flat = [[(k, v) for v in vs] for k, vs in config.items()]
    combinations = [dict(items) for items in it.product(*flat)]
    return combinations


def search_sequence_config():
    config = {}
    config['delta'] = [0.1]
    config['learning_rate'] = [1e-3]
    config['duration'] = [20]
    config['interval'] = [2]
    flat = [[(k, v) for v in vs] for k, vs in config.items()]
    combinations = [dict(items) for items in it.product(*flat)]
    return combinations


def main(experiment='sequence'):
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    if experiment == 'sequence':
        combinations = search_sequence_config()
        table = PrettyTable(['delta', 'learning_rate', 'duration', 'interval'])
        keys = ['delta', 'learning_rate', 'duration', 'interval']
    elif experiment == 'mix':
        combinations = search_mix_config()
        table = PrettyTable(['learning_rate', 'Mix probabilities', 'test acc'])
        keys = ['learning_rate', 'probability']
    elif experiment == 'hybrid':
        combinations = search_hybrid_config()
        table = PrettyTable(['kernel size', 'delta', 'learning_rate', 'mix probability', 'triple_weights', 'test acc'])
        keys = ['kernel_size', 'delta', 'learning_rate', 'probability', 'triple_weights']
    for idx, c in enumerate(combinations):
        config['log_dir'] = './runs/'
        for key in c.keys():
            config['log_dir'] += str(c[key]) + '_'
        row = []
        print('Trail %d/%d start ...' % (idx + 1, len(combinations)))
        print(c)
        config['hybrid'] = c
        for k in keys:
            row.append(c[k])
        if experiment == 'sequence':
            simclr = Sequence_train(Sequence(c['duration'], c['interval']), config)
        elif experiment == 'mix':
            simclr = Mix_train(dataset, config)
        elif experiment == 'hybrid':
            simclr = Order_train(dataset, config)
        test_acc = simclr.train()
        row.append(test_acc)
        table.add_row(row)
    print(table)


if __name__ == "__main__":
    main('hybrid')
