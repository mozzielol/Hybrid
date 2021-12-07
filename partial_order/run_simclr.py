from models.simclr_train import Simclr_train
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import itertools as it
from prettytable import PrettyTable
import numpy as np

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


def search_hybrid_config(config=None):
    config = {} if config is None else config
    config['kernel'] = ([5, 5])
    config['sigma'] = ([1.5, 1.5])
    config['delta'] = (0.1,)
    config['probability'] = ([1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1])
    config['weights'] = ([1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1],
                                [1, 0, 1,  1])
    config['pair'] = ('AB_A', 'AB_A1', 'AB_AB')
    config['temperature'] = (0.5, 1.5)
    config['learning_rate'] = (1e-3,)

    # Turn scalar to iterable 1-tuple for computing product combinations
    for k, v in config.items():
        if not isinstance(v, tuple):
            config[k] = (v,)

    flat = [[(k, v) for v in vs] for k, vs in config.items()]
    combinations = [dict(items) for items in it.product(*flat)]
    return combinations


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    combinations = search_hybrid_config(config['hybrid'])
    table = PrettyTable(['kernel size', 'delta', 'learning_rate', 'mix probability', 'triple_weights', 'pair', 'test acc'])
    keys = ['kernel', 'delta', 'learning_rate', 'probability', 'triple_weights', 'pair']
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
        simclr = Simclr_train(dataset, config)
        test_acc = simclr.train()
        row.append(test_acc)
        table.add_row(row)
    print(table)


if __name__ == "__main__":
    main()
