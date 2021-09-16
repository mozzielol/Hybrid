from models.train import Order_train
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
    triplet_weights = (np.linspace(0, 1, 2), ) * 5
    weights = [list(items) for items in it.product(*triplet_weights)]
    weights.remove([0, 0, 0, 0, 0])
    return weights


def search_config():
    config = {}
    config['kernel_size'] = [5, 15, 31, 47]
    config['delta'] = [0.1]
    config['triple_weights'] = [(0, 0, 0, 0, 1), (0, 0, 1, 0, 1), (0, 0, 1, 1, 1), (1, 0, 0, 0, 1), (1, 0, 1, 0, 1), (1, 0, 1, 1, 1)] # set_triplet_weights()
    config['learning_rate'] = [1e-3]
    flat = [[(k, v) for v in vs] for k, vs in config.items()]
    combinations = [dict(items) for items in it.product(*flat)]
    return combinations


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    combinations = search_config()
    table = PrettyTable(['kernel size', 'delta', 'learning_rate', 'triple_weights', 'test acc'])
    keys = ['kernel_size', 'delta', 'learning_rate', 'triple_weights']
    for idx, c in enumerate(combinations):
        row = []
        print('Trail %d/%d start ...' % (idx + 1, len(combinations) + 1))
        print(c)
        config['hybrid'] = c
        for k in keys:
            row.append(c[k])
        simclr = Order_train(dataset, config)
        test_acc = simclr.train()
        row.append(test_acc)
        table.add_row(row)
    print(table)


if __name__ == "__main__":
    main()
