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
    C - different image from A & B
    
The weights of triples follow the order:
'A1_B', 'AB_C', 'AB_A1', 'AB_B'
"""


def set_triple_weights():
    w_A1_B, w_AB_C, w_AB_A1, w_AB_B = np.linspace(0, 1, 2), np.linspace(0, 1, 2), np.linspace(0, 1, 2), np.linspace(0,
                                                                                                                    1,
                                                                                                                    2)
    weightes = [list(items) for items in it.product(*[w_A1_B, w_AB_C, w_AB_A1, w_AB_B])]
    weightes.remove([0, 0, 0, 0])
    return weightes


def search_config():
    config = {}
    config['kernel_size'] = [[3, 3]]
    config['delta'] = [0.01]
    config['triple_weights'] = set_triple_weights()
    flat = [[(k, v) for v in vs] for k, vs in config.items()]
    combinations = [dict(items) for items in it.product(*flat)]
    return combinations


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    combinations = search_config()
    table = PrettyTable(['kernel size', 'delta', 'triple_weights', 'test acc'])
    keys = ['kernel_size', 'delta', 'triple_weights']
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
