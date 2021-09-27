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


def search_config():
    config = {}
    config['delta'] = [0.1]
    config['learning_rate'] = [1e-3]
    config['duration'] = [10]
    config['interval'] = [2]
    flat = [[(k, v) for v in vs] for k, vs in config.items()]
    combinations = [dict(items) for items in it.product(*flat)]
    return combinations


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    combinations = search_config()
    table = PrettyTable(['delta', 'learning_rate', 'test acc'])
    keys = [ 'delta', 'learning_rate']
    for idx, c in enumerate(combinations):
        config['log_dir'] = 'runs/'
        for key in c.keys():
            config['log_dir'] += str(c[key]) + '_'
        row = []
        print('Trail %d/%d start ...' % (idx + 1, len(combinations)))
        print(c)
        config['sequence'] = c
        for k in keys:
            row.append(c[k])
        simclr = Order_train(dataset, config)
        test_acc = simclr.train()
        row.append(test_acc)
        table.add_row(row)
    print(table)


if __name__ == "__main__":
    main()
