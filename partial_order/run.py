from models.train import Order_train
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import itertools as it
from prettytable import PrettyTable


def search_config():
    config = {}
    config['kernel_size'] = [[3, 3], [6, 6]]
    config['delta'] = [0.01, 0.1]
    flat = [[(k, v) for v in vs] for k, vs in config.items()]
    combinations = [dict(items) for items in it.product(*flat)]
    return combinations

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    combinations = search_config()
    table = PrettyTable(['kernel size', 'delta', 'test acc'])
    keys = ['kernel_size', 'delta']
    for idx, c in enumerate(combinations):
        row = []
        print('Trail %d/%d start ...' % (idx + 1, len(combinations) + 1))
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
