from models.train import Order_train
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import itertools as it
from prettytable import PrettyTable


def search_config():
    config = {}
    config['probs'] = [.5, 1.5]
    config['kernel_size'] = [[3, 3], [6, 6]]
    config['weights'] = [[0.5, 0.5], [0.2, 0.8]]
    flat = [[(k, v) for v in vs] for k, vs in config.items()]
    combinations = [dict(items) for items in it.product(*flat)]
    return combinations

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['hybrid']['switch_on'], config['batch_size'], **config['dataset'])

    combinations = search_config()
    table = PrettyTable(['hybrid probability', 'kernel size', 'weights', 'test acc'])
    keys = ['probs', 'kernel_size', 'weights']
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
