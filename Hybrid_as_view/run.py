from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import itertools as it


def search_config():
    config = {}
    config['probs'] = [0, .25, .5, .75]
    config['kernel_size'] = [[3, 3], [6, 6]]
    config['weights'] = [[0.5, 0.5], [0.2, 0.8]]
    flat = [[(k, v) for v in vs] for k, vs in config.items()]
    combinations = [dict(items) for items in it.product(*flat)]
    return combinations

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['hybrid']['switch_on'], config['batch_size'], **config['dataset'])

    combinations = search_config()
    for idx, c in enumerate(combinations):
        print('Trail %d/%d start ...' % (idx + 1, len(combinations) + 1))
        config['hybrid'] = c
        print(config['hybrid'])
        simclr = SimCLR(dataset, config)
        simclr.train()


if __name__ == "__main__":
    main()
