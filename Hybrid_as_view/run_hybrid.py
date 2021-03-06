from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import itertools


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['hybrid']['switch_on'], config['batch_size'], **config['dataset'])
    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
