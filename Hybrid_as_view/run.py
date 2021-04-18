from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import itertools


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['hybrid']['switch_on'], config['batch_size'], **config['dataset'])

    hybrid_switch = [True, False]
    return_origin = [True, False]
    augmentation = [True, False]
    use_cosine_similarity = [True, False]

    combinations = list(itertools.product(hybrid_switch, return_origin, augmentation, use_cosine_similarity))
    for idx, c in enumerate(combinations):
        print('Trail %d/%d start ...' % (idx + 1, len(combinations) + 1))
        config['log_dir'] = 'trail_' + str(idx)
        config['hybrid']['switch_on'] = c[0]
        config['hybrid']['return_origin'] = c[1]
        config['dataset']['augmentation'] = c[2]
        config['loss']['use_cosine_similarity'] = c[3]
        simclr = SimCLR(dataset, config)
        simclr.train()


if __name__ == "__main__":
    main()
