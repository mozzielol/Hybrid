from models.hybrid_model import Hybrid_Clf
from data.dataloader import Dataloader
import yaml


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = Dataloader(config['batch_size'], **config['dataset'])

    simclr = Hybrid_Clf(dataset, config)
    simclr.train()

if __name__ == "__main__":
    main()
