from models.hybrid_model import Hybrid_Clf
from data.dataloader import Dataloader
import yaml
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os
import ray
import torch
import random


def tune_params(config):
    config['lr'] = tune.grid_search([5e-4])  # tune.loguniform(1e-4, 1e-1)
    config['batch_size'] = tune.grid_search([64, 128])
    config['loss']['multi_loss_weight'] = tune.grid_search([0, .25, .5, .75])
    config['dataset']['augmentation'] = tune.grid_search([True, False])
    config['dataset']['standardization'] = tune.grid_search([True, False])
    config['hybrid']['kernel'] = tune.grid_search([3, 9, 15])
    return config


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    config['datapath'] = os.getcwd() + '/datasets'  # Please DO NOT change this
    dataset = Dataloader(config['datapath'], config['batch_size'], **config['dataset'])
    simclr = Hybrid_Clf(dataset, config)
    if config['tune_params']:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        ray.init()
        config = tune_params(config)
        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=100,
            grace_period=10,
            reduction_factor=3)
        reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["loss", "accuracy", "training_iteration"])
        result = tune.run(
            simclr.train,
            resources_per_trial={"cpu": 1, "gpu": 1},
            config=config,
            scheduler=scheduler,
            progress_reporter=reporter,
            checkpoint_at_end=True,
            num_samples=1,
            name='hybrid_tune')
        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))
    else:
        # Reproducibility
        seed = 2021
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        simclr.train(config=config)


if __name__ == "__main__":
    main()
