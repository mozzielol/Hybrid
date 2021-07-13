from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os
import ray
import torch
import random
from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import itertools


def tune_params(config):
    config['hybrid']['probs'] = tune.grid_search([0, .25, .5, .75])
    config['hybrid']['kernel_size'] = tune.grid_search([[3, 3], [6, 6]])
    config['hybrid']['weights'] = tune.grid_search([[0.5, 0.5], [0.2, 0.8]])
    return config


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    config['datapath'] = os.getcwd() + '/datasets'  # Please DO NOT change this
    dataset = DataSetWrapper(config['hybrid']['switch_on'], config['batch_size'], **config['dataset'])
    simclr = SimCLR(dataset, config)
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
            resources_per_trial={"cpu": 1, "gpu": 0},
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
