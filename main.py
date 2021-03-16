from models.hybrid_model import Hybrid_Clf
from data.dataloader import Dataloader
import yaml
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os


def tune_params(config):
    config['train']['lr'] = tune.loguniform(1e-4, 1e-1)
    return config


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = Dataloader(os.getcwd() + '/datasets', config['batch_size'], **config['dataset'])
    simclr = Hybrid_Clf(dataset, config)
    if config['tune_params']:
        max_num_epochs = 1
        config = tune_params(config)
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["loss", "accuracy", "training_iteration"])
        result = tune.run(
            simclr.train,
            resources_per_trial={"cpu": 1, "gpu": 0},
            config=config,
            scheduler=scheduler,
            progress_reporter=reporter,
            checkpoint_at_end=True)
        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))
    else:
        simclr.train()

if __name__ == "__main__":
    main()
