import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
from data_aug.hybrid import generate_pairs_with_hybrid_images
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
import os
import shutil
import sys
from eval_model import _load_stl10, eval_trail
from ray import tune
import timeit


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class SimCLR(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        # self.writer = SummaryWriter(log_dir='runs/' + config['log_dir'])
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        self.X_train, self.y_train = _load_stl10("train")
        self.X_test, self.y_test = _load_stl10("test")

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, label=None):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs, label)
        return loss

    def _step_hybrid(self, model, x):
        hybrid_paris, sim_matrix = generate_pairs_with_hybrid_images(x,
                                                                     self.config['hybrid']['kernel_size'],
                                                                     self.config['hybrid']['weights'])
        # get the representations and the projections
        xis = hybrid_paris[:len(x)].to(self.device)
        xjs = hybrid_paris[len(x):].to(self.device)
        return self._step(model, xis, xjs, sim_matrix)

    def train(self, config=None):
        if config is not None:
            self.config = config

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        # model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        # model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        # _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        final_test_acc = 0.
        print('Training start ...')

        for epoch_counter in range(self.config['epochs']):
            start = timeit.default_timer()
            for (xis, xjs, x_ori), _ in train_loader:
                optimizer.zero_grad()
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                if self.config['hybrid']['probs'] > 1:
                    loss = self._step_hybrid(model, x_ori.to(self.device))
                else:
                    loss = self._step(model, xis, xjs)
                    if np.random.random_sample() < self.config['hybrid']['probs']:
                        loss += self._step_hybrid(model, x_ori)

                # if n_iter % self.config['log_every_n_steps'] == 0:
                #     self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                loss = loss.to(self.device)
                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            train_acc, test_acc = eval_trail(model, self.X_train, self.y_train, self.X_test, self.y_test, self.config, self.device)
            final_test_acc = test_acc
            print('Train acc: %.3f, Test acc: %.3f' % (train_acc, test_acc))
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    # torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                valid_n_iter += 1
            if epoch_counter >= 10:
                scheduler.step()

            if self.config['tune_params']:
                tune.report(loss=best_valid_loss, accuracy=test_acc)
            stop = timeit.default_timer()
            print('Epoch', epoch_counter, 'Time: ', stop - start)
        if self.config['tune_params']:
            with tune.checkpoint_dir(n_iter) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
        return final_test_acc

            # warmup for the first 10 epochs

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs, _), _ in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                loss = self._step(model, xis, xjs)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
