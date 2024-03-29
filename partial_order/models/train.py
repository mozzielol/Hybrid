import torch
from torch.utils.tensorboard import SummaryWriter
from models.resnet_simclr import ResNetSimCLR
import torch.nn.functional as F
import numpy as np
from eval_model import _load_stl10, eval_trail
from util.util import get_device
import timeit
from loss.order_loss import Order_loss, Sequence_loss
from data_aug.hybrid import *
import os
from pathlib import Path

torch.manual_seed(0)


class Order_train(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = get_device()
        Path(os.path.join(config['log_dir'], 'checkpoints')).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        self.loss_func = Order_loss(config['model']['out_dim'], config['hybrid']['delta'], **config['loss'])
        self.dataset = dataset
        self.X_train, self.y_train = _load_stl10("train")
        self.X_test, self.y_test = _load_stl10("test")

    def _step(self, model, xis, xjs, x_anchor):
        xis, xjs = xis.to(self.device), xjs.to(self.device)
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        r_anchor, z_anchor = model(x_anchor)
        if self.config['loss']['use_cosine_similarity']:
            # normalize projection feature vectors
            zis = F.normalize(zis, dim=1)
            zjs = F.normalize(zjs, dim=1)
            z_anchor = F.normalize(z_anchor, dim=1)

        loss = self.loss_func(zis, zjs, z_anchor, None)
        return loss

    def _step_by_indices(self, model, xis, x_anchor, negative_indicators):
        _, zis = model(xis)  # [N,C]
        _, z_anchor = model(x_anchor)
        if self.config['loss']['use_cosine_similarity']:
            zis = F.normalize(zis, dim=1)
            z_anchor = F.normalize(z_anchor, dim=1)
        loss = self.loss_func(zis, None, z_anchor, negative_indicators)
        return loss

    def train(self, config=None):

        def backprop(opt, ls):
            """
            Back-propagation step
            :param opt: optimizer
            :param ls: loss
            :return: N/A
            """
            ls.to(self.device)
            ls.backward()
            opt.step()

        if config is not None:
            self.config = config

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        if self.config["resume_saved_runs"]:
            model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), self.config['hybrid']['learning_rate'],
                                     weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        final_test_acc = 0.
        print('Training start ...')

        for epoch_counter in range(self.config['epochs']):
            start = timeit.default_timer()
            counter = 0
            if epoch_counter == 1 and self.config['testing_phase']:
                break
            for (A1, _, x_anchor), _ in train_loader:
                if counter == 1 and self.config['testing_phase']:
                    break

                A1, x_anchor = A1.to(self.device), x_anchor.to(self.device)
                counter += 1
                loss = 0
                optimizer.zero_grad()

                w_A1_B, w_AB_C, w_A1_AB, w_AB_B, w_A1_C = self.config['hybrid']['triple_weights']
                composite_kwargs = (
                    {'method': 'hybrid', 'kernel': self.config['hybrid']['kernel'],
                     'sigma': self.config['hybrid']['sigma']},
                    {'method': 'cutmix', 'beta': self.config['hybrid']['cutmix_beta']},
                    {'method': 'mixup', 'ratio_offset': self.config['hybrid']['mixup_ratio_offset']}
                )

                config_probs = np.array(self.config['hybrid']['probability'])
                rand_probs = np.random.rand(len(config_probs))

                # When all composite methods are not selected in this epoch, use the one with max random prob
                if all(rand_probs - config_probs) > 0:
                    rand_probs[rand_probs.argmax()] += 1

                if w_A1_C > 0:
                    loss += w_A1_C * self._step_by_indices(model, A1, x_anchor, torch.logical_not(torch.eye(len(A1))))
                    if self.config['multi_step_update']:
                        backprop(optimizer, loss)

                for rand_prob, config_prob, kwargs in zip(rand_probs, config_probs, composite_kwargs):
                    if rand_prob > config_prob:
                        continue

                    if any(self.config['hybrid']['triple_weights'][:-1]) > 0:
                        if self.config['multi_step_update']:
                            loss = 0
                            optimizer.zero_grad()

                        AB, B, negative_indicators = get_composite_images(x_anchor, **kwargs)
                        AB, B = AB.to(self.device), B.to(self.device)
                        if w_A1_B > 0:
                            loss += w_A1_B * self._step(model, A1, B, x_anchor)
                        if w_AB_C > 0:
                            loss += w_AB_C * self._step_by_indices(model, AB, x_anchor, negative_indicators)
                        if w_A1_AB > 0:
                            loss += w_A1_AB * self._step(model, A1, AB, x_anchor)
                        if w_AB_B > 0:
                            loss += w_AB_B * self._step(model, AB, B, x_anchor)

                        if self.config['multi_step_update']:
                            backprop(optimizer, loss)

                if not self.config['multi_step_update']:
                    backprop(optimizer, loss)

                if self.config['verbose']:
                    print(loss.item())
                n_iter += 1

            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                torch.save(model.state_dict(), os.path.join(self.config['log_dir'], 'checkpoints', 'model.pth'))
                train_acc, test_acc = eval_trail(model, self.X_train, self.y_train, self.X_test, self.y_test,
                                                 self.config, self.device)
                final_test_acc = test_acc
                print('Train acc: %.3f, Test acc: %.3f' % (train_acc, test_acc))
            if epoch_counter >= 10:
                scheduler.step()

            # tune.report(loss=best_valid_loss, accuracy=test_acc)
            stop = timeit.default_timer()
            print('Epoch', epoch_counter, 'Time: ', stop - start)

        return final_test_acc

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, _, x_anchor), _ in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                x_anchor = x_anchor.to(self.device)
                loss = self._step(model, xis, xjs, x_anchor)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(self.config['log_dir'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model


class Mix_train(Order_train):

    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        self.X_train, self.y_train = _load_stl10("train")
        self.X_test, self.y_test = _load_stl10("test")

    def train(self, config=None):
        if config is not None:
            self.config = config

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        if self.config["resume_saved_runs"]:
            model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), self.config['hybrid']['learning_rate'],
                                     weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        n_iter = 0
        final_test_acc = 0.
        print('Training start ...')

        for epoch_counter in range(self.config['epochs']):
            start = timeit.default_timer()
            counter = 0
            if epoch_counter == 1 and self.config['testing_phase']:
                break
            for (_, _, x_anchor), _ in train_loader:
                x_anchor = x_anchor.to(self.device)
                if counter == 1 and self.config['testing_phase']:
                    break
                counter += 1
                optimizer.zero_grad()
                w_cutmix, w_mix_up = self.config['hybrid']['probability']

                loss = 0
                if np.random.rand() < w_cutmix:
                    cutmix_image, src_b = compose_cutmix_image(x_anchor)
                    loss += self._step(model, cutmix_image, src_b, x_anchor)
                if np.random.rand() < w_mix_up:
                    mix_up_image, src_b = compose_mixup_image(x_anchor)
                    loss += self._step(model, mix_up_image, src_b, x_anchor)

                loss = loss.to(self.device)
                loss.backward()
                optimizer.step()
                n_iter += 1
                if self.config['verbose']:
                    print(loss.item())

            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                torch.save(model.state_dict(), os.path.join(self.config['log_dir'], 'checkpoints', 'model.pth'))
                train_acc, test_acc = eval_trail(model, self.X_train, self.y_train, self.X_test, self.y_test,
                                                 self.config, self.device)
                final_test_acc = test_acc
                print('Train acc: %.3f, Test acc: %.3f' % (train_acc, test_acc))
            if epoch_counter >= 10:
                scheduler.step()

            # tune.report(loss=best_valid_loss, accuracy=test_acc)
            stop = timeit.default_timer()
            print('Epoch', epoch_counter, 'Time: ', stop - start)

        return final_test_acc


class Sequence_train(Order_train):

    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        self.loss_func = Sequence_loss(config['model']['out_dim'], config['hybrid']['delta'], **config['loss'])

    def train(self, config=None):
        if config is not None:
            self.config = config

        train_loader = self.dataset
        self.X_train, self.y_train, self.X_test, self.y_test = train_loader.train_test_split(5, 20)

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), self.config['hybrid']['learning_rate'],
                                     weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        final_test_acc = 0.
        print('Training start ...')

        for epoch_counter in range(self.config['epochs']):
            start = timeit.default_timer()
            counter = 0
            if epoch_counter == 1 and self.config['testing_phase']:
                break
            for (pre_anchor, anchor, post_anchor), _ in train_loader:
                if counter == 1 and self.config['testing_phase']:
                    break
                counter += 1
                optimizer.zero_grad()
                pre_anchor, anchor, post_anchor = pre_anchor.to(self.device), anchor.to(self.device), post_anchor.to(
                    self.device)
                loss = 0
                for idx in range(len(pre_anchor) - 1, 0, -1):
                    loss += self._step(model, pre_anchor[idx].unsqueeze(0), pre_anchor[idx - 1].unsqueeze(0), anchor)
                for idx in range(len(post_anchor) - 1):
                    loss += self._step(model, post_anchor[idx].unsqueeze(0), post_anchor[idx + 1].unsqueeze(0), anchor)

                loss = loss.to(self.device)
                loss.backward()
                if self.config['verbose']:
                    print(loss.item())
                optimizer.step()
                n_iter += 1
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                torch.save(model.state_dict(), os.path.join(self.config['log_dir'], 'checkpoints', 'model.pth'))
                train_acc, test_acc = eval_trail(model, self.X_train, self.y_train, self.X_test, self.y_test,
                                                 self.config, self.device)
                final_test_acc = test_acc
                print('Train acc: %.3f, Test acc: %.3f' % (train_acc, test_acc))
            if epoch_counter >= 10:
                scheduler.step()

            # tune.report(loss=best_valid_loss, accuracy=test_acc)
            stop = timeit.default_timer()
            print('Epoch', epoch_counter, 'Time: ', stop - start)

        return final_test_acc
