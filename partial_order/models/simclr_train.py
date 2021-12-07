from .train import Order_train
from loss import NTXentLoss
import torch.nn.functional as F
from models.resnet_simclr import ResNetSimCLR
import torch
import numpy as np
from data_aug.hybrid import *
import timeit
from eval_model import _load_stl10, eval_trail
import os


class Simclr_train(Order_train):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        self.loss_func = NTXentLoss(config['batch_size'], config['hybrid']['temperature'],
                                    config['loss']['use_cosine_similarity'])

    def _step(self, model, xis, xjs):
        xis, xjs = xis.to(self.device), xjs.to(self.device)
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]
        if self.config['loss']['use_cosine_similarity']:
            # normalize projection feature vectors
            zis = F.normalize(zis, dim=1)
            zjs = F.normalize(zjs, dim=1)
        loss = self.loss_func(zis, zjs)
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

        def get_pairs(A1, A):
            if self.config['hybrid']['pair'] == 'AB_A':
                return A
            elif self.config['hybrid']['pair'] == 'AB_A1':
                return A1
            elif self.config['hybrid']['pair'] == 'AB_AB':
                return None

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
            for (A1, A2, x_anchor), _ in train_loader:
                if counter == 1 and self.config['testing_phase']:
                    break

                A1, A2, x_anchor = A1.to(self.device), A2.to(self.device), x_anchor.to(self.device)
                counter += 1
                loss = 0
                optimizer.zero_grad()

                w_hybrid, w_cut_mix, w_mixup, w_simclr = self.config['hybrid']['weights']
                composite_kwargs = (
                    {'method': 'hybrid', 'kernel': self.config['hybrid']['kernel'],
                     'sigma': self.config['hybrid']['sigma'], 'weight': w_hybrid},
                    {'method': 'cutmix', 'beta': self.config['hybrid']['cutmix_beta'], 'weight': w_cut_mix},
                    {'method': 'mixup', 'ratio_offset': self.config['hybrid']['mixup_ratio_offset'],
                     'weight': w_mixup}
                )

                config_probs = np.array(self.config['hybrid']['probability'])
                rand_probs = np.random.rand(len(config_probs))

                # When all composite methods are not selected in this epoch, use the one with max random prob
                if all(rand_probs - config_probs) > 0:
                    rand_probs[rand_probs.argmax()] += 1

                if w_simclr > 0:
                    pair = get_pairs(A2, x_anchor)
                    if pair is None:
                        pair = A2
                    loss += w_simclr * self._step(model, A1, pair)
                    if self.config['multi_step_update']:
                        backprop(optimizer, loss)

                for rand_prob, config_prob, kwargs in zip(rand_probs, config_probs, composite_kwargs):
                    if rand_prob > config_prob:
                        continue

                    if any(self.config['hybrid']['triple_weights'][:-1]) > 0:
                        if self.config['multi_step_update']:
                            loss = 0
                            optimizer.zero_grad()

                        AB, _, _ = get_composite_images(x_anchor, **kwargs)
                        AB= AB.to(self.device)
                        pair = get_pairs(A2, x_anchor)
                        if pair is None:
                            pair, _, _ = get_composite_images(x_anchor, **kwargs)
                        loss += kwargs['weight'] * self._step(model, AB, pair)

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
            for (xis, xjs, x_anchor), _ in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                x_anchor = x_anchor.to(self.device)
                loss = self._step(model, xis, xjs, x_anchor)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss