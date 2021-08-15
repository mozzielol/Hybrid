import torch
from models.resnet_simclr import ResNetSimCLR
import torch.nn.functional as F
import numpy as np
from eval_model import _load_stl10, eval_trail
from util.util import get_device
import timeit
from loss.order_loss import Order_loss
from ray import tune
import os
from data_aug.hybrid import get_hybrid_images


torch.manual_seed(0)


class Order_train(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = get_device()
        # self.writer = SummaryWriter(log_dir='runs/' + config['log_dir'])
        self.loss_func = Order_loss(config['hybrid']['delta'], **config['loss'])
        self.dataset = dataset
        self.X_train, self.y_train = _load_stl10("train")
        self.X_test, self.y_test = _load_stl10("test")

    def _step(self, model, xis, xjs, x_anchor):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        r_anchor, z_anchor = model(x_anchor)

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        z_anchor = F.normalize(z_anchor, dim=1)

        loss = self.loss_func(zis, zjs, z_anchor)
        return loss

    def train(self, config=None):
        if config is not None:
            self.config = config

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        # model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

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
            for (A1, x_anchor), _ in train_loader:
                if counter == 1 and self.config['testing_phase']:
                    break
                counter += 1
                optimizer.zero_grad()
                w_A1_B, w_AB_C, w_A1_AB, w_AB_B = self.config['hybrid']['triple_weights']
                AB, B, C = get_hybrid_images(x_anchor, self.config['hybrid']['kernel_size'])
                A1, AB, B, C = A1.to(self.device), AB.to(self.device), B.to(self.device), C.to(self.device)
                loss = 0
                if w_A1_B > 0:
                    loss += w_A1_B * self._step(model, A1, B, x_anchor)
                if w_AB_C > 0:
                    loss += w_AB_C * self._step(model, AB, C, x_anchor)
                if w_A1_AB > 0:
                    loss += w_A1_AB * self._step(model, A1, AB, x_anchor)
                if w_AB_B > 0:
                    loss += w_AB_B * self._step(model, AB, B, x_anchor)

                # if n_iter % self.config['log_every_n_steps'] == 0:
                #     self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                loss = loss.to(self.device)
                loss.backward()

                optimizer.step()
                n_iter += 1
                print(epoch_counter, loss.item())
            train_acc, test_acc = eval_trail(model, self.X_train, self.y_train, self.X_test, self.y_test, self.config, self.device)
            final_test_acc = test_acc
            print('Train acc: %.3f, Test acc: %.3f' % (train_acc, test_acc))
            if epoch_counter >= 10:
                scheduler.step()

            # tune.report(loss=best_valid_loss, accuracy=test_acc)
            stop = timeit.default_timer()
            print('Epoch', epoch_counter, 'Time: ', stop - start)

        return final_test_acc

            # warmup for the first 10 epochs

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, x_anchor), _ in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                x_anchor = x_anchor.to(self.device)
                loss = self._step(model, xis, xjs, x_anchor)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
