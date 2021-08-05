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
            for (xis, xjs, x_anchor), _ in train_loader:
                optimizer.zero_grad()
                xjs = get_hybrid_images(xjs, self.config['hybrid']['kernel_size'])
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                x_anchor = x_anchor.to(self.device)
                loss = self._step(model, xis, xjs, x_anchor)


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
