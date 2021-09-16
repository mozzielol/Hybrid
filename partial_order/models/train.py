import torch
from models.resnet_simclr import ResNetSimCLR
import torch.nn.functional as F
import numpy as np
from eval_model import _load_stl10, eval_trail
from util.util import get_device
import timeit
from loss.order_loss import Order_loss
from data_aug.hybrid import get_hybrid_images, GaussianSmoothing
from torch.utils.tensorboard import SummaryWriter
import os
from data_aug.seq_loader import Sequence

torch.manual_seed(0)


class Order_train(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = get_device()
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        try:
            os.makedirs(os.path.join(config['log_dir'], 'checkpoints'))
        except FileExistsError:
            pass
        self.loss_func = Order_loss(config['model']['out_dim'], config['hybrid']['delta'], **config['loss'])
        self.dataset = dataset

    def _step(self, model, xis, xjs, x_anchor):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        r_anchor, z_anchor = model(x_anchor)

        # normalize projection feature vectors
        zis = F.normalize(zis.unsqueeze(0), dim=1)
        zjs = F.normalize(zjs.unsqueeze(0), dim=1)
        z_anchor = F.normalize(z_anchor.unsqueeze(0), dim=1)

        loss = self.loss_func(zis, zjs, z_anchor, True)
        return loss

    def _step_by_indices(self, model, xis, x_anchor):
        ris, zis = model(xis)  # [N,C]
        r_anchor, z_anchor = model(x_anchor)
        zis = F.normalize(zis, dim=1)
        z_anchor = F.normalize(z_anchor, dim=1)
        loss = self.loss_func(zis, z_anchor, z_anchor)
        return loss

    def train(self, config=None):
        if config is not None:
            self.config = config

        train_loader = Sequence(10)
        self.X_train, self.y_train, self.X_test, self.y_test = train_loader.train_test_split(5, 2000)

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
                pre_anchor, anchor, post_anchor = pre_anchor.to(self.device), anchor.to(self.device), post_anchor.to(self.device)
                loss = 0
                for idx in range(len(pre_anchor) - 1, 0, -1):
                    loss += self._step(model, pre_anchor[idx].unsqueeze(0), pre_anchor[idx - 1].unsqueeze(0), anchor)
                for idx in range(len(post_anchor) - 1):
                    loss += self._step(model, post_anchor[idx].unsqueeze(0), post_anchor[idx + 1].unsqueeze(0), anchor)

                loss = loss.to(self.device)
                loss.backward()
                print(loss.item())
                optimizer.step()
                n_iter += 1
            if epoch_counter // self.config['eval_every_n_epochs'] == 0:
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

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(self.config['log_dir'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model
