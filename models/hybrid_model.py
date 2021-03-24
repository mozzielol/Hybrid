import torch
from models.load_model import model_loader
from torch.utils.tensorboard import SummaryWriter
from data.hybrid import get_hybrid_images
import os
import shutil
import numpy as np
from ray import tune
from data.dataloader import Dataloader

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class Hybrid_Clf(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        # self.writer = SummaryWriter()
        self.dataset = dataset
        self.multi_criterion = torch.nn.BCELoss(reduce='sum')
        self.single_criterion = torch.nn.CrossEntropyLoss()

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, x, y, model):
        sin_y = y
        sin_x = x

        sin_x = sin_x.to(self.device)
        sin_y = sin_y.to(self.device)

        if self.config['loss']['multi_loss'] and self.config['loss']['multi_loss_weight'] > 0:
            # Single loss
            single_logits = model(sin_x)
            loss = self.single_criterion(single_logits, sin_y) * (1 - self.config['loss']['multi_loss_weight'])

            # Multi loss
            x, mul_y = get_hybrid_images(x, (3, 3), y, self.config['model']['out_dim'])
            x = x.to(self.device)
            mul_y = mul_y.to(self.device) * self.config['loss']['multi_loss_weight']
            multi_logits = model.forward_multi(x)
            loss += self.multi_criterion(multi_logits, mul_y)
        else:
            single_logits = model(sin_x)
            loss = self.single_criterion(single_logits, sin_y)

        return loss

    def train(self, config=None):
        if config is not None:
            self.config = config
            self.dataset = Dataloader(config['datapath'], config['batch_size'], **config['dataset'])
        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = model_loader(self.config['dataset']['name'], self.config['loss']['multi_loss'], **self.config["model"]).to(self.device)
        print(model)
        model = self._load_pre_trained_weights(model)

        # optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config['lr'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        # model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        # if self.config['save_model']:
        #     _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = self._step(x, y, model)
                # if n_iter % self.config['log_every_n_steps'] == 0:
                #     self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                loss.backward()

                optimizer.step()
                if n_iter % self.config['print_every_n_iters'] == 0:
                    print('Epoch {}/{}, training loss: {:.4f}'
                          .format(epoch_counter, self.config['epochs'], loss.item()))
                    test_acc = self._validate(model, valid_loader, return_acc=True)
                    print('Test accuracy is ', test_acc)
                n_iter += 1
            print('')
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    # if self.config['save_model']:
                    #     torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                # self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            # self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)
            test_acc = self._validate(model, valid_loader, return_acc=True)
            print('Test accuracy is ', test_acc)

            if config['tune_params']:
                    tune.report(loss=best_valid_loss, accuracy=test_acc)

        if config['tune_params']:
            with tune.checkpoint_dir(n_iter) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, return_acc=False):
        # validation steps
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for x, y in valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                if return_acc:
                    outputs = model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                else:
                    loss = self._step(x, y, model)
                    valid_loss += loss.item()
                    counter += 1
        if return_acc:
            return 100 * correct / total
        valid_loss /= counter
        model.train()
        return valid_loss
