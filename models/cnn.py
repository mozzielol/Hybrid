import torch
import torch.nn as nn


class CNN(nn.Module):
    """CNN."""

    def __init__(self, dataset, multi_loss, base_model, out_dim):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        if dataset == 'cifar10':
            intermediate_dim = 4096
        elif dataset == 'stl10':
            intermediate_dim = 36864
        else:
            raise ValueError('Please define the intermediate dimension ...')
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(intermediate_dim, 1024),  # cifar10: 4096, stl10: 36864
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, out_dim)
        )
        if multi_loss:
            self.multi_fc_layer = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(intermediate_dim, 1024), # cifar10: 4096, stl10: 36864
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(512, out_dim),
                nn.Sigmoid()
            )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x

    def forward_multi(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.contiguous().view(x.size(0), -1)

        # fc layer
        x = self.multi_fc_layer(x)

        return x