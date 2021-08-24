import torch

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on:", device)
    return device
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrastive_feature_extractor(torch.nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(in_dims, 512),
            nn.ReLU(),
            nn.Linear(512, in_dims),
            nn.Tanh()
        )
        self.extractor = nn.Sequential(
            nn.Linear(in_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x, select=False):
        x = F.gumbel_softmax(self.selector(x), hard=True) * x if select else x
        latent = self.extractor(x)
        return latent, self.projector(latent)

def compute_loss(y_pred, y_true, delta=0.2):
    loss = 0
    def svm_loss(idx):
        assert idx in [0, 1], 'out of range'
        mask = y_true[:, idx] > 0
        return torch.clamp(y_true[mask][:, idx] - y_pred[mask][:, idx] + delta, min = 0) \
            + torch.clamp( - y_true[mask][:, idx] + y_pred[mask][:, 1 - idx] - delta, min = 0)
    for i in range(2):
        loss += torch.mean(svm_loss(i))
    return loss

model = Contrastive_feature_extractor(334)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion ="""