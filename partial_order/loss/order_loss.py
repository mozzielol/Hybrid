import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Order_loss(torch.nn.Module):

    def __init__(self, dim, delta, use_cosine_similarity):
        super(Order_loss, self).__init__()
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.measure_similarity = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.delta = delta
        self.use_cosine_similarity = use_cosine_similarity
        self.projector = torch.nn.Linear(dim, dim, bias=False).to(device)
        self.mahalanobis_cov = torch.nn.Parameter(torch.tensor(torch.rand((dim, dim)), requires_grad=True))

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._metrics_similarity

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def _metrics_similarity(self, x, y):
        # return torch.sum(torch.square(self.projector(x) - self.projector(y)), dim=1)
        return self._mahalanobis_distance(self.projector(x), self.projector(y))

    def _mahalanobis_distance(self, u, v):
        cov = torch.mm(self.mahalanobis_cov, self.mahalanobis_cov.t())\
            .add_(torch.eye(len(self.mahalanobis_cov))).to(device)
        cov_inv = torch.inverse(cov)
        delta = u - v
        dist = torch.sqrt(torch.einsum('ij, ij -> i', torch.matmul(delta, cov_inv), delta))
        return torch.sum(dist)

    def forward(self, zis, zjs, z_anchor, single_pair=False):
        """
        :param single_pair:
        :param zis: similar to anchor
        :param zjs: dissimilar to anchor
        :param z_anchor: anchor image
        :return:
        """
        s1 = torch.diag(self.measure_similarity(zis, z_anchor)) if self.use_cosine_similarity else self.measure_similarity(zis, z_anchor)
        if single_pair:
            s2 = torch.diag(self.measure_similarity(zjs, z_anchor)) if self.use_cosine_similarity else self.measure_similarity(zjs, z_anchor)
            differences = torch.clamp(s2 - s1 + self.delta, min=0)
        else:
            if self.use_cosine_similarity:
                s2 = self.measure_similarity(zjs, z_anchor)
            else:
                s2 = []
                for sample in z_anchor:
                    s2.append(self.measure_similarity(zjs, sample))
                s2 = torch.stack(s2)
            # loss = -torch.sum(torch.log(torch.mean(torch.clamp(s2 - s1 + self.delta, min=1e-5, max=1.), dim=-1)))
            differences = torch.clamp(s2 - s1.reshape(-1, 1) + self.delta, min=0)
        loss = self.criterion(differences, torch.zeros_like(differences))
        return loss
