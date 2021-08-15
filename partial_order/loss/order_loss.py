import torch
import numpy as np


class Order_loss(torch.nn.Module):

    def __init__(self, delta, use_cosine_similarity):
        super(Order_loss, self).__init__()
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.measure_similarity = self._get_similarity_function(use_cosine_similarity)
        self.delta = delta

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._distance_similarity

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

    def _distance_similarity(self, x, y, targets=None):
        similarity_matrix = torch.zeros([x.size(0), x.size(0)])
        for x_idx, x_ele in enumerate(x):
            for y_idx, y_ele in enumerate(y):
                if x_idx == y_idx:
                    similarity_matrix[x_idx, y_idx] = 1.
                else:
                    if targets is None:
                        similarity_matrix[x_idx, y_idx] = torch.exp(-torch.sqrt(torch.mean(torch.square(x - y))))
                    else:
                        similarity_matrix[x_idx, y_idx] = torch.exp(-torch.sqrt(torch.mean(torch.square(x - y - targets[x_idx, y_idx]))))

        return similarity_matrix

    def forward(self, zis, zjs, z_anchor):
        """
        :param zis: similar to anchor
        :param zjs: dissimilar to anchor
        :param z_anchor: anchor image
        :return:
        """
        s1 = self.measure_similarity(zis, z_anchor)
        s2 = self.measure_similarity(zjs, z_anchor)
        loss = torch.mean(torch.clamp(s2 - s1 + self.delta, min=0))
        return loss
