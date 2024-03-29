import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Order_loss(nn.Module):

    def __init__(self, dim, delta, use_cosine_similarity, loss_function, additional_penalty=True):
        super(Order_loss, self).__init__()
        self.measure_similarity = self._get_similarity_function(use_cosine_similarity)
        self.criterion = nn.BCELoss(reduction='sum') if loss_function.lower() == 'bce' else nn.MSELoss(reduction='sum')
        self.delta = delta
        self.use_cosine_similarity = use_cosine_similarity
        self.loss_function = loss_function
        self.projector = nn.Linear(dim, dim, bias=False).to(device)
        self.mahalanobis_cov = nn.Parameter(torch.rand((dim, dim)), requires_grad=True)
        self.additional_penalty = additional_penalty

    def _get_similarity_function(self, use_cosine_similarity):
        return nn.CosineSimilarity() if use_cosine_similarity else self._mahalanobis_distance

    @staticmethod
    def _dot_similarity(x, y):
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _metrics_similarity(self, x, y):
        # return torch.sum(torch.square(self.projector(x) - self.projector(y)), dim=1)
        return self._mahalanobis_distance(x, y)

    def _mahalanobis_distance(self, u, v):
        """
        Compute the mahalanobis distance between each element-pairs in u and v.
        :param u: a batch of vectors in matrix format, shape Num x Dim
        :param v: a batch of vectors in matrix format, shape Num x Dim
        :return: a vector of mahalanobis distances: dist[i] = dist(u[i], v[i])
        """
        cov = torch.mm(self.mahalanobis_cov, self.mahalanobis_cov.t())\
            .add_(torch.eye(len(self.mahalanobis_cov))).to(device)
        cov_inv = torch.inverse(cov)
        delta = u - v
        dist = torch.sqrt(torch.einsum('ij, ij -> i', torch.matmul(delta, cov_inv), delta))
        return dist

    def forward(self, zis, zjs, z_anchor, negative_pairs=None):
        """
        Function to calculate the loss
        Parameters
        ----------
        zis: torch Tensor, (N * F) latent representation similar to anchor
        zjs: torch Tensor, (N * F) latent representation DISSIMILAR to anchor
        z_anchor: torch Tensor, (N * F) latent representation of anchor
        negative_pairs: torch Tensor, 256 * 256 bool tensors, row i is the
            negative pairs of image i
        Returns
        -------
        """
        s1 = self.measure_similarity(zis, z_anchor)  # shape (N, )

        if negative_pairs is None:
            s2 = self.measure_similarity(zjs, z_anchor)  # shape (N, )
            differences = torch.clamp(s2 - s1 + self.delta, min=0)
        else:
            s2 = []
            for anchor, negative_sample_indices in zip(z_anchor, negative_pairs):
                negative_samples = z_anchor[negative_sample_indices]
                anchor_repeat = anchor.repeat(len(negative_samples), 1)
                s2.append(self.measure_similarity(anchor_repeat, negative_samples))
            s2 = torch.stack(s2)  # shape (N, N') in which N': number of negative samples for each anchor
            differences = torch.clamp(s2 - s1.unsqueeze(1) + self.delta, min=0)

        if not self.use_cosine_similarity and self.loss_function.lower() == 'bce':  # BCE requires inputs in [0, 1]
            differences = differences - differences.min()
            differences = differences / differences.max()

        differences = torch.clamp(differences, min=0, max=1)

        loss = self.criterion(differences, torch.zeros_like(differences))
        if self.additional_penalty:
            loss += nn.BCELoss(reduction='sum')(s1, torch.ones_like(s1))
            loss += nn.BCELoss(reduction='sum')(s2, torch.zeros_like(s2))
        return loss


class Sequence_loss(Order_loss):
    def __init__(self, dim, delta, use_cosine_similarity, loss_function):
        super().__init__(dim, delta, use_cosine_similarity, loss_function)

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