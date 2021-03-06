import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity, mse_loss):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.bce_criterion = torch.nn.MSELoss(reduction="mean") if mse_loss else torch.nn.BCELoss(reduction='sum')
        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.use_cosine_similarity = use_cosine_similarity
        self.mse_loss = mse_loss

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._distance_similarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

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

    def forward(self, zis, zjs, labels=None):
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)
        if labels is not None:
            labels = labels.to(self.device)
            logits = (similarity_matrix + 1) / 2 if self.use_cosine_similarity else similarity_matrix
            if self.mse_loss:
                loss = self.bce_criterion(logits, labels)
            else:
                loss = self.bce_criterion(logits + 1 - labels, torch.heaviside(labels, torch.tensor([0])))
        else:
            # filter out the scores from the positive samples
            l_pos = torch.diag(similarity_matrix, self.batch_size)
            r_pos = torch.diag(similarity_matrix, -self.batch_size)
            positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

            negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

            logits = torch.cat((positives, negatives), dim=1)
            logits /= self.temperature

            labels = torch.zeros(2 * self.batch_size).to(self.device).long()

            loss = self.ce_criterion(logits, labels)
        return loss / (2 * self.batch_size)
