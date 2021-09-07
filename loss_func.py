"""
This file implement a series of clustering loss.
We try to determine the different impact for clustering with different loss combination.
"""
from abc import ABC

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BaseLoss:

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class DECLoss(BaseLoss, ABC):
    """
    Deep embedding clustering.
    Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis."
    International conference on machine learning. PMLR, 2016.
    """

    def __init__(self):
        super(DECLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')

    def __call__(self, logist, **kwargs):
        Q = self.target_distribution(logist).detach()
        loss = self.criterion(logist.log(), Q) / logist.shape[0]
        return loss

    def target_distribution(self, logist) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (logist ** 2) / torch.sum(logist, 0)
        return (weight.t() / torch.sum(weight, 1)).t()


class DDCLoss(BaseLoss, ABC):
    """
    Michael Kampffmeyer et al. "Deep divergence-based approach to clustering"
    """

    def __init__(self, num_cluster, epsilon=1e-9, rel_sigma=0.15, device='cpu'):
        """

        :param epsilon:
        :param rel_sigma: Gaussian kernel bandwidth
        """
        super(DDCLoss, self).__init__()
        self.epsilon = epsilon
        self.rel_sigma = rel_sigma
        self.device = device
        self.num_cluster = num_cluster

    def __call__(self, logist, hidden):
        hidden_kernel = self._calc_hidden_kernel(hidden)
        l1_loss = self._l1_loss(logist, hidden_kernel, self.num_cluster)
        l2_loss = self._l2_loss(logist)
        l3_loss = self._l3_loss(logist, hidden_kernel, self.num_cluster)
        return l1_loss + l2_loss + l3_loss

    def _l1_loss(self, logist, hidden_kernel, num_cluster):
        return self._d_cs(logist, hidden_kernel, num_cluster)

    def _l2_loss(self, logist):
        n = logist.size(0)
        return 2 / (n * (n - 1)) * self._triu(logist @ torch.t(logist))

    def _l3_loss(self, logist, hidden_kernel, num_cluster):
        if not hasattr(self, 'eye'):
            self.eye = torch.eye(num_cluster, device=self.device)
        m = torch.exp(-self._cdist(logist, self.eye))
        return self._d_cs(m, hidden_kernel, num_cluster)

    def _triu(self, X):
        # Sum of strictly upper triangular part
        return torch.sum(torch.triu(X, diagonal=1))

    def _calc_hidden_kernel(self, x):
        return self._kernel_from_distance_matrix(self._cdist(x, x), self.epsilon)

    def _d_cs(self, A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.

        :param A: Cluster assignment matrix
        :type A:  torch.Tensor
        :param K: Kernel matrix
        :type K: torch.Tensor
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :return: CS-divergence
        :rtype: torch.Tensor
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

        nom = self._atleast_epsilon(nom, eps=self.epsilon)
        dnom_squared = self._atleast_epsilon(dnom_squared, eps=self.epsilon ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * self._triu(nom / torch.sqrt(dnom_squared))
        return d

    def _atleast_epsilon(self, X, eps):
        """
        Ensure that all elements are >= `eps`.

        :param X: Input elements
        :type X: torch.Tensor
        :param eps: epsilon
        :type eps: float
        :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
        :rtype: torch.Tensor
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    def _cdist(self, X, Y):
        """
        Pairwise distance between rows of X and rows of Y.

        :param X: First input matrix
        :type X: torch.Tensor
        :param Y: Second input matrix
        :type Y: torch.Tensor
        :return: Matrix containing pairwise distances between rows of X and rows of Y
        :rtype: torch.Tensor
        """
        xyT = X @ torch.t(Y)
        x2 = torch.sum(X ** 2, dim=1, keepdim=True)
        y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
        d = x2 - 2 * xyT + torch.t(y2)
        return d

    def _kernel_from_distance_matrix(self, dist, min_sigma):
        """
        Compute a Gaussian kernel matrix from a distance matrix.

        :param dist: Disatance matrix
        :type dist: torch.Tensor
        :param min_sigma: Minimum value for sigma. For numerical stability.
        :type min_sigma: float
        :return: Kernel matrix
        :rtype: torch.Tensor
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = F.relu(dist)
        sigma2 = self.rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(- dist / (2 * sigma2))
        return k


class SimSiamLoss(BaseLoss, ABC):
    """
    SimSiam Loss.
    Negative cosine similarity.

    """

    def __call__(self, p1, p2, z1, z2):
        return self._D(p1, z2) / 2  + self._D(p2, z1) / 2

    def _D(self, p, z):
        """
        The original implementation like below, but we could try the faster version.
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()
        :param p:
        :param z:
        :return:
        """
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


class SimCLRLoss(BaseLoss, ABC):
    large_num = 1e9

    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.temp = args.temperature

    @staticmethod
    def _norm(mat):
        return torch.nn.functional.normalize(mat, p=2, dim=1)

    @classmethod
    def _normalized_projections(cls, ps):
        n = ps.size(0) // 2
        h1, h2 = ps[:n], ps[n:]
        h2 = cls._norm(h2)
        h1 = cls._norm(h1)
        return n, h1, h2

    def _loss_func(self, ps):
        n, h1, h2 = self._normalized_projections(ps)

        labels = torch.arange(0, n, device=self.device, dtype=torch.long)
        masks = torch.eye(n, device=self.device)

        logits_aa = ((h1 @ h1.t()) / self.temp) - masks * self.large_num
        logits_bb = ((h2 @ h2.t()) / self.temp) - masks * self.large_num

        logits_ab = (h1 @ h2.t()) / self.temp
        logits_ba = (h2 @ h1.t()) / self.temp

        loss_a = torch.nn.functional.cross_entropy(torch.cat((logits_ab, logits_aa), dim=1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat((logits_ba, logits_bb), dim=1), labels)

        loss = (loss_a + loss_b)

        return loss

    def __call__(self, ps):
        return self._loss_func(ps)


