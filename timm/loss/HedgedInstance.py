import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import VonMisesFisher


class HedgedInstance(nn.Module):
    def __init__(self, kappa_init=10, b=0, n_samples=16, supervised=False, device=torch.device('cuda:0')):
        """
        kappa_init: Float, which temperature to use
        n_samples: int, how many samples to draw from the vMF distributions
        supervised: bool, whether to define positivity/negativity from class labels (target) or to ignore them
                    and only consider the two crops of the same image as positive
        """
        super().__init__()

        self.n_samples = n_samples
        self.log_n_samples_const = torch.log(torch.ones(1, device=device) * self.n_samples)
        self.kappa = torch.nn.Parameter(torch.ones(1, device=device) * kappa_init, requires_grad=True)
        self.b = torch.nn.Parameter(torch.ones(1, device=device) * b, requires_grad=True)
        self.supervised = supervised
        self.EPS = 1e-5

    def forward(self, output, unc, target, features: torch.Tensor, classifier: nn.Module):
        # Build vMFs from the features
        mu = F.normalize(features, dim=-1)
        kappa = 1 / unc  # kappa is certainty, not uncertainty
        samples = VonMisesFisher(mu, kappa.unsqueeze(1)).rsample(self.n_samples)  # [n_MC, batch, dim]

        # Calculate similarities
        match_prob = (samples.matmul(samples.transpose(-2, -1)) * self.kappa + self.b).sigmoid()  # [n_MC, batch, batch]
        match_prob = match_prob.mean(0) + self.EPS # HIB averages over the samples "on the outside" [batch, batch]

        # Build positive and negative masks
        if not self.supervised:
            target = torch.arange(target.shape[0] // 2, device=target.device, dtype=torch.int64).repeat_interleave(2)
        mask = (target.unsqueeze(1) == target.t().unsqueeze(0)).float()
        pos_mask = mask - torch.diag(torch.ones(mask.shape[0], device=mask.device))
        neg_mask = 1 - mask

        # BCE loss to the positives/negatives
        bce = (match_prob.log() * pos_mask).sum(-1) / pos_mask.sum(-1) - (match_prob.log() * neg_mask).sum(-1) / neg_mask.sum(-1)

        return -bce.mean()
