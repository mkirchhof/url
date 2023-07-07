import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import VonMisesFisher
from .expected_likelihood import elk_sim


class MCInfoNCE(nn.Module):
    def __init__(self, kappa_init=10, n_samples=16, supervised=False, device=torch.device('cuda:0')):
        """
        kappa_init: Float, = 1 / temperature
        n_samples: int, how many samples to draw from the vMF distributions
        supervised: bool, whether to define positivity/negativity from class labels (target) or to ignore them
                    and only consider the two crops of the same image as positive
        """
        super().__init__()

        self.n_samples = n_samples
        self.log_n_samples_const = torch.log(torch.ones(1, device=device) * self.n_samples)
        self.kappa = torch.nn.Parameter(torch.ones(1, device=device) * kappa_init, requires_grad=True)
        self.supervised = supervised

    def forward(self, output, unc, target, features: torch.Tensor, classifier: nn.Module):
        # Build vMFs from the features
        mu = F.normalize(features, dim=-1)
        kappa = 1 / unc  # kappa is certainty, not uncertainty
        samples = VonMisesFisher(mu, kappa.unsqueeze(1)).rsample(self.n_samples)  # [n_MC, batch, dim]

        # Calculate similarities
        sim = samples.matmul(samples.transpose(-2, -1)) * self.kappa  # [n_MC, batch, batch]

        # Build positive and negative masks
        if not self.supervised:
            target = torch.arange(target.shape[0] // 2, device=target.device, dtype=torch.int64).repeat_interleave(2)
        mask = (target.unsqueeze(1) == target.t().unsqueeze(0)).float()
        pos_mask = mask - torch.diag(torch.ones(mask.shape[0], device=mask.device))
        neg_mask = 1 - mask
        # Things with mask = 0 should be ignored in the sum.
        # If we just gave a zero, it would be log sum exp(0) != 0
        # So we need to give them a small value, with log sum exp(-1000) \approx 0
        pos_mask_add = neg_mask * (-1000)
        neg_mask_add = pos_mask * (-1000)

        # calculate the standard log contrastive loss for each vmf sample ([n_MC, batch])
        log_infonce_per_sample_per_example = (sim * pos_mask + pos_mask_add).logsumexp(-1) - (sim * neg_mask + neg_mask_add).logsumexp(-1)

        # Average over the samples (we actually want a logmeanexp, that's why we substract log(n_samples)) ([batch])
        log_infonce_per_example = log_infonce_per_sample_per_example.logsumexp(0) - self.log_n_samples_const

        # Calculate loss ([1])
        log_infonce = torch.mean(log_infonce_per_example)
        return -log_infonce


class InfoNCE(nn.Module):
    def __init__(self, kappa_init=10, supervised=False, device=torch.device('cuda:0')):
        """
        kappa_init: Float, = 1 / temperature
        n_samples: int, how many samples to draw from the vMF distributions
        supervised: bool, whether to define positivity/negativity from class labels (target) or to ignore them
                    and only consider the two crops of the same image as positive
        """
        super().__init__()

        self.kappa = torch.nn.Parameter(torch.ones(1, device=device) * kappa_init, requires_grad=True)
        self.supervised = supervised

    def forward(self, output, unc, target, features: torch.Tensor, classifier: nn.Module):
        features = F.normalize(features, dim=-1)

        # Calculate similarities
        sim = features.matmul(features.transpose(-2, -1)) * self.kappa  # [batch, batch]

        # Build positive and negative masks
        if not self.supervised:
            target = torch.arange(target.shape[0] // 2, device=target.device, dtype=torch.int64).repeat_interleave(2)
        mask = (target.unsqueeze(1) == target.t().unsqueeze(0)).float()
        pos_mask = mask - torch.diag(torch.ones(mask.shape[0], device=mask.device))
        neg_mask = 1 - mask
        # Things with mask = 0 should be ignored in the sum.
        # If we just gave a zero, it would be log sum exp(0) != 0
        # So we need to give them a small value, with log sum exp(-1000) \approx 0
        pos_mask_add = neg_mask * (-1000)
        neg_mask_add = pos_mask * (-1000)

        # calculate the standard log contrastive loss for each vmf sample ([batch])
        log_infonce_per_example = (sim * pos_mask + pos_mask_add).logsumexp(-1) - (sim * neg_mask + neg_mask_add).logsumexp(-1)

        # Calculate loss ([1])
        log_infonce = torch.mean(log_infonce_per_example)
        return -log_infonce
