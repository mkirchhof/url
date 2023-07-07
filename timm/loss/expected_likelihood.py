import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import log_vmf_norm_const
from timm.utils import VonMisesFisher as vmf


class ExpectedLikelihoodKernel(nn.Module):
    def __init__(self, inv_temp=10):
        super().__init__()
        self.inv_temp = inv_temp


    def forward(self, pred, unc, y, features, classifier):
        # Get vMFs of classes
        class_mu = F.normalize(classifier.weight, dim=-1)
        class_kappa = torch.maximum(classifier.weight.norm(dim=-1), torch.ones(1).to(features.device)) * self.inv_temp

        # Build vMFs from the features
        batch_mu = F.normalize(features, dim=-1)
        batch_kappa = 1 / unc # kappa is certainty, not uncertainty

        # Compute similarities
        sim_batch_vs_class = elk_sim(batch_mu, batch_kappa, class_mu, class_kappa)
        # returns a [batch, classes] tensor

        # Compute loss
        loss = (-sim_batch_vs_class.gather(dim=1, index=y.unsqueeze(1)) + sim_batch_vs_class.logsumexp(dim=1)).mean()

        return loss


def elk_sim(mu1, kappa1, mu2, kappa2, rho=1):
    # Computes the expected likelihood kernel between all vmfs in (mu1, kappa1) and in (mu2, kappa2)
    # Returns a Tensor of shape [mu1.shape[0], mu2.shape[0]]
    dim = mu1.shape[-1]

    kappa3 = ((kappa1.unsqueeze(1) * mu1).unsqueeze(1) + (kappa2.unsqueeze(1) * mu2).unsqueeze(0)).norm(dim=-1)
    ppk = rho * (log_vmf_norm_const(kappa1.unsqueeze(1), dim) + log_vmf_norm_const(kappa2.unsqueeze(0), dim)) - log_vmf_norm_const(rho * kappa3, dim)

    return ppk


class NonIsotropicVMF(nn.Module):
    def __init__(self, n_classes=1000, embed_dim=2048, n_samples=16, inv_temp=10, device="cuda:0"):
        super().__init__()

        self.device = device
        self.inv_temp = torch.nn.Parameter(torch.ones(1, device=device) * inv_temp)
        self.num_proxies = n_classes
        self.embed_dim = embed_dim
        self.n_samples = n_samples

        # Initialize proxies
        self.proxies = torch.nn.Linear(in_features=self.embed_dim, out_features=self.num_proxies, bias=False, device=device)
        nn.init.xavier_normal_(self.proxies.weight, gain=inv_temp)
        self.proxies = nn.utils.weight_norm(self.proxies, dim=0, name="weight")
        self.kappa = torch.nn.Linear(in_features=self.embed_dim, out_features=self.num_proxies, bias=False, device=device)
        nn.init.constant_(self.kappa.weight, inv_temp)

    def forward(self, pred, unc, y, features, classifier):
        # Get vMFs of classes
        class_mu = F.normalize(classifier.weight, dim=-1)
        class_kappa = torch.maximum(self.kappa.weight, torch.ones(1).cuda() * 0.1)

        # Build vMFs from the features
        batch_mu = F.normalize(features, dim=-1)
        batch_kappa = 1 / unc # kappa is certainty, not uncertainty

        # Compute similarities
        sim_batch_vs_class = nivmf_elk_sim(batch_mu, batch_kappa, class_mu, class_kappa, n_samples=self.n_samples)
        # returns a [batch, classes] tensor

        # Compute loss
        loss = (-sim_batch_vs_class.gather(dim=1, index=y.unsqueeze(1)) + sim_batch_vs_class.logsumexp(dim=1)).mean()

        return loss


def nivmf_elk_sim(mu1, kappa1, mu2, kappa2, n_samples=16):
    # Computes the expected likelihood kernel between all vmfs in (mu1, kappa1) and the nivmfs (classes) in (mu2, kappa2)
    # Returns a Tensor of shape [mu1.shape[0], mu2.shape[0]]
    kappa1 = kappa1.unsqueeze(1)
    mu1 = F.normalize(mu1, dim=-1)
    mu2 = F.normalize(mu2, dim=-1)

    # Draw samples
    distr = vmf(loc=mu1, scale=kappa1)
    samples = distr.rsample(n_samples)  # dim = [n_mc, n_batch, embed_dim]

    # Calculate the term inside the exp() of the nivMF
    mu2 = mu2 * kappa2  # dim = [n_classes, embed_dim]
    norm_mu2 = torch.norm(mu2, dim=-1)  # dim = [n_classes]
    mu2 = F.normalize(mu2, dim=-1)
    # We want samples be a [n_mc, n_batch, 1, embed_dim] tensor,
    # so that when we multiply it with the mu2 [n_classes, embed_dim] tensor,
    # we do that across all combinations resulting in a [n_mc, n_batch, n_classes, embed_dim] tensor
    samples = samples.unsqueeze(-2)
    cos_sim = torch.einsum('...i,...i->...', samples, mu2)

    # Calculate the remaining terms of the nivmf log likelihood
    logcp = log_vmf_norm_const(norm_mu2)
    detterm = torch.sum(torch.log(kappa2), dim=-1) - 2 * torch.log(norm_mu2)
    logl = logcp + norm_mu2 * cos_sim + detterm

    # Average over samples. The avg log likelihood is equal to the ELK(vmf, nivMF)
    one_over_n = torch.log(torch.ones(1) * n_samples)
    if logl.is_cuda:
        one_over_n = one_over_n.cuda()
    logl = logl - one_over_n
    logl = torch.logsumexp(logl, dim=0)

    return logl.squeeze(logl.dim() - 1)
