import torch
import torch.nn as nn
import torch.nn.functional as F


class HETXLHead(nn.Module):
    def __init__(self, rank_V, num_mc, c_mult, num_features, fc):
        super().__init__()
        self.rank_V = rank_V
        self.num_mc = num_mc
        self.num_features = num_features

        self.C = nn.Parameter(c_mult * torch.randn(self.num_features, self.num_features * self.rank_V + self.num_features))
        self.temp = nn.Parameter(torch.ones(()))
        self.fc = fc

    def forward(self, features, calc_cov_log_det=False):
        B, D = features.shape
        R = self.rank_V
        S = self.num_mc
        cov_params = features @ self.C
        V = cov_params[:, :D * R].reshape(-1, D, R)  # [B, D, R]

        d = F.softplus(cov_params[:, D * R:])  # [B, D]

        if calc_cov_log_det:
            cov_diag = V.square().sum(dim=-1) + d  # [B, D]
            cov_log_det = cov_diag.log().sum(dim=-1)  # [B]

        d_sqrt = d.sqrt()

        diag_samples = d_sqrt.unsqueeze(1) * torch.randn(B, S, 1, device=features.device)  # [B, S, D]
        standard_samples = torch.randn(B, S, R, device=features.device)  # [B, S, R]
        einsum_res = torch.einsum("bdr,bsr->bsd", V, standard_samples)  # [B, S, D]
        samples =  einsum_res + diag_samples  # [B, S, D]
        logits = self.fc(features.unsqueeze(1) + samples)  # [B, S, C]
        temp = F.softplus(self.temp)
        avg_logits = F.softmax(logits / temp, dim=-1).mean(dim=1).log()  # [B, C]

        if calc_cov_log_det: 
            return avg_logits, cov_log_det
        
        return avg_logits