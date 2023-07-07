import torch
import torch.nn as nn


class RiskPrediction(nn.Module):
    def __init__(self, lambda_, task_loss=None, unc_loss=None, detach=True):
        super().__init__()
        self.task_loss = task_loss or nn.CrossEntropyLoss(reduction="none")
        self.unc_loss = unc_loss or nn.MSELoss()
        self.lambda_ = lambda_
        self.detach = detach

    def forward(
        self,
        x: torch.Tensor,
        unc: torch.Tensor,
        target: torch.Tensor,
        features: torch.Tensor,
        classifier: nn.Module,
    ) -> torch.Tensor:
        task_loss_per_sample = self.task_loss(x, target)

        if self.detach:
            task_loss_target = task_loss_per_sample.detach()
        else:
            task_loss_target = task_loss_per_sample

        unc_loss = self.unc_loss(unc, task_loss_target)
        task_loss = task_loss_per_sample.mean()

        return task_loss + self.lambda_ * unc_loss
