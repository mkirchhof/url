import torch
import torch.nn as nn


class LossPrediction(nn.Module):
    def __init__(self, lambda_, task_loss=None, unc_loss=None, detach=True, ignore_ce_loss=False):
        super().__init__()
        self.task_loss = task_loss or nn.CrossEntropyLoss(reduction="none")
        self.unc_loss = unc_loss or nn.MSELoss()
        self.lambda_ = lambda_
        self.detach = detach
        self.ce_loss_multiplier = 1 - ignore_ce_loss

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

        return self.ce_loss_multiplier * task_loss + self.lambda_ * unc_loss
