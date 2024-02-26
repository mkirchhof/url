import torch
import torch.nn as nn


class LossPrediction(nn.Module):
    def __init__(self, lambda_, task_loss=None, unc_loss=None, detach=True, inv_temp=1, ignore_ce_loss=False):
        super().__init__()
        self.task_loss = task_loss or nn.CrossEntropyLoss(reduction="none")
        self.unc_loss = unc_loss or nn.MSELoss()
        self.lambda_ = lambda_
        self.detach = detach
        self.inv_temp = inv_temp
        self.ce_loss_multiplier = 1 - ignore_ce_loss

    def forward(
        self,
        x: torch.Tensor,
        unc: torch.Tensor,
        target: torch.Tensor,
        features: torch.Tensor,
        classifier: nn.Module,
    ) -> torch.Tensor:
        task_loss_per_sample = self.task_loss(x * self.inv_temp, target)

        if self.detach:
            task_loss_target = task_loss_per_sample.detach()
        else:
            task_loss_target = task_loss_per_sample

        unc_loss = self.unc_loss(unc, task_loss_target)
        task_loss = task_loss_per_sample.mean()

        return self.ce_loss_multiplier * task_loss + self.lambda_ * unc_loss


class LossOrderLoss(nn.Module):
    def __init__(self, leeway_rho=0., gap=0.1):
        super().__init__()
        self.gap = gap
        self.leeway_rho = leeway_rho

    def forward(self,
                pred_loss,
                true_loss):
        if pred_loss.shape[0] % 2 != 0:
            raise AssertionError("For using losspred-order, batchsize must be a multiple of 2.")

        pred_pairs = pred_loss.view(*pred_loss.shape[:-1], pred_loss.shape[-1] // 2, 2)
        true_pairs = true_loss.view(*true_loss.shape[:-1], true_loss.shape[-1] // 2, 2)

        actually_bigger = (true_pairs[:, 0] > true_pairs[:, 1]).float() * 2 - 1
        is_bigger_than_allowed_leeway = ((true_pairs[:, 0] - true_pairs[:, 1]).abs() >= self.leeway_rho).float()
        actually_bigger = actually_bigger * is_bigger_than_allowed_leeway

        loss = torch.maximum(torch.zeros(1, device=pred_loss.device),
                             - actually_bigger * (pred_pairs[:, 0] - pred_pairs[:, 1]) + self.gap)
        loss = loss.repeat_interleave(2, -1)

        return loss.mean()
