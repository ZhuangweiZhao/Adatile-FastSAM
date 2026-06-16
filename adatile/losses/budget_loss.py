"""Budget / sparsity control losses.

FixedBudget     — (imp.mean - target)²  (manual control)
EntropyBudget   — maximize entropy → push to 0/1
LearnableBudget — both target and weight learned via gradient descent
"""

import math
import torch
import torch.nn as nn


class FixedBudget(nn.Module):
    """Fixed-target activation rate loss: L = (imp.mean - target)²."""

    def __init__(self, target=0.15):
        super().__init__()
        self.target = target

    def forward(self, importance):
        imp_mean = importance.mean()
        return (imp_mean - self.target) ** 2, imp_mean.item()


class EntropyBudget(nn.Module):
    """Maximize entropy → push importance toward 0 or 1."""

    def forward(self, importance):
        imp_c = importance.clamp(1e-7, 1 - 1e-7)
        H = -(imp_c * imp_c.log() + (1 - imp_c) * (1 - imp_c).log()).mean()
        return -H, importance.mean().item()


class LearnableBudget(nn.Module):
    """Learnable budget: both target rate and weight are trained.

    target = σ(w)  (sigmoid of learned logit, always in [0,1])
    weight = exp(log_λ)  (positive via exponential)

    L = exp(log_λ) * (imp_mean - σ(w))² + λ_reg * σ(w)
         ↑ budget strength            ↑ sparsity penalty
    """

    def __init__(self, init_target=0.15, init_lambda=5.0, lambda_reg=0.1):
        super().__init__()
        self.logit_target = nn.Parameter(
            torch.tensor(LearnableBudget._inv_sigmoid(init_target)))
        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda)))
        self.lambda_reg = lambda_reg

    @staticmethod
    def _inv_sigmoid(x):
        x = max(min(x, 0.99), 0.01)
        return math.log(x / (1 - x))

    def get_target(self):
        return self.logit_target.sigmoid().item()

    def get_lambda(self):
        return self.log_lambda.exp().item()

    def forward(self, importance):
        target = self.logit_target.sigmoid()
        weight = self.log_lambda.exp()
        imp_mean = importance.mean()
        loss = weight * (imp_mean - target) ** 2
        loss = loss + self.lambda_reg * target  # sparsity penalty
        return loss, imp_mean.item(), target.item(), weight.item()
