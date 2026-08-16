import torch
import torch.nn as nn


def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
    loss = torch.clamp(input, min=0) - input * target + torch.log1p(torch.exp(-torch.abs(input)))
    return loss


class EqualizedFocalClassificationLoss(nn.Module):
    """Equalized focal loss with per-class positive/negative gradient EMA."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 beta: float = 0.75, tau: float = 0.5, momentum: float = 0.9,
                 min_weight: float = 0.25, max_weight: float = 2.0, eps: float = 1e-6):
        super(EqualizedFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.tau = tau
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.eps = eps
        self.register_buffer('pos_grad', torch.zeros(0))
        self.register_buffer('neg_grad', torch.zeros(0))

    def _update_gradient_stats(self, pred_sigmoid, target, weights):
        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        grad = torch.abs(pred_sigmoid.detach() - target)
        reduce_dims = tuple(range(target.dim() - 1))
        pos_grad = (grad * target * weights).sum(dim=reduce_dims)
        neg_grad = (grad * (1.0 - target) * weights).sum(dim=reduce_dims)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(pos_grad, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(neg_grad, op=torch.distributed.ReduceOp.SUM)

        if self.pos_grad.numel() != pos_grad.numel() or self.pos_grad.device != pos_grad.device:
            self.pos_grad = torch.zeros_like(pos_grad)
            self.neg_grad = torch.zeros_like(neg_grad)

        self.pos_grad.mul_(self.momentum).add_(pos_grad, alpha=1.0 - self.momentum)
        self.neg_grad.mul_(self.momentum).add_(neg_grad, alpha=1.0 - self.momentum)

    def _equalization_weight(self, target):
        ratio = self.pos_grad / (self.neg_grad + self.eps)
        ratio = ratio.clamp(min=0.0, max=1.0)
        tail_factor = torch.pow(1.0 - ratio, self.tau)
        pos_weight = (1.0 + self.beta * tail_factor).clamp(max=self.max_weight)
        neg_weight = (1.0 - self.beta * tail_factor).clamp(min=self.min_weight)
        view_shape = [1] * (target.dim() - 1) + [-1]
        pos_weight = pos_weight.view(*view_shape)
        neg_weight = neg_weight.view(*view_shape)
        return target * pos_weight + (1.0 - target) * neg_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        pred_sigmoid = torch.sigmoid(input)
        with torch.no_grad():
            self._update_gradient_stats(pred_sigmoid, target, weights)

        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        loss = focal_weight * sigmoid_cross_entropy_with_logits(input, target)
        loss = loss * self._equalization_weight(target)

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()
        return loss * weights
