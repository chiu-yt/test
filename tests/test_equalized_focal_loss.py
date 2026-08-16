import torch

from pcdet.utils.equalized_focal_loss import EqualizedFocalClassificationLoss


def test_equalized_focal_loss_keeps_zero_weight_queries_zero():
    loss_fn = EqualizedFocalClassificationLoss()
    logits = torch.zeros((1, 2, 3), dtype=torch.float32)
    targets = torch.zeros_like(logits)
    targets[0, 0, 2] = 1.0
    weights = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss = loss_fn(logits, targets, weights)

    assert torch.isfinite(loss).all()
    assert torch.equal(loss[0, 1], torch.zeros(3))


def test_equalized_focal_loss_updates_per_class_gradient_stats():
    loss_fn = EqualizedFocalClassificationLoss(momentum=0.0)
    logits = torch.zeros((1, 3, 3), dtype=torch.float32)
    targets = torch.zeros_like(logits)
    targets[0, 0, 0] = 1.0
    targets[0, 1, 0] = 1.0
    targets[0, 2, 2] = 1.0
    weights = torch.ones((1, 3), dtype=torch.float32)

    loss = loss_fn(logits, targets, weights)

    assert torch.isfinite(loss).all()
    assert loss_fn.pos_grad.shape[0] == 3
    assert loss_fn.neg_grad.shape[0] == 3
    assert loss_fn.pos_grad[0] > loss_fn.pos_grad[1]
    assert loss_fn.neg_grad[1] > 0
