import torch


def bernoulli_entropy_from_logits(logits, eps=1e-6):
    prob = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)
    return -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob))


def softmax_entropy_from_logits(logits, dim=1):
    log_prob = torch.log_softmax(logits, dim=dim)
    prob = torch.softmax(logits, dim=dim)
    return -(prob * log_prob).sum(dim=dim)


def entropy_loss_from_logits(logits, entropy_mode='auto', min_valid_terms=1, use_sigmoid=True):
    if logits is None:
        return None, {'valid_terms': 0, 'finite': False, 'shape': None, 'mode': entropy_mode}

    mode = str(entropy_mode).lower()
    if mode == 'auto':
        mode = 'sigmoid' if use_sigmoid else 'softmax'

    finite_mask = torch.isfinite(logits)
    valid_terms = int(finite_mask.sum().item())
    if valid_terms < int(min_valid_terms):
        return None, {'valid_terms': valid_terms, 'finite': False, 'shape': tuple(logits.shape), 'mode': mode}

    safe_logits = logits[finite_mask]
    if mode in ['sigmoid', 'bernoulli']:
        entropy = bernoulli_entropy_from_logits(safe_logits)
    elif mode == 'softmax':
        entropy = softmax_entropy_from_logits(logits, dim=1)
        entropy = entropy[torch.isfinite(entropy)]
        valid_terms = int(entropy.numel())
    else:
        raise NotImplementedError('Unsupported Tent entropy mode: %s' % mode)

    if entropy.numel() == 0:
        return None, {'valid_terms': 0, 'finite': False, 'shape': tuple(logits.shape), 'mode': mode}

    loss = entropy.mean()
    return loss, {
        'valid_terms': valid_terms,
        'finite': bool(torch.isfinite(loss).item()),
        'shape': tuple(logits.shape),
        'mode': mode,
    }
