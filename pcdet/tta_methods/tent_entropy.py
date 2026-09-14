import math

import torch


def bernoulli_entropy_from_logits(logits, eps=1e-6):
    prob = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)
    return -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob))


def softmax_entropy_from_logits(logits, dim=1):
    log_prob = torch.log_softmax(logits, dim=dim)
    prob = torch.softmax(logits, dim=dim)
    return -(prob * log_prob).sum(dim=dim)


def extract_detection_entropy(logits, mode='sigmoid'):
    """Proposal-level entropy H[B, P] from logits [B, C, P] plus its maximum."""
    normalized = str(mode).lower()
    if normalized in ('sigmoid', 'bernoulli'):
        hmax = math.log(2.0)
    elif normalized == 'softmax':
        hmax = math.log(float(logits.shape[1]))
    else:
        raise NotImplementedError('Unsupported entropy mode: %s' % mode)

    if logits.numel() == 0:
        return logits.new_zeros((logits.shape[0], logits.shape[2])), hmax
    if normalized in ('sigmoid', 'bernoulli'):
        entropy = bernoulli_entropy_from_logits(logits).mean(dim=1)
    else:
        entropy = softmax_entropy_from_logits(logits, dim=1)
    return entropy, hmax


def entropy_loss_from_logits(logits, entropy_mode='auto', min_valid_terms=1, use_sigmoid=True):
    if logits is None:
        return None, {'valid_terms': 0, 'finite': False, 'shape': None, 'mode': entropy_mode}

    mode = str(entropy_mode).lower()
    if mode == 'auto':
        mode = 'sigmoid' if use_sigmoid else 'softmax'
    is_sigmoid = mode in ('sigmoid', 'bernoulli')
    if not is_sigmoid and mode != 'softmax':
        raise NotImplementedError('Unsupported Tent entropy mode: %s' % mode)

    entropy, _ = extract_detection_entropy(logits, mode='sigmoid' if is_sigmoid else 'softmax')

    if is_sigmoid:
        valid_terms = int(torch.isfinite(logits).sum().item())
    else:
        valid_terms = int(torch.isfinite(entropy).sum().item())
    if valid_terms < int(min_valid_terms):
        return None, {'valid_terms': valid_terms, 'finite': False, 'shape': tuple(logits.shape), 'mode': mode}

    finite_entropy = entropy[torch.isfinite(entropy)]
    if finite_entropy.numel() == 0:
        return None, {'valid_terms': 0, 'finite': False, 'shape': tuple(logits.shape), 'mode': mode}

    loss = finite_entropy.mean()
    return loss, {
        'valid_terms': valid_terms,
        'finite': bool(torch.isfinite(loss).item()),
        'shape': tuple(logits.shape),
        'mode': mode,
    }
