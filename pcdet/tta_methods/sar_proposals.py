import torch


class SARProposalAlignmentError(RuntimeError):
    """Raised when proposal identities cannot safely align SAR entropy."""


def validated_proposal_ids(proposal_ids, entropy, source):
    if proposal_ids is None:
        raise SARProposalAlignmentError('SAR %s forward did not expose proposal IDs' % source)
    proposal_ids = proposal_ids.detach()
    if proposal_ids.shape != entropy.shape or proposal_ids.dim() != 2:
        raise SARProposalAlignmentError(
            'SAR %s proposal IDs must match entropy shape: %s != %s' % (
                source, tuple(proposal_ids.shape), tuple(entropy.shape),
            )
        )
    if proposal_ids.shape[1] > 1:
        sorted_ids = proposal_ids.sort(dim=1).values
        if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any().item()):
            raise SARProposalAlignmentError('SAR %s proposal IDs must be unique per sample' % source)
    return proposal_ids


def align_entropy_strict(entropy, proposal_ids, reference_ids):
    if proposal_ids.shape[0] != reference_ids.shape[0]:
        raise SARProposalAlignmentError('SAR prediction and adaptation batch sizes must match')
    matches = reference_ids.unsqueeze(-1) == proposal_ids.unsqueeze(-2)
    if not bool((matches.sum(dim=-1) == 1).all().item()):
        raise SARProposalAlignmentError(
            'SAR unperturbed adaptation proposals must match prediction proposals'
        )
    indices = matches.to(dtype=torch.long).argmax(dim=-1)
    return entropy.gather(1, indices)


def align_selected_entropy(entropy, proposal_ids, reference_ids, reference_mask):
    selected_positions = reference_mask.nonzero(as_tuple=False)
    selected_batches = selected_positions[:, 0]
    selected_ids = reference_ids[reference_mask]
    candidate_ids = proposal_ids[selected_batches]
    matches = candidate_ids == selected_ids.unsqueeze(-1)
    present = matches.any(dim=-1)
    indices = matches.to(dtype=torch.long).argmax(dim=-1)
    selected_entropy = entropy[selected_batches].gather(1, indices.unsqueeze(-1)).squeeze(-1)
    return selected_entropy[present]
