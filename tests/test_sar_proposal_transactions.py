import pytest
import torch
import torch.nn as nn

from pcdet.tta_methods.sar import SAR, SARStepInput
from pcdet.tta_methods.sar_optimizer import SAM


class _Detector(nn.Module):
    def __init__(self, dense_head):
        super().__init__()
        self.dense_head = dense_head

    def forward(self, batch):
        return self.dense_head.predict(batch), {}


class _SequencedProposalDenseHead(nn.Module):
    def __init__(self, proposal_ids, logit_magnitudes):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.proposal_ids = proposal_ids
        self.logit_magnitudes = logit_magnitudes
        self.call_count = 0

    def predict(self, _inputs):
        index = self.call_count
        self.call_count += 1
        self.last_top_proposals = torch.tensor(
            [self.proposal_ids[index]], device=self.weight.device
        )
        magnitudes = self.weight.new_tensor(self.logit_magnitudes[index]) + self.weight
        logits = torch.stack((magnitudes, -magnitudes), dim=0).unsqueeze(0)
        return {'heatmap': logits}


def _build_sar(model):
    optimizer = SAM(
        model.parameters(), torch.optim.SGD,
        lr=0.1, momentum=0.9, rho=0.05,
    )
    return optimizer, SAR(model, optimizer, {'RELIABLE_MARGIN_NORM': 0.4})


def _assert_gradient_clean(parameter):
    assert parameter.grad is None or bool((parameter.grad == 0).all().item())


def test_adapt_aligns_reordered_unperturbed_proposals_by_identity():
    # Given one reliable official proposal whose fresh-forward position changes.
    dense_head = _SequencedProposalDenseHead(
        proposal_ids=[[20, 10], [10, 20]],
        logit_magnitudes=[[0.0, 10.0], [10.0, 10.0]],
    )
    model = _Detector(dense_head)
    _, sar = _build_sar(model)
    step = SARStepInput(
        {}, torch.tensor([[0.01, 0.9]]), torch.tensor([[10, 20]]), 1.0, 6
    )

    # When SAR aligns the graph-connected first entropy to official IDs.
    result = sar.adapt(step)

    # Then the selected ID uses its low-entropy value rather than position zero.
    assert result.skip_reason is None
    assert result.loss_first is not None
    assert result.loss_first < 0.01


def test_adapt_rejects_missing_unperturbed_proposal_before_sam():
    # Given a fresh unperturbed Top-K that does not contain every official ID.
    dense_head = _SequencedProposalDenseHead(
        proposal_ids=[[10, 30]], logit_magnitudes=[[10.0, 10.0]],
    )
    model = _Detector(dense_head)
    optimizer, sar = _build_sar(model)
    step = SARStepInput(
        {}, torch.tensor([[0.01, 0.01]]), torch.tensor([[10, 20]]), 1.0, 7
    )

    # When SAR validates identity before the first backward and perturbation.
    with pytest.raises(RuntimeError, match='unperturbed adaptation proposals'):
        sar.adapt(step)

    # Then no SAM transaction or stale gradient survives.
    assert optimizer.transaction_active is False
    _assert_gradient_clean(model.dense_head.weight)


def test_adapt_intersects_perturbed_topk_with_first_selected_ids():
    # Given SAM replaces one selected official proposal in its perturbed Top-K.
    dense_head = _SequencedProposalDenseHead(
        proposal_ids=[[10, 20], [20, 30]],
        logit_magnitudes=[[10.0, 10.0], [10.0, 10.0]],
    )
    model = _Detector(dense_head)
    _, sar = _build_sar(model)
    step = SARStepInput(
        {}, torch.tensor([[0.01, 0.01]]), torch.tensor([[10, 20]]), 1.0, 8
    )

    # When the second pass aligns by identity rather than position.
    result = sar.adapt(step)

    # Then only the surviving official proposal is a second-pass candidate.
    assert result.skip_reason is None
    assert result.second_candidates == 1
    assert result.second_selected == 1


def test_adapt_rolls_back_when_no_selected_id_survives_perturbed_topk():
    # Given SAM replaces every selected official proposal in its perturbed Top-K.
    dense_head = _SequencedProposalDenseHead(
        proposal_ids=[[10, 20], [30, 40]],
        logit_magnitudes=[[10.0, 10.0], [10.0, 10.0]],
    )
    model = _Detector(dense_head)
    optimizer, sar = _build_sar(model)
    before = model.dense_head.weight.detach().clone()
    step = SARStepInput(
        {}, torch.tensor([[0.01, 0.01]]), torch.tensor([[10, 20]]), 1.0, 10
    )

    # When no first-selected identity remains after the SAM perturbation.
    result = sar.adapt(step)

    # Then SAR uses its normal empty-second-set rollback path.
    assert result.skip_reason == 'second_empty_or_nonfinite'
    assert result.second_candidates == 0
    assert torch.equal(model.dense_head.weight, before)
    assert optimizer.transaction_active is False
