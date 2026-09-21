"""Scene explanations must stay within the evaluated matching evidence."""
from dataclasses import replace
from pathlib import Path

import pytest

from tools.figure5_utils.candidates import evaluate_frame
from tools.figure5_utils.domain import Box3D, Detection, DetectionScore, FrameRecord, SampleToken, YawRadians
from tools.figure5_utils.matching import match_frame
from tools.figure5_utils.palette import DetectionClass
from tools.figure5_utils.summary import write_summary


def detection(distance: float, kind: DetectionClass = DetectionClass.CAR) -> Detection:
    return Detection(SampleToken('scene'), kind, DetectionScore(0.9),
                     Box3D((distance, 0.0, 0.0), (2.0, 4.0, 1.0), YawRadians(0.0)))


def field(text: str, label: str) -> str:
    prefix = '- **%s:** ' % label
    values = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    assert len(values) == 1, 'Expected one scene field: %s' % label
    return values[0]


@pytest.mark.parametrize('source_recovers', [True, False])
def test_explanation_attributes_recovery_when_only_one_baseline_misses_gt(
        tmp_path: Path, source_recovers: bool) -> None:
    """Given asymmetric baseline matches, when summarized, then only the missing baseline has a recovery issue."""
    target = detection(35.0, DetectionClass.PEDESTRIAN)
    source, merge = ((), (target,)) if source_recovers else ((target,), ())
    frame = FrameRecord(SampleToken('scene'), (target,), source, merge, (target,))
    candidate = evaluate_frame(frame, match_frame(frame))

    write_summary((candidate,), tmp_path, ('test provenance',))

    text = (tmp_path / 'figure5_candidate_summary.md').read_text()
    missed = field(text, 'Source-only issue' if source_recovers else 'CodeMerge behavior')
    shared = field(text, 'CodeMerge behavior' if source_recovers else 'Source-only issue')
    assert '1 GT object(s) matched by ReFuse-TTA were unmatched' in missed
    assert 'No ReFuse-only GT recovery' in shared
    assert 'complete GT coverage' in shared
    assert field(text, 'Token') == '`scene`'
    assert field(text, 'Types') == 'far_range_recovery, small_object_recovery'
    assert field(text, 'Involved classes') == 'pedestrian'
    assert field(text, 'Primary distances (m)') == '35.00'
    improvement = field(text, 'ReFuse-TTA improvement')
    assert ('vs Source-only: %d' % int(source_recovers)) in improvement
    assert ('vs CodeMerge: %d' % int(not source_recovers)) in improvement
    reason = field(text, 'Recommendation reason')
    assert 'Review GT-assisted recovery' in reason
    assert '2 selected callout(s)' in reason
    assert 'pedestrian at 35.00 m' in field(text, 'Callouts')


@pytest.mark.parametrize('remaining', [False, True])
def test_explanation_limits_fp_claim_when_reduction_is_only_matching_evidence(
        tmp_path: Path, remaining: bool) -> None:
    """Given fewer unmatched ReFuse boxes, when summarized, then FP is a proxy and missing ROIs are explicit."""
    baseline = (detection(10.0), detection(20.0))
    refuse = (detection(15.0),) if remaining else ()
    frame = FrameRecord(SampleToken('scene'), (), baseline, baseline, refuse)
    candidate = evaluate_frame(frame, match_frame(frame))

    write_summary((candidate,), tmp_path, ())

    text = (tmp_path / 'figure5_candidate_summary.md').read_text()
    expected = 1 if remaining else 2
    for label in ('Source-only issue', 'CodeMerge behavior'):
        explanation = field(text, label)
        assert 'Conservative unmatched-prediction excess: %d' % expected in explanation
        assert 'not confirmed false positives' in explanation
    assert 'Unmatched-prediction reduction: %d' % expected in field(text, 'ReFuse-TTA improvement')
    reason = field(text, 'Recommendation reason')
    assert 'Review unmatched-prediction reduction' in reason
    if remaining:
        assert 'aggregate evidence only' in reason
        assert 'no selected callout ROI' in reason
        assert field(text, 'Callouts') == 'none available'
    else:
        assert '1 selected callout(s)' in reason
        assert 'false_positive_reduction' in field(text, 'Callouts')


def test_explanation_qualifies_localization_when_one_baseline_has_no_match(tmp_path: Path) -> None:
    """Given only Source matches a GT, when summarized, then localization is not attributed to absent CodeMerge."""
    target = detection(20.0)
    frame = FrameRecord(SampleToken('scene'), (target,), (detection(21.0),), (), (target,))
    candidate = evaluate_frame(frame, match_frame(frame))

    write_summary((candidate,), tmp_path, ())

    text = (tmp_path / 'figure5_candidate_summary.md').read_text()
    for label in ('Source-only issue', 'CodeMerge behavior'):
        assert 'per-baseline localization attribution is unavailable' in field(text, label)
    improvement = field(text, 'ReFuse-TTA improvement')
    assert '1 GT match(es) with >=0.5 m lower center error' in improvement
    assert 'available matching baseline(s)' in improvement
    assert 'absent baseline' in improvement
    assert 'localization' in field(text, 'Recommendation reason')
    assert 'better_localization: car at 20.00 m' in field(text, 'Callouts')


def test_explanation_declines_recommendation_when_no_positive_evidence_exists(tmp_path: Path) -> None:
    """Given missed GT in every run, when summarized, then no issue or improvement is invented."""
    frame = FrameRecord(SampleToken('scene'), (detection(10.0),), (), (), ())
    candidate = evaluate_frame(frame, match_frame(frame))

    write_summary((candidate,), tmp_path, ())

    text = (tmp_path / 'figure5_candidate_summary.md').read_text()
    assert field(text, 'Types') == 'no_evidence_theme'
    assert field(text, 'Involved classes') == 'none'
    assert field(text, 'Primary distances (m)') == 'none'
    for label in ('Source-only issue', 'CodeMerge behavior'):
        assert 'No measured relative issue' in field(text, label)
    assert 'No measured comparative improvement' in field(text, 'ReFuse-TTA improvement')
    assert 'Not recommended on current evidence' in field(text, 'Recommendation reason')
    assert field(text, 'Callouts') == 'none available'


def test_explanation_uses_context_only_when_far_small_match_is_shared(tmp_path: Path) -> None:
    """Given identical far-small matches, when summarized, then detection is not called a comparative gain."""
    target = detection(35.0, DetectionClass.PEDESTRIAN)
    frame = FrameRecord(SampleToken('scene'), (target,), (target,), (target,), (target,))
    candidate = evaluate_frame(frame, match_frame(frame))

    write_summary((candidate,), tmp_path, ())

    text = (tmp_path / 'figure5_candidate_summary.md').read_text()
    assert 'No measured comparative improvement' in field(text, 'ReFuse-TTA improvement')
    assert 'Far-small GT matches: 1' in field(text, 'ReFuse-TTA improvement')
    assert 'Context only' in field(text, 'Recommendation reason')


def test_rank_sections_preserve_protocol_and_buckets_when_more_than_thirty_candidates_exist(tmp_path: Path) -> None:
    """Given 31 ranked candidates, when summarized, then only Top30 get separate scene entries."""
    target = detection(35.0)
    frame = FrameRecord(SampleToken('scene'), (target,), (), (), (target,))
    candidate = evaluate_frame(frame, match_frame(frame))
    ranked = tuple(replace(candidate, frame=replace(frame, sample_token=SampleToken('scene-%02d' % index)))
                   for index in range(31))

    write_summary(ranked, tmp_path, ('test provenance',))

    text = (tmp_path / 'figure5_candidate_summary.md').read_text()
    assert [line for line in text.splitlines() if line.startswith('### Rank ')] == [
        '### Rank %d' % rank for rank in range(1, 31)]
    assert '`scene-29`' in text and '`scene-30`' not in text
    assert '## Protocol and provenance\n\n- test provenance' in text
    assert '- far_range_recovery: 31 available, 20 retained in by-type CSV.' in text
