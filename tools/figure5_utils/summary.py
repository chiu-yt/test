"""Evidence-limited candidate review summary; not a benchmark report."""
from pathlib import Path
from typing import Sequence

from .candidates import CandidateEvaluation, bucket_candidates


def _baseline_explanation(recovered: int, fp_reduction: int, localized: int) -> str:
    """Describe one baseline without inferring object-level attribution from aggregate counts."""
    parts = [
        '%d GT object(s) matched by ReFuse-TTA were unmatched by this baseline under the configured gates.' % recovered
        if recovered else
        'No ReFuse-only GT recovery relative to this baseline; this does not establish complete GT coverage.'
    ]
    if fp_reduction:
        parts.append('Conservative unmatched-prediction excess: %d relative to ReFuse-TTA '
                     '(shared lower-bound proxy, not confirmed false positives).' % fp_reduction)
    if localized:
        parts.append('%d ReFuse-TTA localization improvement(s) were recorded against available matching '
                     'baselines; per-baseline localization attribution is unavailable in these counters.' % localized)
    if not (recovered or fp_reduction or localized):
        parts.append('No measured relative issue in the recorded recovery, FP-proxy, or localization evidence.')
    return ' '.join(parts)


def write_summary(ranked: Sequence[CandidateEvaluation], output: Path,
                  provenance: Sequence[str]) -> None:
    lines = ['# Figure 5 candidate review', '', '## Protocol and provenance', '']
    lines.extend('- ' + item for item in provenance)
    lines.extend([
        '', '## Interpretation', '',
        'These are ranked qualitative candidates, not the final three scenes. '
        'All ten detection classes in the infos are retained; GT is never score-filtered. '
        'Native non-detection `ignore` annotations are not detection GT.',
        '', 'Recovery means a GT object matched by ReFuse but unmatched by the named baseline '
        'under the configured gates. It is not an official nuScenes metric or a correctness guarantee. '
        'False-positive reduction is a conservative count of fewer unmatched predictions, '
        'not proof that a particular baseline box is false. Better localization requires at least '
        '0.5 m lower center error than every baseline that matches that GT; an absent baseline '
        'does not establish a localization comparison.',
        '', 'Distances are horizontal sensor-origin ranges, not matching errors. '
        'Primary distances include relevant GT and unmatched-baseline anchors; '
        'their sorted list is not paired positionally with the class list. '
        'Callout JSON pairs class and range per ROI: [x_min, y_min, x_max, y_max] in LiDAR metres. '
        'Plots show lateral y horizontally and forward x vertically within [-50, 50] m; '
        'objects outside that view may still contribute to counts and ranking.',
        '', 'Score = 3*recovered_from_source + 2*recovered_from_codemerge '
        '+ 1.5*false_positive_removed + far_small_detected - 0.1*num_gt_objects. '
        'Ties use ascending sample token. Far means >=30 m; very far means >=40 m. '
        'Small classes: pedestrian, bicycle, motorcycle, traffic_cone.',
        '', '## Bucket availability', '',
    ])
    for kind, candidates in bucket_candidates(ranked).items():
        lines.append('- %s: %d available, %d retained in by-type CSV.' % (
            kind.value, len(candidates), min(20, len(candidates))))
    lines.extend(['', '## Top 30 review'])
    for rank, candidate in enumerate(ranked[:30], 1):
        evidence = candidate.evidence
        theme = ', '.join(kind.value for kind in candidate.memberships) or 'no_evidence_theme'
        classes = ', '.join(kind.value for kind in evidence.involved_classes) or 'none'
        distances = ', '.join('%.2f' % distance for distance in evidence.primary_distances) or 'none'
        source = _baseline_explanation(evidence.recovered_from_source, evidence.false_positive_removed,
                                       evidence.better_localization)
        codemerge = _baseline_explanation(evidence.recovered_from_codemerge, evidence.false_positive_removed,
                                          evidence.better_localization)
        improvement = [
            'GT-assisted recovery vs Source-only: %d; vs CodeMerge: %d (counts may overlap).' % (
                evidence.recovered_from_source, evidence.recovered_from_codemerge),
            'Unmatched-prediction reduction: %d (FP proxy, not per-box correctness).' % evidence.false_positive_removed,
        ]
        if evidence.better_localization:
            improvement.append('%d GT match(es) with >=0.5 m lower center error than every available '
                               'matching baseline(s); an absent baseline is not a localization comparison.' %
                               evidence.better_localization)
        else:
            improvement.append('No measured localization gain under the configured criterion.')
        improvement.append('Far-small GT matches: %d; this count alone does not establish a baseline advantage.' %
                           evidence.far_small_detected)
        reasons = []
        if evidence.recovered_from_source or evidence.recovered_from_codemerge:
            reasons.append('GT-assisted recovery (Source-only: %d, CodeMerge: %d)' % (
                evidence.recovered_from_source, evidence.recovered_from_codemerge))
        if evidence.false_positive_removed:
            reasons.append('unmatched-prediction reduction (%d, FP proxy)' % evidence.false_positive_removed)
        if evidence.better_localization:
            reasons.append('localization (%d improved GT match(es))' % evidence.better_localization)
        callouts = candidate.callouts
        annotations = '; '.join('%s: %s at %.2f m' % (
            item.kind.value, item.class_name.value, item.distance_m) for item in callouts) or 'none available'
        if reasons:
            recommendation = 'Review ' + '; '.join(reasons) + '.'
        else:
            improvement.append('No measured comparative improvement in the recorded evidence.')
            recommendation = (
                'Context only: %d far-small GT match(es), without measured comparative improvement.' %
                evidence.far_small_detected if evidence.far_small_detected else
                'Not recommended on current evidence: no measured recovery, unmatched-prediction '
                'reduction, localization gain, or far-small GT match.'
            )
        recommendation += (
            ' Inspect %d selected callout(s) from %d available; ROIs guide review, not correctness certification.' %
            (len(callouts), len(candidate.available_callouts)) if callouts else
            ' No localized evidence to inspect: no selected callout ROI; aggregate evidence only.'
        )
        lines.extend([
            '', '### Rank %d' % rank, '',
            '- **Token:** `%s`' % candidate.sample_token,
            '- **Types:** ' + theme,
            '- **Involved classes:** ' + classes,
            '- **Primary distances (m):** ' + distances,
            '- **Counts:** GT %d; Source-only / CodeMerge / ReFuse-TTA predictions %d / %d / %d.' % (
                evidence.num_gt_objects, evidence.num_source_preds, evidence.num_codemerge_preds,
                evidence.num_refuse_preds),
            '- **Source-only issue:** ' + source,
            '- **CodeMerge behavior:** ' + codemerge,
            '- **ReFuse-TTA improvement:** ' + ' '.join(improvement),
            '- **Callouts:** ' + annotations,
            '- **Recommendation reason:** ' + recommendation,
        ])
    lines.extend(['', 'Review the clean and callout HD pairs before choosing scenes. '
                  'No statistical significance, final-scene selection, or dataset-wide improvement is asserted.', ''])
    (output / 'figure5_candidate_summary.md').write_text('\n'.join(lines), encoding='utf-8')
