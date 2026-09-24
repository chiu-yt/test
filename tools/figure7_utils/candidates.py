import csv
import json
from pathlib import Path
from typing import Final, Mapping, Tuple

from .contracts import CaptureBundle, Crop, LedgerEntry, LedgerReference, SelectedRecord
from .rendering import crop_for_case, render_preview, save_figure


CSV_FIELDS: Final[Tuple[str, ...]] = (
    'ledger_index', 'record_id', 'status', 'token', 'frame_id', 'epoch', 'model_step',
    'global_rank', 'batch_index', 'proposal_index', 'class_name', 'confidence_q',
    'accepted', 'rescued', 'match_1', 'match_2', 'match_3', 'match_4',
    'quality_1', 'quality_2', 'quality_3', 'quality_4', 'exact_reliability_r',
    'case_a_eligible', 'case_b_eligible', 'case_a_rank_key', 'case_b_rank_key',
    'admitted_pools', 'retained', 'selected_case', 'crop', 'preview',
)


def _record_key(record: SelectedRecord) -> Tuple[str, int, str]:
    candidate = record.candidate
    return candidate.record_id, candidate.reference_index, candidate.pool


def preview_names(bundle: CaptureBundle) -> Mapping[Tuple[str, int, str], str]:
    names = {}
    for pool, prefix in (('variable', 'figure7_caseA_candidates'),
                         ('stable', 'figure7_caseB_candidates')):
        records = sorted((item for item in bundle.retained if item.candidate.pool == pool),
                         key=lambda item: (item.candidate.rank_key, item.candidate.record_id))
        for rank, record in enumerate(records, 1):
            candidate = record.candidate
            filename = '%03d_%s_p%03d.png' % (
                rank, candidate.identity.token, candidate.reference_index)
            names[_record_key(record)] = prefix + '/' + filename
    return names


def _selected_case(bundle: CaptureBundle, entry: LedgerEntry,
                   reference: LedgerReference) -> str:
    key = entry.record_id, reference.index
    if key == (bundle.case_a.candidate.record_id, bundle.case_a.candidate.reference_index):
        return 'A'
    if key == (bundle.case_b.candidate.record_id, bundle.case_b.candidate.reference_index):
        return 'B'
    return ''


def _candidate_record(bundle: CaptureBundle, entry: LedgerEntry,
                      reference: LedgerReference) -> SelectedRecord | None:
    matches = tuple(item for item in bundle.retained
                    if item.candidate.record_id == entry.record_id
                    and item.candidate.reference_index == reference.index)
    return matches[0] if matches else None


def _csv_row(bundle: CaptureBundle, ledger_index: int, entry: LedgerEntry,
             reference: LedgerReference, names: Mapping[Tuple[str, int, str], str],
             point_range: Crop) -> Mapping[str, str | int | float | bool]:
    retained = _candidate_record(bundle, entry, reference)
    pool = '' if retained is None else retained.candidate.pool
    preview = '' if retained is None else names[_record_key(retained)]
    crop = '' if retained is None else json.dumps(crop_for_case(retained, point_range))
    matches = reference.matches + (-1,) * (4 - len(reference.matches))
    qualities = reference.qualities + (0.0,) * (4 - len(reference.qualities))
    return {
        'ledger_index': ledger_index, 'record_id': entry.record_id, 'status': entry.status,
        'token': entry.token, 'frame_id': entry.frame_id, 'epoch': entry.epoch,
        'model_step': entry.model_step, 'global_rank': entry.global_rank,
        'batch_index': entry.batch_index, 'proposal_index': reference.index,
        'class_name': reference.class_name,
        'confidence_q': '' if reference.score is None else '%.17g' % reference.score,
        'accepted': reference.accepted, 'rescued': reference.rescued,
        'match_1': matches[0], 'match_2': matches[1], 'match_3': matches[2],
        'match_4': matches[3], 'quality_1': '%.17g' % qualities[0],
        'quality_2': '%.17g' % qualities[1], 'quality_3': '%.17g' % qualities[2],
        'quality_4': '%.17g' % qualities[3],
        'exact_reliability_r': '%.17g' % reference.reliability,
        'case_a_eligible': reference.variable_key is not None,
        'case_b_eligible': reference.stable_key is not None,
        'case_a_rank_key': '' if reference.variable_key is None else json.dumps(reference.variable_key),
        'case_b_rank_key': '' if reference.stable_key is None else json.dumps(reference.stable_key),
        'admitted_pools': '|'.join(entry.admitted_pools), 'retained': retained is not None,
        'selected_case': _selected_case(bundle, entry, reference), 'crop': crop,
        'preview': preview, 'pool': pool,
    }


def write_candidates(bundle: CaptureBundle, output: Path, point_range: Crop) -> None:
    names = preview_names(bundle)
    with (output / 'figure7_candidates.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_FIELDS)
        for ledger_index, entry in enumerate(bundle.ledger, 1):
            for reference in entry.references:
                row = _csv_row(bundle, ledger_index, entry, reference, names, point_range)
                writer.writerow(tuple(row[field] for field in CSV_FIELDS))
    for directory in ('figure7_caseA_candidates', 'figure7_caseB_candidates'):
        (output / directory).mkdir()
    for record in sorted(bundle.retained, key=lambda item: (
            item.candidate.pool, item.candidate.rank_key, item.candidate.record_id)):
        save_figure(render_preview(record, point_range), output / names[_record_key(record)])
