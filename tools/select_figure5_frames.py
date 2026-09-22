#!/usr/bin/env python3
"""Select an auditable 4/3/3 shortlist from an existing Figure 5 package."""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from figure5_curation import CATEGORIES, QUOTAS, CandidateRow, CuratedResult, curate_candidates


def _panel_path(input_dir: Path, row: CandidateRow) -> str:
    stem = '%d_%s.png' % (row.rank, row.token)
    for directory in ('qual_panels_top30_hd', 'qual_panels_top60'):
        path = input_dir / directory / stem
        if path.is_file():
            return str(path.relative_to(input_dir))
    return ''


def _records(result: CuratedResult, input_dir: Path) -> List[Dict[str, str]]:
    category_ranks = {category: 0 for category in CATEGORIES}
    records = []
    for selection_rank, item in enumerate(result.selections, 1):
        category_ranks[item.category] += 1
        record = dict(item.row.raw)
        record.update({
            'selection_rank': str(selection_rank),
            'scene_category': item.category.value,
            'category_rank': str(category_ranks[item.category]),
            'evidence_tier': str(item.tier),
            'selection_reason': item.reason,
            'panel_path': _panel_path(input_dir, item.row),
        })
        records.append(record)
    return records


def _write_csv(records: Sequence[Dict[str, str]], path: Path) -> None:
    leading = ('selection_rank', 'scene_category', 'category_rank', 'evidence_tier',
               'selection_reason', 'panel_path')
    trailing = tuple(key for key in records[0] if key not in leading) if records else ()
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=leading + trailing)
        writer.writeheader()
        writer.writerows(records)


def _write_summary(result: CuratedResult, records: Sequence[Dict[str, str]], path: Path) -> None:
    lines = [
        '# Figure 5 curated frame shortlist', '',
        '- Status: **%s**' % ('complete' if result.complete else 'insufficient evidence'),
        '- Fixed quotas: distant recovery 4; small-target recovery 3; localization/FP improvement 3.',
        '- This is GT-assisted qualitative screening and requires manual review.',
        '- Size and orientation improvements are not asserted by the available aggregate evidence.', '',
    ]
    for index, category in enumerate(CATEGORIES):
        lines.extend(['## %s (%d/%d)' % (category.value, result.filled[index], QUOTAS[index]), ''])
        category_rows = [record for record in records if record['scene_category'] == category.value]
        if not category_rows:
            lines.append('- No qualifying frame found.')
        for record in category_rows:
            lines.append('- `%s` (overall rank %s, tier %s): %s Panel: `%s`' % (
                record['sample_token'], record['rank'], record['evidence_tier'],
                record['selection_reason'], record['panel_path'] or 'not rendered'))
        lines.append('')
    if not result.complete:
        lines.append('Shortages (distant/small/error): %s.' % '/'.join(map(str, result.shortages)))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    candidate_csv = args.input_dir / 'figure5_candidates_top100.csv'
    if not candidate_csv.is_file():
        parser.error('missing candidate table: %s' % candidate_csv)
    with candidate_csv.open(newline='', encoding='utf-8') as stream:
        rows = tuple(CandidateRow.from_mapping(row) for row in csv.DictReader(stream))
    result = curate_candidates(rows)
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _records(result, args.input_dir)
    _write_csv(records, output_dir / 'figure5_selected10.csv')
    _write_summary(result, records, output_dir / 'figure5_selected10.md')
    print('Selected %d/10 frames (%s): %s' % (
        len(records), 'complete' if result.complete else 'insufficient evidence', output_dir), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
