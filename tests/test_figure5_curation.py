import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'figure5_curation.py'
SPEC = importlib.util.spec_from_file_location('figure5_curation_under_test', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CURATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CURATION
SPEC.loader.exec_module(CURATION)


def _row(rank, token, category):
    values = {
        'rank': str(rank), 'sample_token': token, 'recovered_from_source': '0',
        'recovered_from_codemerge': '0', 'false_positive_removed': '0',
        'far_small_detected': '0', 'better_localization': '0',
        'involved_classes': '', 'primary_distances': '', 'memberships': '',
    }
    if category == 'far':
        values.update(recovered_from_source='2', recovered_from_codemerge='1',
                      primary_distances='45.0', memberships='far_range_recovery')
    elif category == 'small':
        values.update(recovered_from_source='1', recovered_from_codemerge='1',
                      involved_classes='pedestrian', memberships='small_object_recovery')
    elif category == 'error':
        values.update(better_localization='1')
    return CURATION.CandidateRow.from_mapping(values)


class CurateCandidatesTest(unittest.TestCase):
    def test_fills_exact_433_quotas_with_unique_tokens(self):
        rows = ([_row(index + 1, 'far%d' % index, 'far') for index in range(4)] +
                [_row(index + 5, 'small%d' % index, 'small') for index in range(3)] +
                [_row(index + 8, 'error%d' % index, 'error') for index in range(3)])
        result = CURATION.curate_candidates(rows)
        self.assertTrue(result.complete)
        self.assertEqual(result.filled, (4, 3, 3))
        self.assertEqual(len({item.row.token for item in result.selections}), 10)

    def test_one_multi_category_frame_is_assigned_only_once(self):
        overlap = dict(_row(1, 'overlap', 'far').raw)
        overlap.update(involved_classes='pedestrian', memberships='far_range_recovery;small_object_recovery')
        rows = [CURATION.CandidateRow.from_mapping(overlap)]
        rows.extend(_row(index + 2, 'far%d' % index, 'far') for index in range(4))
        rows.extend(_row(index + 6, 'small%d' % index, 'small') for index in range(3))
        rows.extend(_row(index + 9, 'error%d' % index, 'error') for index in range(3))
        result = CURATION.curate_candidates(reversed(rows))
        self.assertTrue(result.complete)
        self.assertEqual(len({item.row.token for item in result.selections}), 10)

    def test_strict_progressive_far_evidence_beats_relaxed_evidence(self):
        relaxed = dict(_row(1, 'relaxed', 'far').raw)
        relaxed['recovered_from_codemerge'] = '0'
        result = CURATION.curate_candidates([
            CURATION.CandidateRow.from_mapping(relaxed), _row(2, 'strict', 'far')])
        distant = [item for item in result.selections if item.category == CURATION.SceneCategory.DISTANT]
        self.assertEqual(distant[0].row.token, 'strict')
        self.assertEqual(distant[0].tier, 0)

    def test_reports_shortages_without_padding(self):
        result = CURATION.curate_candidates([_row(1, 'only', 'error')])
        self.assertFalse(result.complete)
        self.assertEqual(result.filled, (0, 0, 1))
        self.assertEqual(result.shortages, (4, 3, 2))
        self.assertEqual(len(result.selections), 1)


if __name__ == '__main__':
    unittest.main()
