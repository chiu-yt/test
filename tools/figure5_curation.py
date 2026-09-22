from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, Sequence, Tuple


class SceneCategory(str, Enum):
    DISTANT = 'distant_target_recovery'
    SMALL = 'small_target_recovery'
    ERROR = 'localization_or_fp_improvement'


CATEGORIES = (SceneCategory.DISTANT, SceneCategory.SMALL, SceneCategory.ERROR)
QUOTAS = (4, 3, 3)
SMALL_CLASSES = frozenset(('pedestrian', 'bicycle', 'motorcycle', 'traffic_cone'))


def _integer(row: Mapping[str, str], name: str) -> int:
    try:
        return int(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('invalid integer field %s for token %s' % (
            name, row.get('sample_token', '<unknown>'))) from error


def _values(value: str) -> Tuple[str, ...]:
    return tuple(item for item in value.split(';') if item)


@dataclass(frozen=True)
class CandidateRow:
    rank: int
    token: str
    recovered_source: int
    recovered_codemerge: int
    false_positive_removed: int
    far_small_detected: int
    better_localization: int
    classes: Tuple[str, ...]
    distances: Tuple[float, ...]
    memberships: Tuple[str, ...]
    raw: Mapping[str, str]

    @classmethod
    def from_mapping(cls, row: Mapping[str, str]) -> 'CandidateRow':
        token = row.get('sample_token', '')
        if not token:
            raise ValueError('candidate row has no sample_token')
        try:
            distances = tuple(float(value) for value in _values(row.get('primary_distances', '')))
        except ValueError as error:
            raise ValueError('invalid primary_distances for token %s' % token) from error
        return cls(
            rank=_integer(row, 'rank'), token=token,
            recovered_source=_integer(row, 'recovered_from_source'),
            recovered_codemerge=_integer(row, 'recovered_from_codemerge'),
            false_positive_removed=_integer(row, 'false_positive_removed'),
            far_small_detected=_integer(row, 'far_small_detected'),
            better_localization=_integer(row, 'better_localization'),
            classes=_values(row.get('involved_classes', '')),
            distances=distances,
            memberships=_values(row.get('memberships', '')),
            raw=dict(row),
        )


@dataclass(frozen=True)
class Eligibility:
    row: CandidateRow
    category: SceneCategory
    tier: int
    strength: int
    reason: str


@dataclass(frozen=True)
class CuratedResult:
    selections: Tuple[Eligibility, ...]
    filled: Tuple[int, int, int]

    @property
    def complete(self) -> bool:
        return self.filled == QUOTAS

    @property
    def shortages(self) -> Tuple[int, int, int]:
        return (QUOTAS[0] - self.filled[0], QUOTAS[1] - self.filled[1],
                QUOTAS[2] - self.filled[2])


def candidate_eligibilities(row: CandidateRow) -> Tuple[Eligibility, ...]:
    eligible = []
    farthest = max(row.distances, default=0.0)
    far_recovery = 'far_range_recovery' in row.memberships and row.recovered_source > 0
    progressive = row.recovered_source > row.recovered_codemerge > 0
    if far_recovery and farthest > 30.0:
        tier = 0 if progressive and farthest > 40.0 else 1 if progressive else 2 if farthest > 40.0 else 3
        reason = ('Source-only misses distant GT; CodeMerge reduces but does not eliminate the miss; '
                  'ReFuse matches the remaining GT.' if progressive else
                  'ReFuse matches a >30 m GT missed by Source-only; inspect CodeMerge manually.')
        eligible.append(Eligibility(row, SceneCategory.DISTANT, tier,
                                    int(farthest) + 10 * row.recovered_source, reason))

    small_classes = sorted(SMALL_CLASSES.intersection(row.classes))
    small_recovery = 'small_object_recovery' in row.memberships
    if small_classes and small_recovery and row.recovered_source > 0:
        missed_by_both = row.recovered_codemerge > 0
        tier = 0 if missed_by_both else 1
        reason = ('ReFuse matches sparse-S5 small-class GT missed by Source-only%s: %s.' % (
            ' and CodeMerge' if missed_by_both else '', ', '.join(small_classes)))
        eligible.append(Eligibility(row, SceneCategory.SMALL, tier,
                                    10 * row.far_small_detected + row.recovered_source, reason))

    if row.better_localization > 0 or row.false_positive_removed > 0:
        if row.better_localization > 0 and row.false_positive_removed > 0:
            tier = 0
            reason = ('ReFuse has GT-relative center improvement and fewer unmatched predictions; '
                      'the latter remains an FP proxy.')
        elif row.better_localization > 0:
            tier = 1
            reason = 'ReFuse has at least 0.5 m lower GT-relative center error.'
        else:
            tier = 2
            reason = 'ReFuse has fewer unmatched predictions; this is an FP proxy, not confirmed FP removal.'
        eligible.append(Eligibility(row, SceneCategory.ERROR, tier,
                                    10 * row.better_localization + row.false_positive_removed, reason))
    return tuple(eligible)


def _quality(items: Sequence[Eligibility]) -> Tuple[object, ...]:
    return (
        sum(item.tier for item in items),
        -sum(item.strength for item in items),
        sum(item.row.rank for item in items),
        tuple(sorted((CATEGORIES.index(item.category), item.row.token) for item in items)),
    )


def curate_candidates(rows: Iterable[CandidateRow]) -> CuratedResult:
    ordered = sorted(rows, key=lambda row: (row.rank, row.token))
    if len({row.token for row in ordered}) != len(ordered):
        raise ValueError('candidate tokens must be unique')
    states: Dict[Tuple[int, int, int], Tuple[Eligibility, ...]] = {(0, 0, 0): ()}
    for row in ordered:
        previous = tuple(states.items())
        for counts, selected in previous:
            for item in candidate_eligibilities(row):
                index = CATEGORIES.index(item.category)
                if counts[index] >= QUOTAS[index]:
                    continue
                next_counts = (
                    counts[0] + int(index == 0), counts[1] + int(index == 1),
                    counts[2] + int(index == 2),
                )
                proposal = selected + (item,)
                current = states.get(next_counts)
                if current is None or _quality(proposal) < _quality(current):
                    states[next_counts] = proposal
    if QUOTAS in states:
        counts = QUOTAS
    else:
        counts = min(states, key=lambda value: (
            -sum(value), -value[0], -value[1], -value[2], _quality(states[value])))
    selected = tuple(sorted(states[counts], key=lambda item: (
        CATEGORIES.index(item.category), item.tier, -item.strength,
        item.row.rank, item.row.token)))
    return CuratedResult(selected, counts)
