from dataclasses import dataclass, replace
from typing import Final, Mapping, Sequence, Tuple

from .candidates import CandidateEvaluation
from .domain import Callout, CalloutKind, FrameRecord, ImageRoi, SampleToken


FINAL_SAMPLE_TOKENS: Final[Tuple[SampleToken, ...]] = (
    SampleToken('9a476217cc324813a5760c9852643324'),
    SampleToken('3425c66163af46cca4c96006d425e0eb'),
    SampleToken('2128fc958907421ca888ab014a73348a'),
)
BEV_MIN: Final[float] = -50.0
BEV_MAX: Final[float] = 50.0
CROP_CONTEXT_M: Final[float] = 4.0
CALLOUT_VISUAL_PADDING_M: Final[float] = 0.75
MAX_CALLOUTS: Final[int] = 2


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class FinalRowSelection:
    row_number: int
    frame: FrameRecord
    callouts: Tuple[Callout, ...]
    crop: ImageRoi


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class FinalSelectionError(ValueError):
    token: SampleToken
    detail: str

    def __str__(self) -> str:
        return '%s: %s' % (self.token, self.detail)


def filter_predictions(frame: FrameRecord, threshold: float) -> FrameRecord:
    return replace(
        frame,
        source_only=tuple(item for item in frame.source_only if item.score >= threshold),
        codemerge=tuple(item for item in frame.codemerge if item.score >= threshold),
        refuse_tta=tuple(item for item in frame.refuse_tta if item.score >= threshold),
    )


def _visible(callout: Callout) -> bool:
    x_min, y_min, x_max, y_max = callout.roi
    return BEV_MIN <= x_min < x_max <= BEV_MAX and BEV_MIN <= y_min < y_max <= BEV_MAX


def _callouts(candidate: CandidateEvaluation, row_number: int) -> Tuple[Callout, ...]:
    visible = tuple(item for item in candidate.available_callouts if _visible(item))
    if row_number == 1:
        selected = tuple(item for item in visible if item.kind is CalloutKind.FAR_RANGE_RECOVERY)
    elif row_number == 2:
        selected = tuple(item for item in visible if item.kind is CalloutKind.SMALL_OBJECT_RECOVERY)
    else:
        localized = tuple(item for item in visible if item.kind is CalloutKind.BETTER_LOCALIZATION)
        selected = localized or tuple(
            item for item in visible if item.kind is CalloutKind.FALSE_POSITIVE_REDUCTION
        )
    selected = tuple(sorted(selected, key=lambda item: (
        -item.distance_m, item.class_name.value, item.roi,
    )))[:MAX_CALLOUTS]
    if not selected:
        raise FinalSelectionError(candidate.sample_token, 'no visible evidence for fixed row purpose')
    return tuple(replace(item, roi=(
        max(BEV_MIN, item.roi[0] - CALLOUT_VISUAL_PADDING_M),
        max(BEV_MIN, item.roi[1] - CALLOUT_VISUAL_PADDING_M),
        min(BEV_MAX, item.roi[2] + CALLOUT_VISUAL_PADDING_M),
        min(BEV_MAX, item.roi[3] + CALLOUT_VISUAL_PADDING_M),
    )) for item in selected)


def _bounded_interval(center: float, side: float) -> Tuple[float, float]:
    lower = center - side / 2.0
    upper = center + side / 2.0
    if lower < BEV_MIN:
        upper += BEV_MIN - lower
        lower = BEV_MIN
    if upper > BEV_MAX:
        lower -= upper - BEV_MAX
        upper = BEV_MAX
    return lower, upper


def _crop(callouts: Sequence[Callout], minimum_side: float) -> ImageRoi:
    x_min = max(BEV_MIN, min(item.roi[0] for item in callouts))
    y_min = max(BEV_MIN, min(item.roi[1] for item in callouts))
    x_max = min(BEV_MAX, max(item.roi[2] for item in callouts))
    y_max = min(BEV_MAX, max(item.roi[3] for item in callouts))
    side = min(BEV_MAX - BEV_MIN, max(
        minimum_side,
        x_max - x_min + 2.0 * CROP_CONTEXT_M,
        y_max - y_min + 2.0 * CROP_CONTEXT_M,
    ))
    crop_x_min, crop_x_max = _bounded_interval((x_min + x_max) / 2.0, side)
    crop_y_min, crop_y_max = _bounded_interval((y_min + y_max) / 2.0, side)
    return crop_x_min, crop_y_min, crop_x_max, crop_y_max


def select_final_rows(
        candidates: Mapping[SampleToken, CandidateEvaluation]) -> Tuple[FinalRowSelection, ...]:
    rows = []
    for row_number, token in enumerate(FINAL_SAMPLE_TOKENS, 1):
        if token not in candidates:
            raise FinalSelectionError(token, 'fixed token is absent from evaluated artifacts')
        candidate = candidates[token]
        callouts = _callouts(candidate, row_number)
        rows.append(FinalRowSelection(
            row_number, candidate.frame, callouts,
            _crop(callouts, 40.0 if row_number == 1 else 30.0),
        ))
    return tuple(rows)
