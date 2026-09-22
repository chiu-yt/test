"""Build shell-safe Figure 6 run provenance at the training CLI boundary."""

import shlex
import sys
from typing import Optional, TypedDict


class CaptureProvenance(TypedDict):
    config: str
    command: str
    source_checkpoint: str
    fixed_seed: str


def build_capture_provenance(config: str, checkpoint: Optional[str],
                             fixed_seed: bool) -> CaptureProvenance:
    return CaptureProvenance(
        config=config,
        command=shlex.join([sys.executable] + sys.argv),
        source_checkpoint=checkpoint or 'unavailable',
        fixed_seed=str(fixed_seed),
    )
