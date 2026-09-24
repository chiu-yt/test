from dataclasses import replace

import numpy as np

from test_spcra_k4_evidence import load_contract_module
from pcdet.utils.figure6_schema import Occurrence
from pcdet.utils.figure7_schema import CurrentPoints, Observation, Provenance


def observation(token: str, stable: bool = True, score: float = .9) -> Observation:
    core = load_contract_module('spcra_k4_core')
    codec = load_contract_module('spcra_k4_evidence')
    seeding = load_contract_module('spcra_k4_seeding')
    boxes = np.array([[10., 0., 0., 2., 2., 2., 0.]])
    reference = core.Predictions(boxes, np.array([8]), np.array([score]))
    views = tuple(core.Predictions(
        boxes + (0 if stable or index == 0 else np.array([50., 0, 0, 0, 0, 0, 0])),
        np.array([8]), np.array([score]),
    ) for index in range(4))
    samples = seeding.ViewSamples(
        indices=tuple(np.arange(3) for _ in range(4)),
        transforms=tuple(np.eye(4, dtype=np.float32) for _ in range(4)),
        fingerprints=('a', 'b', 'c', 'd'), attempts=(0, 0, 0, 0), law_identifier='test-law',
    )
    evidence = codec.build_k4_evidence(
        codec.K4EvidenceInput(reference, views, samples),
        core.compute_k4_reliability(reference, views, core.K4Policy()),
    )
    points = np.arange(12, dtype=np.float32).reshape(3, 4)
    return Observation(
        Occurrence(token, 'frame-' + token, 0, 0, 0, 0, 1, 0, 1), 0,
        evidence, CurrentPoints(points, tuple(points for _ in range(4)), np.eye(4)),
    )


def provenance() -> Provenance:
    return Provenance('cfg.yaml', 'python train.py', 'source.pth', '1024')


def with_reliability(frame: Observation, reliability: float) -> Observation:
    assert frame.evidence is not None
    return replace(frame, evidence=replace(
        frame.evidence, reliability=np.array([reliability]),
    ))
