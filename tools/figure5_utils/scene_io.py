"""Read nuScenes JSON metadata without importing the SDK."""
import json
from pathlib import Path
from typing import Dict, Mapping

from .domain import SampleToken, SceneToken
from .input_types import ArtifactFormatError, DuplicateTokenError, MissingTokenError, SceneMetadata


def _table(path: Path) -> Dict[str, dict]:
    with path.open() as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ArtifactFormatError(str(path), 'expected a JSON table')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('token'), str) or not row['token']:
            raise MissingTokenError(str(path), 'missing token')
        if row['token'] in indexed:
            raise DuplicateTokenError(str(path), row['token'])
        indexed[row['token']] = row
    return indexed


def load_scene_metadata(table_directory: Path) -> Mapping[SampleToken, SceneMetadata]:
    """Use scene first_sample_token and sample next links for zero-based indices."""
    samples = _table(table_directory / 'sample.json')
    scenes = _table(table_directory / 'scene.json')
    result = {}
    for scene_token, scene in scenes.items():
        token = scene.get('first_sample_token')
        name = scene.get('name')
        if not isinstance(token, str) or not isinstance(name, str):
            raise ArtifactFormatError(scene_token, 'invalid scene name or first sample')
        index = 0
        while token:
            if token not in samples:
                raise MissingTokenError(scene_token, token)
            if token in result:
                raise DuplicateTokenError(scene_token, 'repeated sample in scene chain: ' + token)
            sample = samples[token]
            if sample.get('scene_token') != scene_token:
                raise ArtifactFormatError(token, 'sample belongs to a different scene')
            result[SampleToken(token)] = SceneMetadata(SceneToken(scene_token), name, index)
            token = sample.get('next')
            if not isinstance(token, str):
                raise ArtifactFormatError(scene_token, 'missing next sample link')
            index += 1
    return result
