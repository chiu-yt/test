"""Process-owned Figure 6 observation of the unchanged online training stream."""

import json
import logging
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np
import torch

from .figure6_artifacts import write_record
from .figure6_runtime import Figure6RuntimeCollector
from .figure6_schema import ArtifactError, FIXED_TOKENS, Occurrence


class Figure6StreamCapture:
    """Own one collector and finalize its selected transactions after optimize."""

    def __init__(self, run_cfg, provenance: Mapping[str, str], location: Tuple[Path, int]) -> None:
        capture_cfg = run_cfg.TTA.FIGURE6_CAPTURE
        tokens = capture_cfg.get('TOKENS', FIXED_TOKENS)
        if not isinstance(tokens, (list, tuple)) or not tokens or any(
                not isinstance(token, str) or token not in FIXED_TOKENS for token in tokens):
            raise ArtifactError('Figure 6 TOKENS must select fixed Figure 5 nuScenes tokens')
        self.tokens = frozenset(tokens)
        ckpt_dir, local_rank = location
        self.output = Path(capture_cfg['OUTPUT_DIR']) if capture_cfg.get('OUTPUT_DIR') else (
            ckpt_dir.parent / 'figure6_capture')
        self.global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else local_rank
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        data_cfg = run_cfg.DATA_CONFIG
        corruption = data_cfg.get('CORRUPTION', {})
        sparsity = corruption.get('LIDAR_SPARSITY', {})
        point_cloud_range = data_cfg.get('POINT_CLOUD_RANGE', None)
        self.protocol = {
            'method': str(run_cfg.TTA.get('METHOD', 'mos')).lower(),
            'config': provenance.get('config') or 'unavailable',
            'command': provenance.get('command') or 'unavailable',
            'source_checkpoint': provenance.get('source_checkpoint') or 'unavailable',
            'fixed_seed': provenance.get('fixed_seed') or 'unavailable',
            'point_cloud_range': (
                json.dumps([float(value) for value in point_cloud_range])
                if point_cloud_range is not None else 'unavailable'
            ),
            'max_sweeps': str(data_cfg.get('MAX_SWEEPS', 'unavailable')),
            'corruption.enabled': str(corruption.get('ENABLED', False)),
            'corruption.apply_in': str(corruption.get('APPLY_IN', ['test'])),
            'lidar_sparsity.enabled': str(sparsity.get('ENABLED', False)),
            'lidar_sparsity.mode': str(sparsity.get('MODE', 'random_keep')),
            'lidar_sparsity.severity': str(sparsity.get('SEVERITY', 1)),
            'lidar_sparsity.seed': str(sparsity.get('SEED', 0)),
            'lidar_sparsity.seed_policy': (
                'sample_token_hash' if sparsity.get('MODE', 'random_keep') == 'density_dec_global'
                else 'numpy_global_rng'),
            'global_rank': str(self.global_rank),
            'world_size': str(self.world_size),
        }
        if any(not value for value in self.protocol.values()):
            raise ArtifactError('Figure 6 protocol values must be nonempty')
        self.collector = Figure6RuntimeCollector()
        self.logger = logging.Logger('figure6.rank_%05d' % self.global_rank, logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
        self.logger.addHandler(handler)
        self.logger.info('Capture enabled output=%s tokens=%s', self.output, sorted(self.tokens))

    def begin(self, batch_dict, progress: Tuple[int, int, int, int]) -> None:
        metadata = batch_dict.get('metadata')
        if not isinstance(metadata, (list, tuple, np.ndarray)):
            raise ArtifactError('Figure 6 requires metadata as a collated sequence of token mappings')
        selected = [index for index, entry in enumerate(metadata)
                    if isinstance(entry, Mapping) and isinstance(entry.get('token'), str)
                    and entry['token'] in self.tokens]
        if not selected:
            return
        epoch, iteration, samples_seen, batch_size = progress
        frames = batch_dict.get('frame_id')
        if len(metadata) != batch_size or any(
                not isinstance(entry, Mapping) or not isinstance(entry.get('token'), str)
                or not entry['token'] for entry in metadata):
            raise ArtifactError('Figure 6 selected batch has missing/malformed metadata tokens')
        if not isinstance(frames, (list, tuple, np.ndarray)) or len(frames) != batch_size:
            raise ArtifactError('Figure 6 selected batch has missing/malformed frame_id sequence')
        if any(not isinstance(frames[index], str) or not frames[index] for index in selected):
            raise ArtifactError('Figure 6 selected sample requires a nonempty frame_id')
        identities = tuple(Occurrence(
            token=metadata[index]['token'], frame_id=frames[index], epoch=epoch,
            accumulated_iter_before=iteration, samples_seen=samples_seen,
            global_rank=self.global_rank, world_size=self.world_size,
            batch_index=index, batch_size=batch_size,
        ) for index in selected)
        self.collector.begin(identities, self.protocol)

    def finish(self) -> None:
        for record in self.collector.finalize():
            destination = write_record(self.output, record)
            self.logger.info('Capture written token=%s occurrence=%s path=%s',
                             record.identity.token, record.identity.selection_key, destination)
