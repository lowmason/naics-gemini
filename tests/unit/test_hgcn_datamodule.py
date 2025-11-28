from typing import Any, Dict, List

import pytest
import torch

from naics_embedder.graph_model.dataloader import hgcn_datamodule as hgcn_module
from naics_embedder.graph_model.dataloader.hgcn_datamodule import HGCNDataModule
from naics_embedder.utils.config import GraphConfig, StreamingConfig

def _graph_cfg(**overrides) -> GraphConfig:
    '''Create a GraphConfig tailored for small deterministic tests.'''
    defaults: Dict[str, Any] = {
        'batch_size': 2,
        'num_workers': 0,
        'shuffle': False,
        'k_total': 2,
        'n_positive_samples': 4,
    }
    defaults.update(overrides)
    return GraphConfig(**defaults)

def _make_triplet(anchor_idx: int, positive_idx: int, n_negatives: int = 2) -> Dict[str, Any]:
    negatives = []
    for offset in range(n_negatives):
        neg_idx = positive_idx * 10 + offset
        negatives.append(
            {
                'negative_idx': neg_idx,
                'negative_code': f'N{neg_idx}',
                'negative_relation': offset + 1,
                'negative_distance': float(offset) + 0.5,
                'relation_margin': float(offset) + 0.1,
                'distance_margin': float(offset) + 0.2,
            }
        )

    return {
        'anchors': {
            'anchor_idx': anchor_idx,
            'anchor_code': f'A{anchor_idx}'
        },
        'positives': {
            'positive_idx': positive_idx,
            'positive_code': f'P{positive_idx}'
        },
        'negatives': negatives,
    }

@pytest.mark.unit
class TestHGCNDataModule:

    def test_datamodule_setup(self, monkeypatch: pytest.MonkeyPatch):
        '''DataModule should split cached data into train/val datasets.'''
        sample_data = [_make_triplet(idx, 100 + idx) for idx in range(4)]
        loader_calls: List[StreamingConfig] = []

        def fake_loader(cfg, *args, **kwargs):
            loader_calls.append(cfg)
            return sample_data

        monkeypatch.setattr(hgcn_module, 'load_streaming_triplets', fake_loader)

        dm = HGCNDataModule(_graph_cfg(), val_split=0.25)
        dm.prepare_data()
        dm.setup('fit')

        assert len(loader_calls) == 1  # setup reuses prepared cache
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None
        assert len(dm._train_dataset) == 3
        assert len(dm._val_dataset) == 1

    def test_batch_structure(self, monkeypatch: pytest.MonkeyPatch):
        '''Batches should contain padded negatives and expected tensor dtypes.'''
        sample_data = [
            _make_triplet(0, 10, n_negatives=1),
            _make_triplet(1, 11, n_negatives=3),
        ]
        monkeypatch.setattr(hgcn_module, 'load_streaming_triplets', lambda *_, **__: sample_data)

        dm = HGCNDataModule(_graph_cfg(), val_split=0.0)
        dm.prepare_data()
        dm.setup('fit')

        batch = next(iter(dm.train_dataloader()))

        expected_keys = {
            'anchor_idx',
            'positive_idx',
            'negative_indices',
            'negative_relation_ids',
            'negative_distances',
            'negative_relation_margins',
            'negative_distance_margins',
        }
        assert set(batch.keys()) == expected_keys

        assert batch['anchor_idx'].shape == torch.Size([2])
        assert batch['positive_idx'].shape == torch.Size([2])
        assert batch['negative_indices'].shape == torch.Size([2, 3])
        assert batch['negative_relation_ids'].dtype == torch.long
        assert batch['negative_distances'].dtype == torch.float32

        # Padding should duplicate the final element for rows with fewer negatives.
        assert batch['negative_indices'][0, -1].item() == batch['negative_indices'][0, 0].item()

    def test_curriculum_filtering(self, monkeypatch: pytest.MonkeyPatch):
        '''Loader config filters should be forwarded into the streaming config.'''
        sample_data = [_make_triplet(0, 10)]
        captured_cfg: Dict[str, StreamingConfig] = {}

        def fake_loader(cfg, *args, **kwargs):
            captured_cfg['value'] = cfg
            return sample_data

        monkeypatch.setattr(hgcn_module, 'load_streaming_triplets', fake_loader)

        dm = HGCNDataModule(_graph_cfg())
        dm.loader_cfg.anchor_level = [3]
        dm.loader_cfg.relation_margin = [1]
        dm.loader_cfg.positive_relation = [5]
        dm.loader_cfg.negative_relation = [7]
        dm.loader_cfg.n_positive_samples = 5
        dm.loader_cfg.n_negatives = 2
        dm._streaming_cfg = hgcn_module._streaming_cfg_from_loader(dm.loader_cfg)

        dm.prepare_data()

        cfg = captured_cfg['value']
        assert cfg.anchor_level == [3]
        assert cfg.relation_margin == [1]
        assert cfg.positive_relation == [5]
        assert cfg.negative_relation == [7]
        assert cfg.n_positives == 5
        assert cfg.n_negatives == 2
