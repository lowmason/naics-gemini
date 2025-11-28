import polars as pl
import pytest

from naics_embedder.graph_model.dataloader import hgcn_streaming_dataset as streaming
from naics_embedder.utils.config import StreamingConfig

def _sample_cached_triplets():
    '''Create cached triplets resembling load_streaming_triplets output.'''

    def _neg(negative_idx):
        return {
            'negative_idx': negative_idx,
            'negative_code': f'N{negative_idx}',
            'negative_relation': negative_idx % 3,
            'negative_distance': float(negative_idx) * 0.1,
            'relation_margin': 0.1 * negative_idx,
            'distance_margin': 0.05 * negative_idx,
        }

    return [
        {
            'anchors': {
                'anchor_idx': 1,
                'anchor_code': '11',
            },
            'positives': {
                'positive_idx': 101,
                'positive_code': '110',
            },
            'negatives': [_neg(201), _neg(202)],
        },
        {
            'anchors': {
                'anchor_idx': 2,
                'anchor_code': '12',
            },
            'positives': {
                'positive_idx': 102,
                'positive_code': '120',
            },
            'negatives': [_neg(301)],
        },
    ]

def _write_triplet_rows(base_dir, anchor_idx, rows):
    '''Persist minimal parquet rows for anchor directory.'''
    anchor_dir = base_dir / f'anchor={anchor_idx}'
    anchor_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(anchor_dir / 'part-0.parquet')

def _row(
    anchor_idx, anchor_code, *, positive_idx, positive_relation, negative_idx, negative_relation
):
    '''Construct a single parquet row with required columns.'''
    return {
        'anchor_idx': anchor_idx,
        'anchor_code': anchor_code,
        'anchor_level': len(anchor_code),
        'positive_idx': positive_idx,
        'positive_code': f'{positive_idx}',
        'positive_relation': positive_relation,
        'positive_distance': 1.0,
        'negative_idx': negative_idx,
        'negative_code': f'N{negative_idx}',
        'negative_relation': negative_relation,
        'negative_distance': 3.0,
        'relation_margin': 0.25,
        'distance_margin': 0.5,
        'excluded': False,
    }

@pytest.mark.unit
def test_streaming_triplet_access(monkeypatch: pytest.MonkeyPatch):
    '''Streaming generator should provide deterministic, indexable triplets.'''
    cached = _sample_cached_triplets()
    cfg = StreamingConfig(n_positives=2, n_negatives=2, seed=7)

    load_calls = []

    def fake_loader(cfg_arg, *, worker_id='Main', allow_cache_save=True, log_stats=True):
        load_calls.append(worker_id)
        return cached

    monkeypatch.setattr(streaming, 'load_streaming_triplets', fake_loader)

    first_pass = list(streaming.create_streaming_generator(cfg))
    second_pass = list(streaming.create_streaming_generator(cfg))

    assert load_calls == ['Main', 'Main']
    assert first_pass == second_pass
    assert first_pass[1]['anchor_idx'] == cached[1]['anchors']['anchor_idx']
    assert set(first_pass[0]['negatives'][0].keys()) == {
        'negative_idx',
        'negative_code',
        'relation_margin',
        'distance_margin',
    }
    assert 'negative_relation' not in first_pass[0]['negatives'][0]

@pytest.mark.unit
def test_relation_filtering(tmp_path):
    '''_build_polars_query should respect relation-based filters.'''
    triplets_dir = tmp_path / 'triplets'
    rows = [
        _row(
            1,
            '11',
            positive_idx=101,
            positive_relation=7,
            negative_idx=201,
            negative_relation=3,
        ),
        _row(
            1,
            '11',
            positive_idx=102,
            positive_relation=9,
            negative_idx=202,
            negative_relation=5,
        ),
    ]
    _write_triplet_rows(triplets_dir, anchor_idx=1, rows=rows)

    cfg = StreamingConfig(
        triplets_parquet=str(triplets_dir),
        n_positives=2,
        n_negatives=2,
        positive_relation=[7],
        negative_relation=[3],
        seed=0,
    )

    df = streaming._build_polars_query(
        cfg,
        codes=['11'],
        code_to_idx={'11': 1},
        worker_id='Test',
    )

    records = df.to_dicts()
    assert len(records) == 1
    entry = records[0]
    assert entry['positives']['positive_idx'] == 101
    assert entry['negatives']['negative_idx'] == 201
    assert entry['negatives']['negative_relation'] == 3

@pytest.mark.unit
def test_level_filtering(tmp_path):
    '''anchor_level filters should limit which parquet shards are scanned.'''
    triplets_dir = tmp_path / 'triplets'
    _write_triplet_rows(
        triplets_dir,
        anchor_idx=1,
        rows=[
            _row(
                1,
                '11',
                positive_idx=101,
                positive_relation=7,
                negative_idx=201,
                negative_relation=3,
            )
        ],
    )
    _write_triplet_rows(
        triplets_dir,
        anchor_idx=2,
        rows=[
            _row(
                2,
                '111',
                positive_idx=102,
                positive_relation=7,
                negative_idx=202,
                negative_relation=3,
            )
        ],
    )

    cfg = StreamingConfig(
        triplets_parquet=str(triplets_dir),
        n_positives=2,
        n_negatives=2,
        anchor_level=[2],
        seed=0,
    )

    df = streaming._build_polars_query(
        cfg,
        codes=['11', '111'],
        code_to_idx={
            '11': 1,
            '111': 2,
        },
        worker_id='Test',
    )

    records = df.to_dicts()
    assert records, 'Expected at least one record for filtered anchors'
    anchor_codes = {row['anchors']['anchor_code'] for row in records}
    assert anchor_codes == {'11'}
