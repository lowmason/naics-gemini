import polars as pl
import torch

from naics_embedder.tools.stage4_verification import (
    Stage4VerificationConfig,
    verify_stage4,
)

def _write_embeddings(path, codes, spatial_vectors):
    rows = []
    for idx, (code, vec) in enumerate(zip(codes, spatial_vectors)):
        vec = torch.tensor(vec, dtype=torch.float32)
        time = torch.sqrt(1 + torch.sum(vec**2))
        rows.append(
            {
                'index': idx,
                'level': len(code),
                'code': code,
                'hyp_e0': float(time),
                'hyp_e1': float(vec[0]),
                'hyp_e2': float(vec[1]),
            }
        )
    pl.DataFrame(rows).write_parquet(path)

def _write_distance_matrix(path, codes):
    data = {}
    base = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    ).numpy()
    for idx, code in enumerate(codes):
        data[f'idx_{idx}-code_{code}'] = base[:, idx]
    pl.DataFrame(data).write_parquet(path)

def _write_relations(path):
    pl.DataFrame(
        {
            'idx_i': [0, 0],
            'idx_j': [1, 2],
            'code_i': ['11', '11'],
            'code_j': ['111', '112'],
            'relation_id': [1, 1],
            'relation': ['child', 'child'],
        }
    ).write_parquet(path)

def test_verify_stage4_pass(tmp_path):
    codes = ['11', '111', '112']
    stage3_path = tmp_path / 'stage3.parquet'
    stage4_path = tmp_path / 'stage4.parquet'
    distance_path = tmp_path / 'distance.parquet'
    relations_path = tmp_path / 'relations.parquet'

    _write_embeddings(stage3_path, codes, [(0.0, 0.0), (0.2, 0.0), (0.0, 0.2)])
    _write_embeddings(stage4_path, codes, [(0.0, 0.0), (0.15, 0.0), (0.0, 0.15)])
    _write_distance_matrix(distance_path, codes)
    _write_relations(relations_path)

    cfg = Stage4VerificationConfig(
        max_cophenetic_degradation=0.5,
        max_ndcg_degradation=0.5,
        min_local_improvement=0.0,
        ndcg_k=1,
    )

    result = verify_stage4(
        stage3_path,
        stage4_path,
        distance_path,
        relations_path,
        cfg,
    )

    assert result['passed'] is True
    assert 'cophenetic_correlation' in result['pre']
    assert f'parent_retrieval@{cfg.parent_top_k}' in result['post']

def test_verify_stage4_threshold_failure(tmp_path):
    codes = ['11', '111', '112']
    stage3_path = tmp_path / 'stage3.parquet'
    stage4_path = tmp_path / 'stage4.parquet'
    distance_path = tmp_path / 'distance.parquet'
    relations_path = tmp_path / 'relations.parquet'

    vectors = [(0.0, 0.0), (0.2, 0.0), (0.0, 0.2)]
    _write_embeddings(stage3_path, codes, vectors)
    _write_embeddings(stage4_path, codes, vectors)
    _write_distance_matrix(distance_path, codes)
    _write_relations(relations_path)

    cfg = Stage4VerificationConfig(
        max_cophenetic_degradation=0.0,
        max_ndcg_degradation=0.0,
        min_local_improvement=0.1,
        ndcg_k=1,
    )

    result = verify_stage4(
        stage3_path,
        stage4_path,
        distance_path,
        relations_path,
        cfg,
    )

    assert result['passed'] is False
    assert result['checks']['local_improvement'] is False
