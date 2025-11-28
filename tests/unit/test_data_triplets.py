import polars as pl
import pytest

from naics_embedder.data import create_triplets

@pytest.mark.unit
def test_get_triplets_builds_valid_row():
    positives_df = pl.DataFrame(
        {
            'anchor_idx': [0],
            'positive_idx': [1],
            'anchor_code': ['11'],
            'positive_code': ['111'],
        }
    )
    negatives_df = pl.DataFrame(
        {
            'negative_idx': [2],
            'positive_code': ['111'],
            'negative_code': ['211'],
        }
    )
    positive_distances_df = pl.DataFrame(
        {
            'anchor_code': ['11'],
            'positive_code': ['111'],
            'positive_relation': [1],
            'positive_distance': [0.5],
        }
    )
    negative_distances_df = pl.DataFrame(
        {
            'anchor_code': ['11'],
            'negative_code': ['211'],
            'negative_relation': [3],
            'negative_distance': [2.0],
        }
    )

    triplets = create_triplets._get_triplets(
        positives_df, negatives_df, positive_distances_df, negative_distances_df
    )

    assert triplets.height == 1
    row = triplets.row(0, named=True)
    assert row['anchor_idx'] == 0
    assert row['negative_idx'] == 2
    assert row['relation_margin'] > 0
    assert row['distance_margin'] > 0
    assert row['margin'] > 0
