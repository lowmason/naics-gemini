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

@pytest.mark.unit
def test_get_distances_filters_max_distance_row():
    distances_df = pl.DataFrame(
        {
            'idx_i': [0, 0],
            'idx_j': [1, 2],
            'code_i': ['11', '11'],
            'code_j': ['111', '112'],
            'distance': [0.5, 99.0],
        }
    )

    positives, negatives = create_triplets._get_distances(distances_df)

    assert positives.height == 1
    assert positives.get_column('positive_code').to_list() == ['111']
    assert negatives.height == 2
    assert set(negatives.get_column('negative_code')) == {'111', '112'}

@pytest.mark.unit
def test_get_pairs_deduplicates_and_preserves_order():
    distances_df = pl.DataFrame(
        {
            'idx_i': [0, 0, 0],
            'idx_j': [1, 1, 2],
            'code_i': ['11', '11', '11'],
            'code_j': ['111', '111', '211'],
            'distance': [0.5, 0.5, 3.0],
        }
    )

    positives, negatives = create_triplets._get_pairs(distances_df)

    assert positives.height == 1
    assert positives.row(0, named=True)['positive_code'] == '111'
    assert negatives.height == 2
    assert negatives.get_column('negative_code').to_list() == ['111', '211']

@pytest.mark.unit
def test_get_relations_merges_distance_information():
    relations_df = pl.DataFrame(
        {
            'code_i': ['11', '11'],
            'code_j': ['111', '112'],
            'relation': [1, 3],
        }
    )
    distances_df = pl.DataFrame(
        {
            'idx_i': [0, 0],
            'idx_j': [1, 2],
            'code_i': ['11', '11'],
            'code_j': ['111', '112'],
            'distance': [0.5, 2.0],
        }
    )

    positives, negatives = create_triplets._get_relations(relations_df, distances_df)

    assert positives.height == 1
    pos_row = positives.row(0, named=True)
    assert pos_row['positive_relation'] == 1
    assert pos_row['positive_distance'] == pytest.approx(0.5)

    assert negatives.height == 2
    neg_codes = negatives.get_column('negative_code').to_list()
    assert '112' in neg_codes

@pytest.mark.unit
def test_get_triplets_sets_excluded_margins():
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
            'negative_relation': [5],
            'negative_distance': [0.0],
        }
    )

    triplets = create_triplets._get_triplets(
        positives_df, negatives_df, positive_distances_df, negative_distances_df
    )

    assert triplets.height == 1
    row = triplets.row(0, named=True)
    assert row['excluded'] is True
    assert row['relation_margin'] == pytest.approx(0.1)
    assert row['distance_margin'] == pytest.approx(0.1)
    assert row['margin'] > 0
