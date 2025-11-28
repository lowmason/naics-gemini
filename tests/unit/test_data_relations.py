import polars as pl
import pytest

from naics_embedder.data import compute_relations

@pytest.mark.unit
def test_get_relations_handles_child_sibling_and_cousin():
    depths = {'A': 0, 'B': 1, 'C': 2, 'D': 1, 'E': 2}
    ancestors = {
        'A': ['A'],
        'B': ['A', 'B'],
        'C': ['A', 'B', 'C'],
        'D': ['A', 'D'],
        'E': ['A', 'D', 'E'],
    }

    assert compute_relations._get_relations('A', 'B', depths, ancestors) == 'child'
    assert compute_relations._get_relations('B', 'C', depths, ancestors) == 'child'
    assert compute_relations._get_relations('B', 'D', depths, ancestors) == 'sibling'
    assert compute_relations._get_relations('C', 'E', depths, ancestors) == 'cousin'

@pytest.mark.unit
def test_get_relation_matrix_is_symmetric():
    df = pl.DataFrame({
        'code_i': ['11', '11'],
        'code_j': ['21', '31'],
        'relation_id': [1, 2],
    })

    matrix = compute_relations._get_relation_matrix(df)

    assert matrix.shape == (3, 3)
    cols = matrix.columns
    assert matrix[cols[1]][0] == pytest.approx(1.0)
    assert matrix[cols[0]][1] == pytest.approx(1.0)
    assert matrix[cols[2]][0] == pytest.approx(2.0)
    assert matrix[cols[0]][2] == pytest.approx(2.0)

@pytest.mark.unit
def test_get_relations_handles_extended_family_names():
    depths = {
        'A': 0,
        'B': 1,
        'C': 1,
        'D': 2,
        'E': 3,
        'F': 4,
        'G': 2,
        'H': 3,
        'I': 4,
    }
    ancestors = {
        'A': ['A'],
        'B': ['A', 'B'],
        'C': ['A', 'C'],
        'D': ['A', 'C', 'D'],
        'E': ['A', 'C', 'D', 'E'],
        'F': ['A', 'C', 'D', 'E', 'F'],
        'G': ['A', 'B', 'G'],
        'H': ['A', 'B', 'G', 'H'],
        'I': ['A', 'B', 'G', 'H', 'I'],
    }

    grand_niece = compute_relations._get_relations('B', 'F', depths, ancestors)
    assert grand_niece == 'grand-grand-nephew/niece'

    removed = compute_relations._get_relations('D', 'I', depths, ancestors)
    assert removed == 'cousin_2_times_removed'

@pytest.mark.unit
def test_get_exclusions_matches_description_codes(monkeypatch: pytest.MonkeyPatch):
    descriptions_df = pl.DataFrame(
        {
            'code': ['111111', '222222', '333333'],
            'excluded': ['desc', None, 'desc'],
            'excluded_codes': [['222222'], None, ['111111']],
        }
    )

    monkeypatch.setattr(
        'naics_embedder.data.compute_relations.pl.read_parquet',
        lambda *_args, **_kwargs: descriptions_df,
    )

    relations_df = pl.DataFrame({
        'code_i': ['111111', '333333'],
        'code_j': ['222222', '999999'],
    })

    exclusions = compute_relations._get_exclusions(relations_df)

    assert exclusions.height == 1
    row = exclusions.row(0, named=True)
    assert row['code_i'] == '111111'
    assert row['code_j'] == '222222'
