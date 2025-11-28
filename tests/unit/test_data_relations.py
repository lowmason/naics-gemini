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
