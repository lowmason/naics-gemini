import polars as pl
import pytest

from naics_embedder.data import download_data

@pytest.mark.unit
def test_get_titles_normalizes_index_and_level():
    titles_df = pl.DataFrame(
        {
            'index': [1, 2],
            'code': ['31', '311'],
            'title': ['Manufacturing', 'Food Manufacturing'],
        }
    )

    titles, codes = download_data._get_titles(titles_df)

    assert titles.shape == (2, 4)
    assert titles.get_column('index').to_list() == [0, 1]
    assert titles.get_column('level').to_list() == [2, 3]
    assert codes == {'31', '311'}

@pytest.mark.unit
def test_get_descriptions_filters_cross_references():
    descriptions_df = pl.DataFrame(
        {
            'code': ['31111'],
            'description': [
                'Primary line\r\nCross-References.\r\nThe Sector as a Whole\r\nFinal line',
            ],
        }
    )

    _, descriptions_clean = download_data._get_descriptions_1(descriptions_df)

    cleaned = descriptions_clean.get_column('description').to_list()
    assert cleaned == ['Primary line', 'Final line']
    ids = descriptions_clean.get_column('description_id').to_list()
    assert ids == sorted(ids)
