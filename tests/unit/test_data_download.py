from io import BytesIO
from typing import Dict, cast

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

@pytest.mark.unit
def test_read_xlsx_bytes_renames_columns(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_read_excel(buffer, sheet_name, columns, schema_overrides):
        captured['sheet'] = sheet_name
        assert isinstance(buffer, BytesIO)
        assert columns == ['A']
        assert schema_overrides == {'A': pl.Utf8}
        return pl.DataFrame({'A': ['value']})

    monkeypatch.setattr(download_data.pl, 'read_excel', fake_read_excel)

    schema = cast(Dict[str, pl.DataType], {'A': pl.Utf8})
    result = download_data._read_xlsx_bytes(
        data=b'noop',
        sheet='Sheet1',
        schema=schema,
        cols={'A': 'code'},
    )

    assert captured['sheet'] == 'Sheet1'
    assert result.columns == ['code']
    assert result.get_column('code').to_list() == ['value']

@pytest.mark.unit
def test_get_examples_prefers_spreadsheet_entries():
    examples_df = pl.DataFrame({'code': ['111'], 'examples': ['Sheet example']})
    codes = {'111'}
    descriptions_2 = pl.DataFrame(
        {
            'code': ['111', '111', '111'],
            'description_id': [1, 2, 3],
            'description': ['Intro', 'Illustrative Examples:', 'Example bullet'],
        }
    )
    descriptions_3 = descriptions_2.clone()

    examples, descriptions_examples = download_data._get_examples(
        examples_df, codes, descriptions_2, descriptions_3
    )

    assert examples.height == 1
    row = examples.row(0, named=True)
    assert row['examples'] == 'Sheet example'

    assert descriptions_examples.height == 1
    assert descriptions_examples.row(0, named=True)['description_id_min'] == 3

@pytest.mark.unit
def test_get_exclusions_combines_crossrefs_and_descriptions():
    exclusions_df = pl.DataFrame({
        'code': ['111'],
        'excluded': ['See 222 and 333'],
    })
    descriptions_3 = pl.DataFrame(
        {
            'code': ['111', '111'],
            'description_id': pl.Series('description_id', [1, 2], dtype=pl.UInt32),
            'description': ['Some text', 'Excluded 333'],
        }
    )
    codes = {'111', '222', '333'}

    exclusions, descriptions_exclusions = download_data._get_exclusions(
        exclusions_df, descriptions_3, codes
    )

    assert descriptions_exclusions.height == 1
    assert descriptions_exclusions.row(0, named=True)['description_id'] == 2

    assert exclusions.height == 1
    row = exclusions.row(0, named=True)
    assert set(row['excluded_codes']) == {'222', '333'}

@pytest.mark.unit
def test_get_descriptions_2_removes_flagged_sections():
    descriptions_3 = pl.DataFrame(
        {
            'code': ['111', '111', '111'],
            'description_id': [1, 2, 3],
            'description': ['Keep me', 'Drop exclusion', 'Drop example'],
        }
    )
    descriptions_exclusions = pl.DataFrame({'code': ['111'], 'description_id': [2]})
    descriptions_examples = pl.DataFrame({'code': ['111'], 'description_id_min': [3]})

    cleaned = download_data._get_descriptions_2(
        descriptions_3, descriptions_exclusions, descriptions_examples
    )

    assert cleaned.height == 1
    text = cleaned.row(0, named=True)['description']
    assert 'Drop exclusion' not in text
    assert 'Drop example' not in text
