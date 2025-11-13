# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from io import BytesIO
from typing import Dict, Optional, Set, Tuple

import polars as pl

from naics_gemini.utils.config import DownloadConfig, load_config
from naics_gemini.utils.utilities import download_with_retry as _download_with_retry
from naics_gemini.utils.utilities import make_directories
from naics_gemini.utils.utilities import parquet_stats as _parquet_stats

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------------

def _read_xlsx_bytes(
    data: bytes,
    sheet: str,
    schema: Dict[str, pl.DataType],
    cols: Dict[str, str]
) -> pl.DataFrame:
    
    '''
    Read Excel data from bytes into a Polars DataFrame.
    
    Args:
        data: Excel file content as bytes
        sheet: Sheet name to read
        schema: Column schema mapping
        cols: Column rename mapping
        
    Returns:
        pl.DataFrame: Processed DataFrame
    '''
    
    return (
        pl
        .read_excel(
            BytesIO(data),
            sheet_name=sheet,
            columns=list(schema.keys()),
            schema_overrides=schema
        )
        .rename(mapping=cols)
    )


def _read_xlsx(
    url: str,
    sheet: str,
    schema: Dict[str, pl.DataType],
    cols: Dict[str, str],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    timeout: float = 30.0
) -> Optional[pl.DataFrame]:
    
    '''Download and read Excel file from URL.'''
        
    data = _download_with_retry(
        url,
        max_retries,
        initial_delay,
        backoff_factor,
        timeout
    )

    if data is None:
        return None

    return _read_xlsx_bytes(data, sheet, schema, cols)


# -------------------------------------------------------------------------------------------------
# Download files
# -------------------------------------------------------------------------------------------------

def _download_files(
        cfg: DownloadConfig
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    
    # Convert string schema names to Polars types
    schema_codes = {k: getattr(pl, v) for k, v in cfg.schema_codes.items()}
    schema_index = {k: getattr(pl, v) for k, v in cfg.schema_index.items()}
    schema_descriptions = {k: getattr(pl, v) for k, v in cfg.schema_descriptions.items()}
    schema_exclusions = {k: getattr(pl, v) for k, v in cfg.schema_exclusions.items()}
    
    # NAICS titles
    titles_df = (
        _read_xlsx(
            url=cfg.url_codes, 
            sheet=cfg.sheet_codes, 
            schema=schema_codes, 
            cols=cfg.rename_codes
        )
    )

    # NAICS descriptions
    descriptions_df = (
        _read_xlsx(
            url=cfg.url_descriptions,
            sheet=cfg.sheet_descriptions,
            schema=schema_descriptions,
            cols=cfg.rename_descriptions,
        )
    )

    # NAICS index file for examples
    examples_df = (
        _read_xlsx(
            url=cfg.url_index, 
            sheet=cfg.sheet_index, 
            schema=schema_index, 
            cols=cfg.rename_index
        )
    )

    # NAICS cross reference file for exclusions
    exclusions_df = (
        _read_xlsx(
            url=cfg.url_exclusions,
            sheet=cfg.sheet_exclusions,
            schema=schema_exclusions,
            cols=cfg.rename_exclusions,
        )
    )
    
    df_list = [
        ('Titles', titles_df),
        ('Descriptions', descriptions_df),
        ('Examples', examples_df), 
        ('Exclusions', exclusions_df)
    ]
    
    
    if all(isinstance(df, pl.DataFrame) for _, df in df_list):

        logger.info('Downloaded NAICS files successfully:')

        dfs = []
        for col, df in df_list:

            dfs.append(
                df
                .with_columns( # type: ignore
                    code=pl.when(pl.col('code').eq('31-33')).then(pl.lit('31'))
                           .when(pl.col('code').eq('44-45')).then(pl.lit('44'))
                           .when(pl.col('code').eq('48-49')).then(pl.lit('48'))
                           .otherwise(pl.col('code'))
                )
            )

            logger.info(f'  {col} observations: {df.height: ,}') # type: ignore

        logger.info('')

        return dfs[0], dfs[1], dfs[2], dfs[3]
    
    else:
        raise ValueError('Failed to download one or more NAICS files.')


# -------------------------------------------------------------------------------------------------
# NAICS titles
# -------------------------------------------------------------------------------------------------

def _get_titles(
    titles_df: pl.DataFrame
) -> Tuple[pl.DataFrame, Set[str]]:

    # Load NAICS titles and normalize combined sector codes (31-33, 44-45, 48-49)
    titles = (
        titles_df
        .select(
            index=pl.col('index')
                    .sub(1),
            level=pl.col('code')
                    .str.len_chars()
                    .cast(pl.UInt8),
            code=pl.col('code'),
            title=pl.col('title'),
        )
    )

    # Unique set of NAICS codes
    codes = set(
        titles
        .get_column('code')
        .unique()
        .to_list()
    )

    logger.info('Titles:')
    logger.info(f'  Number of titles: {titles.height: ,}')
    logger.info(f'  Number of codes: {len(codes): ,}\n')

    return titles, codes


# -------------------------------------------------------------------------------------------------
# NAICS descriptions 1 (still need to remove exclusions and examples)
# -------------------------------------------------------------------------------------------------

def _get_descriptions_1(descriptions_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:

    # descriptions: normalize combined sector codes
    descriptions_1 = (
        descriptions_df
        .select('code', 'description')
    )

    # Split multiline descriptions and filter out section headers and cross-references
    descriptions_2 = (
        descriptions_1
        .with_columns(
            description=pl.col('description')
                        .str.split('\r\n')
                        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        )
        .explode('description')
        .with_columns(
            description_id=pl.col('description')
                            .cum_count()
                            .over('code')
        )
        .select('code', 'description_id', 'description')
        .filter(
            (pl.col('description').ne('The Sector as a Whole')) &
            (~pl.col('description').str.contains('Cross-References.')) & 
            (pl.col('description').str.len_chars().gt(0))
        )
    )

    # Clean and normalize description text
    descriptions_3 = (
        descriptions_2
        .select(
            code=pl.col('code')
                .str.strip_chars(),
            description_id=pl.col('description_id'),
            description=pl.col('description')
                        .str.strip_prefix(' ')
                        .str.strip_suffix(' ')
                        .str.replace_all('NULL', '', literal=True)
                        .str.replace_all(r'See industry description for \d{6}\.', '')
                        .str.replace_all(r'<.*?>', '')
                        .str.replace_all(r'\xa0', ' ')
                        .str.replace_all('.', '. ', literal=True)
                        .str.replace_all('U. S. ', 'U.S.', literal=True)
                        .str.replace_all('e. g. ,', 'e.g.,', literal=True)
                        .str.replace_all('i. e. ,', 'i.e.,', literal=True)
                        .str.replace_all(';', '; ', literal=True)
                        .str.replace_all('31-33', '31', literal=True)
                        .str.replace_all('44-45', '44', literal=True)
                        .str.replace_all('48-49', '48', literal=True)
                        .str.replace_all(r'\s{2,}', ' ')
                        .str.strip_prefix(' ')
                        .str.strip_suffix(' ')
        )
    )

    logger.info('Descriptions:')
    logger.info(f'  Number: {descriptions_1.height: ,}')
    logger.info(f'  Number (split on paragraphs): {descriptions_3.height: ,}\n')

    return descriptions_2, descriptions_3


# -------------------------------------------------------------------------------------------------
# NAICS exclusions
# -------------------------------------------------------------------------------------------------

def _get_exclusions(
    exclusions_df: pl.DataFrame,
    descriptions_3: pl.DataFrame,
    codes: Set[str]
 ) -> Tuple[pl.DataFrame, pl.DataFrame]:

    # Load descriptions from cross-reference file
    exclusions_1 = (
        exclusions_df
        .filter(
            pl.col('excluded').str.contains(r' \d{2,6}'),
        )
    )

    # Aggregate exclusions by code
    exclusions_2 = (
        exclusions_1
        .group_by('code', maintain_order=True)
        .agg(
            excluded=pl.col('excluded')
        )
        .select(
            code=pl.col('code'), 
            description_id=pl.lit(1, pl.UInt32), 
            description=pl.col('excluded').list.join(' ')
        )
    )

    # Extract excluded activities (typically last description block for a code)
    exclusions_3 = (
        descriptions_3
        .filter(
            pl.col('description_id').max().over('code').eq(pl.col('description_id')),
            pl.col('description').str.contains_any(['Excluded', 'excluded', 'Exclude', 'exclude']),
            pl.col('description').str.contains(r' \d{2,6}'),
        )
        .select(
            code=pl.col('code')
                .str.strip_chars(),
            description_id=pl.col('description_id'),
            description=pl.col('description'),
        )
    )

    # Exclusions for cleaning descriptions
    descriptions_exclusions = (
        exclusions_3
        .select('code', 'description_id')
    )

    # Combine and extract excluded codes
    exclusions_4 = (
        pl
        .concat([
            exclusions_2, 
            exclusions_3
        ])
        .filter(
            pl.col('description').is_not_null()
        )
        .with_columns(
            digit=pl.col('description')
                    .str.extract_all(r' \d{2,6}')
                    .list.eval(pl.element().str.strip_prefix(' '))
                    .list.set_intersection(codes)
                    .list.drop_nulls()
        )
        .filter(
            pl.col('digit').list.len().gt(0)
        )
    )
    
    # Final exclusions DataFrame
    exclusions = (
        exclusions_4
        .explode('digit')
        .select(
            level=pl.col('code')
                    .str.len_chars()
                    .cast(pl.UInt8),
            code=pl.col('code'),
            excluded=pl.col('description'),
            excluded_codes=pl.col('digit')
        )
        .sort('level', 'code')
        .group_by('level', 'code', maintain_order=True)
        .agg(
            excluded=pl.col('excluded'),
            excluded_codes=pl.col('excluded_codes')
        )
        .with_columns(
            excluded=pl.col('excluded').list.join(' ')
        )
    )

    exclusions_cnt = (
        exclusions
        .with_columns(
            excluded_count=pl.col('excluded_codes').list.len()
        )
        .get_column('excluded_count')
        .sum()
    )

    logger.info('Exclusions:')
    logger.info('  Reference codes:')
    logger.info(f'    Cross-references: {exclusions_2.height: ,}')
    logger.info(f'    Extracted from descriptions: {exclusions_3.height: ,}')
    logger.info(f'    Final: {exclusions.height: ,}')
    logger.info(f'  Excluded codes: {exclusions_cnt: ,}\n')

    return exclusions, descriptions_exclusions


# -------------------------------------------------------------------------------------------------
# NAICS examples
# -------------------------------------------------------------------------------------------------

def _get_examples(
        examples_df: pl.DataFrame,
        codes: Set[str],
        descriptions_2: pl.DataFrame,
        descriptions_3: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]: 
    
    # Example spreadsheet
    examples_1 = (
        examples_df
        .filter(
            pl.col('code').is_in(codes)
        )
        .sort('code')
        .group_by('code', maintain_order=True)
        .agg(
            examples_1=pl.col('examples')
        )
    )

    # Identify where 'Illustrative Examples:' section begins
    examples_2 = (
        descriptions_2
        .filter(
            pl.col('description').str.contains('Illustrative Examples:')
        )
        .select(
            code=pl.col('code'), 
            example_id=pl.col('description_id')
        )
    )
    
    # Extract examples that appear after 'Illustrative Examples:' marker
    examples_3 = (
        descriptions_3
        .join(
            examples_2, 
            how='inner', 
            on='code'
        )
        .filter(
            pl.col('example_id').lt(pl.col('description_id'))
        )
        .group_by('code', maintain_order=True)
        .agg(
            examples_2=pl.col('description'),
            description_id_min=pl.col('description_id').min()
        )
    )

    # Description IDs to exclude in description dataframe
    descriptions_examples = (
        examples_3
        .select('code', 'description_id_min')
    )
    
    # Merge examples, preferring spreadsheet example 
    examples_4 = (
        examples_1
        .join(
            examples_3, 
            how='full', 
            on='code', 
            coalesce=True
        )
        .select(
            code=pl.col('code'), 
            examples=pl.coalesce('examples_1', 'examples_2')
        )
    )

    examples = (
        examples_4
        .select(
            code=pl.col('code'), 
            examples=pl.col('examples')
                       .list.join('; ')
        )
    )

    examples_cnt = (
        examples_4
        .with_columns(
            example_cnt=pl.col('examples')
                          .list.len()
        )
        .get_column('example_cnt')
        .sum()
    )

    logger.info('Examples:')
    logger.info('  Reference codes:')
    logger.info(f'    Cross-references: {examples_1.height: ,}')
    logger.info(f'    Extracted from descriptions: {examples_3.height: ,}')
    logger.info(f'    Final: {examples.height: ,}')
    logger.info(f'  Number of examples: {examples_cnt: ,}\n')

    return examples, descriptions_examples


# -------------------------------------------------------------------------------------------------
# NAICS description 2 (cleaned descriptions)
# -------------------------------------------------------------------------------------------------

def _get_descriptions_2(
    descriptions_3: pl.DataFrame,
    descriptions_exclusions: pl.DataFrame,
    descriptions_examples: pl.DataFrame
) -> pl.DataFrame:
    
    # descriptions: exclude exclusion and example description blocks
    descriptions_4 = (
        descriptions_3
        .join(
            descriptions_exclusions,
            how='anti',
            on=['code', 'description_id']
        )
        .join(
            descriptions_examples,
            how='left',
            on='code'
        )
        .with_columns(
            pl.col('description_id_min')
              .fill_null(999)
        )
        .filter(
            pl.col('description_id').lt(pl.col('description_id_min'))
        )
        .group_by('code', maintain_order=True)
        .agg(
            pl.col('description')
        )
        .with_columns(
            description=pl.col('description')
                          .list.join(' ')
        )
    )

    # Separate complete descriptions from missing ones
    description_complete_1 = (
        descriptions_4
        .filter(
            pl.col('description').ne('')
        )
    )

    # Find 4-digit codes missing descriptions
    description_4_missing = (
        descriptions_4
        .filter(
            pl.col('code').str.len_chars().eq(4), 
            pl.col('description').eq('')
        )
        .select(
            code1=pl.col('code').str.pad_end(5, '1'),
            code2=pl.col('code').str.pad_end(5, '2'),
            code3=pl.col('code').str.pad_end(5, '3'),
            code4=pl.col('code').str.pad_end(5, '4'),
            code9=pl.col('code').str.pad_end(5, '9'),
    ))

    # Find 5-digit codes missing descriptions
    description_5_missing = (
        descriptions_4
        .filter(
            pl.col('code').str.len_chars().eq(5), 
            pl.col('description').eq('')
        )
        .select(code=pl.col('code').str.pad_end(6, '0'))
    )

    logger.info('NAICS missing descriptions:')
    logger.info(f'  Total: {descriptions_4.height: ,}')
    logger.info(f'  Complete: {description_complete_1.height: ,}')
    logger.info(f'  Missing (level 4): {description_4_missing.height: ,}')
    logger.info(f'  Missing (level 5): {description_5_missing.height: ,}\n')

    # Fill missing 5-digit descriptions from 6-digit children
    description_5_complete = (
        description_5_missing
        .join(
            description_complete_1, 
            how='inner', 
            on='code'
        )
        .with_columns(
            code=pl.col('code').str.slice(0, 5)
        )
        .select(
            code=pl.col('code'),
            description=pl.col('description')
                        .str.replace('This industry', 'This NAICS industry', literal=True)
        )
    )

    description_complete_2 = (
        pl
        .concat([
            description_complete_1, 
            description_5_complete
        ])
    )

    # Fill missing 4-digit descriptions from 5-digit children (try multiple suffixes)
    description_4_complete_1 = (description_4_missing.join(
        description_complete_2, how='inner', right_on='code', left_on='code1'
    ))

    description_4_complete_2 = (
        description_4_missing
        .join(
            description_complete_2, 
            how='inner', 
            right_on='code', 
            left_on='code2'
        )
    )

    description_4_complete_3 = (
        description_4_missing
        .join(
            description_complete_2, 
            how='inner', 
            right_on='code', 
            left_on='code3'
        )
    )

    description_4_complete_4 = (
        description_4_missing
        .join(
            description_complete_2, 
            how='inner', 
            right_on='code', 
            left_on='code4'
        )
    )

    description_4_complete_9 = (
        description_4_missing
        .join(
            description_complete_2, 
            how='inner', 
            right_on='code', 
            left_on='code9'
        )
    )

    description_4_complete = (
        pl
        .concat([
            description_4_complete_1,
            description_4_complete_2,
            description_4_complete_3,
            description_4_complete_4,
            description_4_complete_9,
        ])
        .select(
            code=pl.col('code1')
                .str.slice(0, 4),
            description=pl.col('description')
                        .str.replace('This industry', 'This industry group', literal=True)
                        .str.replace('This NAICS industry', 'This industry group', literal=True),
        )
        .unique(subset=['code'])
    )

    # Combine all descriptions
    descriptions = (
        pl.concat([
            description_complete_2, 
            description_4_complete
        ])
    )

    logger.info('NAICS completed descriptions:')
    logger.info(f'  Missing (level 4): {description_4_missing.height: ,}')
    logger.info(f'  Filled missing (level 4): {description_4_complete.height: ,}')
    logger.info(f'  Missing (level 5): {description_5_missing.height: ,}')
    logger.info(f'  Filled missing (level 5): {description_5_complete.height: ,}')
    logger.info(f'  Complete: {descriptions.height: ,}')

    return descriptions


# -------------------------------------------------------------------------------------------------
# Combine all and write final output
# -------------------------------------------------------------------------------------------------

def download_preprocess_data() -> pl.DataFrame:

    # Create directories
    make_directories()
    
    # Load configuration from YAML
    cfg = load_config(DownloadConfig, 'data_generation/download.yaml')

    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
    logger.info('')
    
    titles_df, descriptions_df, examples_df, exclusions_df = _download_files(cfg)

    titles, codes = _get_titles(titles_df)

    descriptions_2, descriptions_3 = _get_descriptions_1(descriptions_df)

    exclusions, descriptions_exclusions = _get_exclusions(
        exclusions_df, 
        descriptions_3, 
        codes
    )

    examples, descriptions_examples = _get_examples(
        examples_df, 
        codes, 
        descriptions_2, 
        descriptions_3
    )

    descriptions = _get_descriptions_2(
        descriptions_3,
        descriptions_exclusions,
        descriptions_examples
    )

    # Join all components and write final output
    naics_final = (
        titles
            .join(
                descriptions, 
                how='inner', 
                on='code'
            )
            .join(
                exclusions, 
                how='left', 
                on='code'
            )
            .join(
                examples, 
                how='left', 
                on='code'
            )
            .select(
                index=pl.col('index'),
                level=pl.col('level'),
                code=pl.col('code'),
                title=pl.col('title'),
                description=pl.col('description'),
                examples=pl.col('examples'),
                excluded=pl.col('excluded'),
                excluded_codes=pl.col('excluded_codes')
            )
            .sort('index')
    )

    (
        naics_final
        .write_parquet(
            cfg.output_parquet
        )
    )    

    _parquet_stats(
        parquet_df=naics_final,
        message='NAICS codes (text + hierarchy) written to:',
        output_parquet=cfg.output_parquet,
        logger=logger
    )

    return naics_final


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    download_preprocess_data()