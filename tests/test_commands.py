"""Tests for Pivotal DSL grammar commands.

Each test creates a small DataFrame, executes a DSL snippet, and asserts
the result. The parser.execute() call runs inside a local namespace dict
so tests are fully isolated.
"""
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pivotal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return pivotal.DSLParser()


@pytest.fixture
def ns(sample_df):
    """A fresh namespace pre-loaded with the sample DataFrame."""
    return {'pd': pd, 'sales': sample_df.copy()}


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id':       [1,    2,      3,      4,       5],
        'product':  ['Laptop', 'Mouse', 'Desk', 'Chair', 'Monitor'],
        'price':    [999.99, 25.50, 299.00, 159.99, 399.00],
        'quantity': [5,    150,    20,     45,      30],
        'category': ['Electronics', 'Electronics', 'Furniture', 'Furniture', 'Electronics'],
    })


@pytest.fixture
def df_with_nulls():
    return pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': ['x', None, 'z', 'w'],
    })


@pytest.fixture
def df_with_dupes():
    return pd.DataFrame({
        'product':  ['Laptop', 'Mouse', 'Laptop', 'Chair', 'Mouse'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Furniture', 'Electronics'],
        'price':    [999.99, 25.50, 999.99, 159.99, 25.50],
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(parser, code, ns):
    """Parse and execute DSL code; return the namespace."""
    parser.execute(code, ns, verbose=False)
    return ns


# ---------------------------------------------------------------------------
# Existing commands (smoke tests to catch regressions)
# ---------------------------------------------------------------------------

def test_load_csv(parser, tmp_path, sample_df):
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, index=False)
    ns = {'pd': pd}
    run(parser, f'load df "{csv_path}"', ns)
    assert 'df' in ns
    assert list(ns['df'].columns) == list(sample_df.columns)


def test_load_parquet(parser, tmp_path, sample_df):
    pytest.importorskip('pyarrow')
    path = tmp_path / "data.parquet"
    sample_df.to_parquet(path, index=False)
    ns = {'pd': pd}
    run(parser, f'load df "{path}"', ns)
    assert 'df' in ns
    assert len(ns['df']) == len(sample_df)


def test_load_excel(parser, tmp_path, sample_df):
    pytest.importorskip('openpyxl')
    path = tmp_path / "data.xlsx"
    sample_df.to_excel(path, index=False)
    ns = {'pd': pd}
    run(parser, f'load df "{path}"', ns)
    assert 'df' in ns
    assert len(ns['df']) == len(sample_df)


def test_filter(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter price > 200', ns)
    assert all(ns['sales']['price'] > 200)


def test_select(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nselect product, price', ns)
    assert list(ns['sales'].columns) == ['product', 'price']


def test_sort(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nsort price desc', ns)
    prices = list(ns['sales']['price'])
    assert prices == sorted(prices, reverse=True)


# ---------------------------------------------------------------------------
# drop
# ---------------------------------------------------------------------------

def test_drop_single_column(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\ndrop quantity', ns)
    assert 'quantity' not in ns['sales'].columns
    assert 'price' in ns['sales'].columns


def test_drop_multiple_columns(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\ndrop quantity, id', ns)
    assert 'quantity' not in ns['sales'].columns
    assert 'id' not in ns['sales'].columns
    assert 'product' in ns['sales'].columns


# ---------------------------------------------------------------------------
# fillna
# ---------------------------------------------------------------------------

def test_fillna_numeric(parser, df_with_nulls):
    ns = {'pd': pd, 'df': df_with_nulls.copy()}
    run(parser, 'df df\nfillna 0', ns)
    assert ns['df'].isna().sum().sum() == 0
    assert ns['df'].loc[2, 'a'] == 0


def test_fillna_string(parser, df_with_nulls):
    ns = {'pd': pd, 'df': df_with_nulls.copy()}
    run(parser, 'df df\nfillna "unknown"', ns)
    assert ns['df'].isna().sum().sum() == 0
    assert ns['df'].loc[1, 'b'] == 'unknown'


# ---------------------------------------------------------------------------
# dropna
# ---------------------------------------------------------------------------

def test_dropna_all(parser, df_with_nulls):
    ns = {'pd': pd, 'df': df_with_nulls.copy()}
    run(parser, 'df df\ndropna', ns)
    assert ns['df'].isna().sum().sum() == 0
    assert len(ns['df']) == 2  # rows 0 and 3 have no nulls


def test_dropna_subset(parser, df_with_nulls):
    ns = {'pd': pd, 'df': df_with_nulls.copy()}
    # Only drop rows where column 'a' is null (row 2)
    run(parser, 'df df\ndropna a', ns)
    assert len(ns['df']) == 3
    assert 2 not in ns['df'].index


# ---------------------------------------------------------------------------
# distinct
# ---------------------------------------------------------------------------

def test_distinct_all_columns(parser, df_with_dupes):
    ns = {'pd': pd, 'df': df_with_dupes.copy()}
    run(parser, 'df df\ndistinct', ns)
    assert len(ns['df']) == 3  # 2 exact duplicate rows removed


def test_distinct_subset(parser, df_with_dupes):
    ns = {'pd': pd, 'df': df_with_dupes.copy()}
    run(parser, 'df df\ndistinct product', ns)
    assert len(ns['df']) == 3  # Laptop, Mouse, Chair


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------

def test_rename_single(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nrename price as cost', ns)
    assert 'cost' in ns['sales'].columns
    assert 'price' not in ns['sales'].columns


def test_rename_multiple(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nrename product as item, quantity as qty', ns)
    assert 'item' in ns['sales'].columns
    assert 'qty' in ns['sales'].columns
    assert 'product' not in ns['sales'].columns
    assert 'quantity' not in ns['sales'].columns


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------

def test_concat(parser, sample_df):
    half1 = sample_df.iloc[:2].copy()
    half2 = sample_df.iloc[2:].copy()
    ns = {'pd': pd, 'half1': half1, 'half2': half2}
    run(parser, 'df half1\nconcat half2', ns)
    assert len(ns['half1']) == len(sample_df)
    assert list(ns['half1'].reset_index(drop=True)['product']) == list(sample_df['product'])


def test_concat_multiple(parser, sample_df):
    part1 = sample_df.iloc[:1].copy()
    part2 = sample_df.iloc[1:2].copy()
    part3 = sample_df.iloc[2:3].copy()
    ns = {'pd': pd, 'part1': part1, 'part2': part2, 'part3': part3}
    run(parser, 'df part1\nconcat part2, part3', ns)
    assert len(ns['part1']) == 3


# ---------------------------------------------------------------------------
# filter: between
# ---------------------------------------------------------------------------

def test_filter_between(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter price between [100, 400]', ns)
    assert all(ns['sales']['price'] >= 100)
    assert all(ns['sales']['price'] <= 400)
    # Desk 299, Chair 159.99, Monitor 399 = 3 rows
    assert len(ns['sales']) == 3


def test_filter_between_combined(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    # Between 100–350 AND Furniture → Desk 299, Chair 159.99
    run(parser, 'df sales\nfilter price between [100, 350] and category == "Furniture"', ns)
    assert len(ns['sales']) == 2
    assert all(ns['sales']['category'] == 'Furniture')


# ---------------------------------------------------------------------------
# filter: string methods
# ---------------------------------------------------------------------------

def test_filter_contains(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter product contains "op"', ns)
    # "Laptop" contains "op"
    assert len(ns['sales']) == 1
    assert ns['sales'].iloc[0]['product'] == 'Laptop'


def test_filter_not_contains(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter category not contains "Furniture"', ns)
    assert all(ns['sales']['category'] == 'Electronics')


def test_filter_startswith(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter product startswith "Mo"', ns)
    assert len(ns['sales']) == 2  # Mouse, Monitor
    assert all(ns['sales']['product'].str.startswith('Mo'))


def test_filter_endswith(parser, sample_df):
    ns = {'pd': pd, 'sales': sample_df.copy()}
    run(parser, 'df sales\nfilter product endswith "r"', ns)
    assert len(ns['sales']) == 2  # Chair, Monitor


# ---------------------------------------------------------------------------
# load: runtime variable path (format detection)
# ---------------------------------------------------------------------------

def test_load_variable_csv(parser, tmp_path, sample_df):
    csv_path = str(tmp_path / "data.csv")
    sample_df.to_csv(csv_path, index=False)
    ns = {'pd': pd, 'my_path': csv_path}
    run(parser, 'load df :my_path', ns)
    assert 'df' in ns
    assert len(ns['df']) == len(sample_df)
